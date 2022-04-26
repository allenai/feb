# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.

Fine-tunes the model to jointly produce labels + rationales.
Modified from (transformers version 2.9.1):
https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_language_modeling.py
"""

# This code is based on https://github.com/allenai/label_rationale_association/blob/main/input_to_label_and_rationale.py

import gpt3
import logging
import math
import os
from typing import List, Dict, Any, NewType

InputDataClass = NewType("InputDataClass", Any)

from transformers import (
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback
)
from transformers.trainer_utils import EvaluationStrategy
from transformers.integrations import TensorBoardCallback
import transformers
from transformers import Trainer

from feature_conversion_methods import format_instance

from custom_args import (
    DataTrainingArguments,
    ModelArguments
)
from metrics import evaluate
import torch
import datasets
import git
import time
from datetime import datetime
import sys
from tqdm import trange
import random 
import pandas as pd 
import jsonlines
from copy import deepcopy 

logger = logging.getLogger(__name__)
transformers.logging.set_verbosity_info()
import re
def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)
set_global_logging_level(logging.ERROR, ["datasets"])


CONFIG_MAPPING = {"t5": T5Config}
MODEL_MAPPING = {"t5": T5ForConditionalGeneration}
TOKENIZER_MAPPING = {"t5": T5Tokenizer}


def set_other_seeds(seed):
    torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

# inspired by DefaultDataCollator from:
# https://github.com/huggingface/transformers/blob/master/src/transformers/data/data_collator.py
# modified to perform batch-level padding.
class SequenceCollator:
    def __init__(self, model, pad_token):
        self.model = model
        self.pad_token_mapping = {
            "labels": -100,
            "attention_mask": 0,
            "decoder_attention_mask": 0,
            "input_ids": pad_token,
        }

        self.columns = [
            "input_ids",
            "attention_mask",
            "labels",
            "decoder_attention_mask",
        ]

    def __call__(self, examples: List[Dict[str, InputDataClass]]) -> Dict[str, torch.Tensor]:
        # re-format inputs for training
        batch = {}
        for key in examples[0].keys():
            if key in self.columns:
                tmp_list = []
                for item in examples:
                    tmp_list.append(item[key])

                # pad lists to max length
                if isinstance(tmp_list[0], list):
                    max_length = max(map(len, tmp_list))
                    tmp_list = [
                        el + [self.pad_token_mapping[key]] * (max_length - len(el))
                        for el in tmp_list
                    ]

                batch[key] = torch.tensor(tmp_list, dtype=torch.long)
        return batch


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    og_start_time = time.time()

    #parser = HfArgumentParser(
    #    (ModelArguments, DataTrainingArguments, TrainingArguments)
    #)
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    model_args, data_args, training_args, unused_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    if unused_args != []:
        raise ValueError(f"Received unused arguments: {unused_args}")
    # make sure only one dataset split pick if manually specifying evaluation file

    if model_args.use_gpt3:
        assert training_args.do_train
        assert not training_args.do_eval
        assert data_args.generations_filepath is None
        if data_args.gpt3_max_eval_size is not None:
            assert data_args.gpt3_max_eval_size <= data_args.fewshot_eval_size
            assert data_args.gpt3_max_eval_size % 2 == 0
            assert data_args.gpt3_max_eval_size % 3 == 0

    if data_args.generations_filepath is not None:
        training_args.do_train = False
        training_args.do_eval = False
        if "train" in data_args.generations_filepath:
            data_args.train_predict = True
            data_args.test_predict = False
            data_args.dev_predict = False
        elif "test" in data_args.generations_filepath:
            data_args.train_predict = False
            data_args.test_predict = True
            data_args.dev_predict = False
        elif "validation" in data_args.generations_filepath:
            data_args.train_predict = False
            data_args.test_predict = False
            data_args.dev_predict = True

    if not training_args.do_train and data_args.generations_filepath is None:
        if not model_args.pretrained_model_file:
            raise Exception(
                "if not training a model from scratch, must specify a trained model to load for evaluation"
            )

    if training_args.do_train:
        # create a save directory and a logfile
        training_args.output_dir = os.path.join(
            training_args.output_dir, datetime.now().strftime("%m%d%y_%H%M%S")
        )
        training_args.logging_dir = training_args.output_dir
        assert not os.path.exists(training_args.output_dir)
        os.makedirs(training_args.output_dir)

        if (
                os.path.exists(training_args.output_dir)
                and os.listdir(training_args.output_dir)
                and training_args.do_train
                and not training_args.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )
        handlers = [
            logging.FileHandler(os.path.join(training_args.output_dir, "logger.log")),
            logging.StreamHandler(),
        ]
    else:
        # don't overwrite existing logfile or create new directory
        training_args.output_dir = model_args.pretrained_model_file
        handlers = [logging.StreamHandler()]

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=handlers,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Save path: %s" % training_args.output_dir)

    # get git hash and branch where deployed
    repo = git.Repo(search_parent_directories=True)
    git_hash = repo.head.object.hexsha
    git_branch = repo.active_branch.name
    logger.info("Git branch: %s" % git_branch)
    logger.info("Git hash: %s" % git_hash)

    model_class = "t5"
    assert data_args.task_name in {"cos_e", "esnli", "sbic", "sensemaking", "ecqa"}

    if training_args.do_train:
        # write command and args to file
        with open(
                os.path.join(training_args.output_dir, "commandline_args.txt"), "w"
        ) as f:
            f.write("Git branch: " + git_branch + "\n")
            f.write("Git hash: " + git_hash + "\n")
            f.write("Command:\n")
            f.write("\n".join(sys.argv[1:]))

    # Set seed
    set_seed(training_args.seed)
    set_other_seeds(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    tokenizer_name = TOKENIZER_MAPPING[model_class]
    logger.info("Loading pretrained tokenizer...")

    tokenizer = tokenizer_name.from_pretrained(model_args.tokenizer_name)#, cache_dir=model_args.cache_dir)
    if data_args.generations_filepath is None:
        model_name = MODEL_MAPPING[model_class]
        if model_args.pretrained_model_file:
            model = T5ForConditionalGeneration.from_pretrained(model_args.pretrained_model_file)

            if model_args.dropout_rate:
                raise Exception("can't update/specify dropout currently when load pretrained model from directory")

        elif model_args.pretrained:
            # load pretrained model from HuggingFace
            logger.info("Loading pretrained model")
            if model_args.dropout_rate:
                model = model_name.from_pretrained(model_args.model_type, dropout_rate=model_args.dropout_rate)
            else:
                model = model_name.from_pretrained(model_args.model_type)
        else:
            # load model from scratch with no pretrained weights
            config_name = CONFIG_MAPPING[model_class]()
            # TODO (Sarah): NOTE THIS ONLY DOES T5-BASE; PASS IN ARGS HERE^
            logger.info(
                "Training new model from scratch using default config (NOTE: SMALL MODELS ONLY FOR NOW)"
            )
            if model_args.dropout_rate:
                raise Exception("sure you want to train a model from scratch?")
            model = model_name.from_config(config_name)
        model.resize_token_embeddings(len(tokenizer))
    else:
        model = None

    data_splits = {'train': None, 'validation': None, 'test': None}
    original_data_splits = {'train': None, 'validation': None, 'test': None}  

    # Data loading from huggingface's datasets
    if data_args.task_name in {"cos_e", "esnli"}:
        version_arg = None
        if data_args.task_name == "cos_e":
            assert data_args.version_name in {"v1.11", "v1.0"}
            version_arg = data_args.version_name

        load_train = True
        if (not training_args.do_train
            and not training_args.do_eval
            and not data_args.train_predict
        ):
            # don't load training dataset
            dataset = {}
            dataset["train"] = None
            dataset["validation"] = datasets.load_dataset(
                data_args.task_name, version_arg, split="validation"
            )
            data_splits['validation'] = dataset["validation"]

            if data_args.task_name == "esnli":
                dataset["test"] = datasets.load_dataset(data_args.task_name, split="test")
                data_splits['test'] = dataset["test"]
            load_train = False
        else:
            dataset = datasets.load_dataset(data_args.task_name, version_arg)

            if data_args.n_shots > 0: # Shots = number of training examples **per label** 
                if data_args.task_name == 'esnli': # Construct a *balanced* random sample of the size `data_args.n_shots*len(labels)` (for train) or `data_args.fewshot_eval_size` (for eval)
                    for split in ["train", "validation", "test"]:
                        split_data = dataset[split]
                        label_subsets = []
                        labels = split_data.features['label'].names
                        sample_size = data_args.n_shots if split == "train" else int(data_args.fewshot_eval_size/len(labels))
                        if data_args.gpt3_max_eval_size is not None and split != 'train':
                            assert len(labels) == 3
                            sample_size = data_args.gpt3_max_eval_size // len(labels)
                        for label in labels:
                            # The following is a hack to only run on `neutral` labels of `esnli` to get data for human eval
                            # if data_args.gpt3_max_eval_size is not None and split != 'train' and label != 'neutral':
                            #     continue
                            label_int = split_data.features['label'].str2int(label)
                            label_set = split_data.filter(lambda example: example['label'] == label_int).shuffle() # all instances of labeled as `label`
                            label_subset = label_set.select(range(sample_size)) #select `sample_size` random instances labeled as `label`
                            label_subsets.append(label_subset)
                        dataset[split] = datasets.concatenate_datasets(label_subsets) #merge all label-specific instances
                elif data_args.task_name == 'cos_e': 
                    for split in ["train", "validation"]: 
                        split_data = dataset[split]
                        sample_size = data_args.n_shots if split == "train" else int(data_args.fewshot_eval_size) #Shots for QA are not label-specific, i.e., `n_shots` is the training data size
                        if data_args.gpt3_max_eval_size is not None and split != 'train':
                            sample_size = data_args.gpt3_max_eval_size
                        dataset[split] = split_data.shuffle().select(range(sample_size)) # select `sample_size` random instances
                else: 
                    raise ValueError('Only cos_e and esnli are supported by Huggingface datasets.')
        # Apply method, and format dataset to torch.Tensor outputs
        for split in dataset.keys():
            if dataset[split] is not None:
                dataset[split] = dataset[split].map(
                    lambda x: format_instance(
                        x,
                        tokenizer,
                        data_args.explanation_sep,
                        datasource=data_args.task_name,
                        io_format=data_args.io_format
                    ),
                    batched=False,
                    load_from_cache_file=False,
                )
        data_splits["train"] = deepcopy(dataset["train"])
        data_splits["validation"] = deepcopy(dataset["validation"])
        if data_args.task_name == "esnli":
            data_splits["test"] = deepcopy(dataset["test"])

        original_data_splits["train"] = deepcopy(dataset["train"])
        original_data_splits["validation"] = deepcopy(dataset["validation"])
        if data_args.task_name == "esnli":
            original_data_splits["test"] = deepcopy(dataset["test"])


    elif data_args.task_name == "sbic":
        split_mapping = {'trn': 'train', 'dev': 'validation', 'tst': 'test'}
        splits = ['trn', 'dev', 'tst'] if training_args.do_train else ['dev', 'tst']
        load_train = True if training_args.do_train else False 
        n_labels = 2 # two labels: offensive, not offensive 
        for split in splits:
            data_splits[split_mapping[split]] = []
            if not training_args.do_train:
                continue
            data_path = os.path.join(os.getcwd(), data_args.data_path, f"SBIC.v2.{split}.modified.csv")
            df = pd.read_csv(data_path)

            if data_args.n_shots > 0: # This condition could probably be removed; we used n_shots=0 to experiment with training with the entire train set
                # Here we create a balanced training set with `data_args.n_shots` examples per label
                not_offensive_df = df.loc[df["offensiveYN"]=="not offensive"]
                frac1 = data_args.n_shots/len(not_offensive_df) if split == 'trn' else int(data_args.fewshot_eval_size/n_labels)/len(not_offensive_df)
                offensive_df = df.loc[df["offensiveYN"]=="offensive"]
                frac2 = data_args.n_shots/len(offensive_df) if split == 'trn' else int(data_args.fewshot_eval_size/n_labels)/len(offensive_df)
                label1_data = not_offensive_df.sample(frac=frac1, replace=False)
                label2_data = offensive_df.sample(frac=frac2, replace=False)
                if data_args.gpt3_max_eval_size is not None and split != 'trn':
                    label1_data = label1_data[:data_args.gpt3_max_eval_size // 2]
                    label2_data = label2_data[:data_args.gpt3_max_eval_size // 2]
                df = pd.concat([label1_data, label2_data])

            for i in trange(len(df["targetStereotype"])):
                new_encoded = format_instance(
                    df.iloc[i],
                    tokenizer,
                    data_args.explanation_sep,
                    datasource=data_args.task_name,
                    io_format=data_args.io_format
                )
                data_splits[split_mapping[split]].append({**df.iloc[i], **new_encoded})
            original_data_splits[split_mapping[split]] = deepcopy(data_splits[split_mapping[split]])
    

    elif data_args.task_name == "sensemaking":
        split_mapping = {'Training': 'train', 'Dev': 'validation', 'Test': 'test'}
        splits = ['Training', 'Dev', 'Test'] if training_args.do_train else ['Dev', 'Test']
        load_train = True if training_args.do_train else False 
        n_labels = 2 # two labels: choice1, choice2
        for split in splits:
            data_splits[split_mapping[split]] = []
            if not training_args.do_train:
                continue
            data_path = os.path.join(os.getcwd(), data_args.data_path, f"SenMaking.{split}.csv")
            df = pd.read_csv(data_path)

            if data_args.n_shots > 0: # This condition could probably be removed; we used n_shots=0 to experiment with training with the entire train set
                # Here we create a balanced training set with `data_args.n_shots` examples per label
                choice1_df = df.loc[df["label"]==0]
                frac1 = data_args.n_shots/len(choice1_df) if split == 'Training' else int(data_args.fewshot_eval_size/n_labels)/len(choice1_df)
                choice2_df = df.loc[df["label"]==1]
                frac2 = data_args.n_shots/len(choice2_df) if split == 'Training' else int(data_args.fewshot_eval_size/n_labels)/len(choice2_df)
                label1_data = choice1_df.sample(frac=frac1, replace=False)
                label2_data = choice2_df.sample(frac=frac2, replace=False)
                if data_args.gpt3_max_eval_size is not None and split != 'Training':
                    label1_data = label1_data[:data_args.gpt3_max_eval_size // 2]
                    label2_data = label2_data[:data_args.gpt3_max_eval_size // 2]
                df = pd.concat([label1_data, label2_data])
            
            for i in trange(len(df)):
                new_encoded = format_instance(
                    df.iloc[i],
                    tokenizer,
                    data_args.explanation_sep,
                    datasource=data_args.task_name,
                    io_format=data_args.io_format
                )
                data_splits[split_mapping[split]].append({**df.iloc[i], **new_encoded})
            original_data_splits[split_mapping[split]] = deepcopy(data_splits[split_mapping[split]])

    elif data_args.task_name == 'ecqa': 
        for split in ["train", "validation"]: 
            ecqa_data_split = []
            data_path = os.path.join(os.getcwd(), data_args.data_path, f"ecqa_{split}.jsonl")
            with jsonlines.open(data_path) as ecqa_split_reader:
                for item in ecqa_split_reader: 
                    formatted_instance = format_instance(item,
                                                         tokenizer,
                                                         data_args.explanation_sep,
                                                         datasource=data_args.task_name,
                                                         io_format=data_args.io_format)
                    ecqa_data_split.append(formatted_instance)
            sample_size = data_args.n_shots if split == "train" else int(data_args.fewshot_eval_size)
            if data_args.gpt3_max_eval_size is not None and split != 'train':
                sample_size = data_args.gpt3_max_eval_size
            data_splits[split] = random.sample(ecqa_data_split, sample_size)
            original_data_splits[split] = deepcopy(data_splits[split])
    else: 
        raise ValueError("Unknown task. Currently supported: esnli, cos_e, sbic, sensemaking, ecqa.")

    logger.info("****LOG****")
    for split in ['train', 'validation', 'test']:
        if data_splits[split]:
            logger.info(split)
            logger.info(len(data_splits[split]))

    
    # I did this to do some manual checks, but we don't need it
    # TODO (Ana): remove this
    '''
    if data_args.n_shots > 0: 
        import jsonlines 
        with jsonlines.open(os.path.join(training_args.output_dir,'train.json'), 'w') as writer:
            for item in original_data_splits['train']:
                writer.write(item)
        with jsonlines.open(os.path.join(training_args.output_dir,'validation.json'), 'w') as writer:
            for item in original_data_splits['validation']:
                writer.write(item)
    '''

    if data_args.generations_filepath is None:
        callbacks = [TensorBoardCallback()]
        if data_args.early_stopping_patience > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=data_args.early_stopping_patience))
            training_args.load_best_model_at_end = True
        else:
            training_args.load_best_model_at_end = False  # use the last model state
        training_args.metric_for_best_model = 'eval_loss'
        training_args.greater_is_better = False
        if training_args.eval_steps is None:
            training_args.evaluation_strategy = EvaluationStrategy.EPOCH
        else:
            training_args.evaluation_strategy = EvaluationStrategy.STEPS

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=data_splits['train'],
            eval_dataset=data_splits['validation'],
            data_collator=SequenceCollator(
                model=model_class, pad_token=tokenizer.pad_token_id
            ),
            callbacks=callbacks,
        )

    # Training. Don't train if it is use_gpt3
    if training_args.do_train and not model_args.use_gpt3:
        start_time = time.time()
        trainer.train()
        train_time = time.time() - start_time
        model = trainer.model
    else:
        start_time = time.time()
        train_time = time.time() - start_time

    # Evaluation
    results = {}
    if training_args.do_eval:
        start_time = time.time()
        logger.info("*** Evaluate on train set***")
        logger.info(len(data_splits['train']))
        train_output = trainer.evaluate(data_splits['train'])
        perplexity = math.exp(train_output["eval_loss"])
        results["perplexity_train"] = perplexity

        # repeat
        logger.info("*** Evaluate on dev set***")
        logger.info(len(data_splits['validation']))
        eval_output = trainer.evaluate(data_splits['validation'])
        perplexity = math.exp(eval_output["eval_loss"])
        results["perplexity_validation"] = perplexity

        if data_args.task_name in {"esnli", "sbic"}:
            # also evaluate on test
            logger.info("*** Evaluate on test set***")
            logger.info(len(data_splits["test"]))
            test_output = trainer.evaluate(data_splits["test"])
            logger.info("test loss @ best dev epoch: %0.4f" % test_output["eval_loss"])
            perplexity = math.exp(test_output["eval_loss"])
            results["perplexity_test"] = perplexity

        eval_time = time.time() - start_time

    if data_args.generations_filepath is None:
        # get folder where to save predictions
        save_path = trainer.state.best_model_checkpoint
        if save_path is None:  # early stopping is disabled, no checkpoints saved
            save_path = training_args.output_dir
        model.eval()
    else:
        save_path = os.path.dirname(data_args.generations_filepath)

    start_time = time.time()
    # Storing predictions & computing BLEUscore. Don't predict on the training set if `use_gpt3`
    # `data_args.train_predict` is NOT used for experiments in the paper
    if data_args.train_predict and not model_args.use_gpt3:
        logger.info("*** Predict on train set***")
        if data_args.generations_filepath is not None:
            assert "train" in data_args.generations_filepath
        
        results = evaluate(
                            save_path,
                            original_data_splits['train'],
                            model,
                            tokenizer,
                            "train",
                            data_args.task_name,
                            training_args.device,
                            data_args.explanation_sep,
                            rationale_only=model_args.rationale_only,
                            generations_file=data_args.generations_filepath,
                            io_format=data_args.io_format
                            )
    # `data_args.test_predict` is NOT used for experiments in the paper
    if data_args.test_predict and data_args.task_name in {"esnli", "sbic"}:
        logger.info("*** Predict on test set***")
        if model_args.use_gpt3:
            # get gpt3 predictions and write them to `data_args.generations_filepath` then pass it to `evaluate`
            data_args.generations_filepath = os.path.join(save_path, "gpt3_test_generations.txt")
            gpt3.run_gpt3(
                train_data=data_splits["train"], test_data=original_data_splits["test"],
                task=data_args.task_name, generations_file=data_args.generations_filepath,
                explanation_sep=data_args.explanation_sep,
                save_path=save_path)

        if data_args.generations_filepath is not None:
            assert "test" in data_args.generations_filepath

        results = evaluate(
                            save_path,
                            original_data_splits['test'],
                            model,
                            tokenizer,
                            "test",
                            data_args.task_name,
                            training_args.device,
                            data_args.explanation_sep,
                            rationale_only=model_args.rationale_only,
                            generations_file=data_args.generations_filepath,
                            io_format=data_args.io_format
                            )
    # `data_args.dev_predict` is used for ALL experiments in the paper
    if data_args.dev_predict:
        logger.info("*** Predict on dev set***")
        if model_args.use_gpt3:
            # get gpt3 predictions and write them to `data_args.generations_filepath` then pass it to `evaluate`
            data_args.generations_filepath = os.path.join(save_path, "gpt3_validation_generations.txt")
            gpt3.run_gpt3(
                train_data=data_splits["train"], test_data=original_data_splits["validation"],
                task=data_args.task_name, generations_file=data_args.generations_filepath,
                explanation_sep=data_args.explanation_sep,
                save_path=save_path)

        if data_args.generations_filepath is not None:
            assert "validation" in data_args.generations_filepath

        if model_args.pretrained_model_file and not training_args.do_train:
            save_path = model_args.pretrained_model_file

        results = evaluate(
                            save_path,
                            original_data_splits["validation"],
                            model,
                            tokenizer,
                            "validation",
                            data_args.task_name,
                            training_args.device,
                            data_args.explanation_sep,
                            rationale_only=model_args.rationale_only,
                            generations_file=data_args.generations_filepath,
                            io_format=data_args.io_format
                        )

    if data_args.generations_filepath is None:
        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
    else:
        output_eval_file = os.path.join(
            os.path.dirname(os.path.dirname(data_args.generations_filepath)),
            "eval_results_lm.txt",
        )
    if data_args.generations_filepath or trainer.is_world_process_zero():
        with open(output_eval_file, "a+") as writer:
            #for key in sorted(results.keys()):
            for key in results.keys():
                if results[key] is not None:
                    logger.info("  %s = %s", key, str(results[key])) # logging this is important for collecting results later
                    writer.write("%s = %s\n" % (key, str(results[key])))

    predict_time = time.time() - start_time

    # final logs
    logger.info("Git branch: %s" % git_branch)
    logger.info("Git hash: %s" % git_hash)
    logger.info("Save path: %s" % training_args.output_dir)
    if training_args.do_train:
        logger.info("total train time: %.4f hours" % (train_time / 60.0 / 60.0))
    if training_args.do_eval:
        logger.info("total eval time: %.4f hours" % (eval_time / 60.0 / 60.0))
    if (
            data_args.train_predict
            or data_args.dev_predict
            or (data_args.test_predict and data_args.task_name in {"esnli", "sbic"})
    ):
        logger.info("total predict time: %.4f hours" % (predict_time / 60.0 / 60.0))
    logger.info(
        "TOTAL SCRIPT TIME: %.4f hours" % ((time.time() - og_start_time) / 60.0 / 60.0)
    )


if __name__ == "__main__":
    main()
