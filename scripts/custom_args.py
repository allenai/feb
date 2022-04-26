'''
This code is based on https://github.com/allenai/label_rationale_association/blob/main/custom_args.py
'''
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_type: str = field(
        metadata={
            "help": "HF pretrained model"
        },
    )
    pretrained: bool = field(
        default=True,
        metadata={
            "help": "Pass a boolean value (false) if want to initialize the model from scratch rather than use pretrained weights"
        },
    )
    rationale_only: bool = field(
        default=False,
        metadata={
            "help": "Only produce rationales and not labels (first model in pipeline)"
        },
    )
    train_on_model_predictions: bool = field(
        default=False,
        metadata={
            "help": "Flag to train on model predictions instead of gold explanations"
        },
    )
    use_dev_real_expls: bool = field(
        default=False,
        metadata={
            "help": "Use this flag for test case where we want to test on gold-label predictions rather than generations (e.g. sufficiency sanity check experiment)"
        },
    )
    #pretrained_model_file: Optional[str] = field(
    pretrained_model_file: str = field(
        default=None,
        metadata={
            "help": "Pass a pretrained model save_path to re-load for evaluation"
        },
    )
    #predictions_model_file: Optional[str] = field(
    predictions_model_file: str = field(
        default=None,
        metadata={
            "help": "Pass a file where can find predictions from generation model for the dev set (first model in pipeline)"
        },
    )
    #config_name: Optional[str] = field(
    config_name: str = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    #tokenizer_name: Optional[str] = field(
    tokenizer_name: str = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    #cache_dir: Optional[str] = field(
    cache_dir: str = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )

    #dropout_rate: Optional[float] = field(
    dropout_rate: float = field(
        default=None,
        metadata={
           "help": "Specify a dropout rate, if don't want to use default in transformers/configuration_t5.py"
        }
    )
    use_gpt3: bool = field(
        default=False,
        metadata={
            "help": "Ignore the HF model and use gpt3. Uses the `--generations_filepath` to name a file for gpt3 predictions."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on."})
    early_stopping_patience: int = field(
        default=10,
        metadata={"help": "The number of patience epochs for early stopping."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    train_predict: bool = field(
        default=False, metadata={"help": "Predict continuations for train set and save"}
    )
    test_predict: bool = field(
        default=False, metadata={"help": "Predict continuations for test set and save"}
    )
    dev_predict: bool = field(
        default=False, metadata={"help": "Predict continuations for dev set and save"}
    )
    #version_name: Optional[str] = field(
    version_name: str = field(
        default="v1.11", metadata={"help": "Version of CoS-E to load"}
    )
    #generations_filepath: Optional[str] = field(
    generations_filepath: str = field(
        default=None,
        metadata={"help": "Path to pre-generated model generations for evaluation"},
    )
    n_shots: int = field(
        default=None,
        metadata={"help": "Number of examples per label for fewshot learning"}
    )
    fewshot_eval_size: float = field(
        default=350,
        metadata={"help": "The size of test set for fewshot learning"}
    )
    io_format: str = field(
        default='standard',
        metadata={"help": "The input-output format for finetuning: standard or masked_cause_generate"}
    )
    explanation_sep: str = field(
        default='explanation',
        metadata={"help": "A token that separates answer tokens from explanation tokens in the output"}
    )
    data_path: str = field(
        default=None,
        metadata={"help": "path to sbic or senmaking folder"}
    )
    gpt3_max_eval_size: int = field(
        default=None,
        metadata={"help": "Test set size for gpt3. Should be <= fewshot_eval_size. Both args are needed " \
                  "to sample the same `fewshot_eval_size` examples but then limit gpt3 evaluation to the first " \
                  "`gpt3_max_eval_size`. This is important or human evaluation where we want to maintain the same "\
                  "evaluation samples for all models"}
    )
