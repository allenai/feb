import datasets
from transformers import set_seed
from input_to_label_and_rationale import set_other_seeds
from copy import deepcopy
import pandas as pd 
import os 
import random 
import jsonlines 
import os 

def make_splits(seed, task_name, data_path, n_shots=48, fewshot_eval_size=350, gpt3_max_eval_size=18):
    set_seed(seed)
    set_other_seeds(seed)

    original_data_splits = {'train': None, 'validation': None, 'test': None}  
    data_splits = {'train': None, 'validation': None, 'test': None}

    if task_name == "esnli":
        dataset = datasets.load_dataset(task_name)

        # Construct a *balanced* random sample of the size `n_shots*len(labels)` (for train) or `fewshot_eval_size` (for eval)
        for split in ["train", "validation", "test"]:
            split_data = dataset[split]
            label_subsets = []
            labels = split_data.features['label'].names
            sample_size = n_shots if split == "train" else int(fewshot_eval_size/len(labels))
            if gpt3_max_eval_size is not None and split != 'train':
                assert len(labels) == 3
                sample_size = gpt3_max_eval_size // len(labels)
            for label in labels:
                # The following is a hack to only run on `neutral` labels of `esnli` to get data for human eval
                # if data_args.gpt3_max_eval_size is not None and split != 'train' and label != 'neutral':
                #     continue
                label_int = split_data.features['label'].str2int(label)
                label_set = split_data.filter(lambda example: example['label'] == label_int).shuffle() # all instances of labeled as `label`
                label_subset = label_set.select(range(sample_size)) #select `sample_size` random instances labeled as `label`
                label_subsets.append(label_subset)
            dataset[split] = datasets.concatenate_datasets(label_subsets) #merge all label-specific instances

        original_data_splits["train"] = deepcopy(dataset["train"])
        original_data_splits["validation"] = deepcopy(dataset["validation"])
        original_data_splits["test"] = deepcopy(dataset["test"])


    elif task_name == "sbic":
        split_mapping = {'trn': 'train', 'dev': 'validation', 'tst': 'test'}
        splits = ['trn', 'dev', 'tst'] 
        n_labels = 2 # two labels: offensive, not offensive 
        for split in splits:
            data_splits[split_mapping[split]] = []

            df = pd.read_csv(f"{data_path}SBIC.v2.{split}.modified.csv")

            # Here we create a balanced training set with `data_args.n_shots` examples per label
            not_offensive_df = df.loc[df["offensiveYN"]=="not offensive"]
            frac1 = n_shots/len(not_offensive_df) if split == 'trn' else int(fewshot_eval_size/n_labels)/len(not_offensive_df)
            offensive_df = df.loc[df["offensiveYN"]=="offensive"]
            frac2 = n_shots/len(offensive_df) if split == 'trn' else int(fewshot_eval_size/n_labels)/len(offensive_df)
            label1_data = not_offensive_df.sample(frac=frac1, replace=False)
            label2_data = offensive_df.sample(frac=frac2, replace=False)
            if gpt3_max_eval_size is not None and split != 'trn':
                label1_data = label1_data[:gpt3_max_eval_size // 2]
                label2_data = label2_data[:gpt3_max_eval_size // 2]
            df = pd.concat([label1_data, label2_data])

            data_splits[split_mapping[split]] = [df.iloc[i].to_dict() for i in range(len(df["targetStereotype"]))]
            original_data_splits[split_mapping[split]] = deepcopy(data_splits[split_mapping[split]])

    elif  task_name == "sensemaking":
        split_mapping = {'Training': 'train', 'Dev': 'validation', 'Test': 'test'}
        splits = ['Training', 'Dev', 'Test'] 
        n_labels = 2 # two labels: choice1, choice2
        for split in splits:
            data_splits[split_mapping[split]] = []
            df = pd.read_csv(f"{data_path}SenMaking.{split}.csv")

            # Here we create a balanced training set with `data_args.n_shots` examples per label
            choice1_df = df.loc[df["label"]==0]
            frac1 = n_shots/len(choice1_df) if split == 'Training' else int(fewshot_eval_size/n_labels)/len(choice1_df)
            choice2_df = df.loc[df["label"]==1]
            frac2 = n_shots/len(choice2_df) if split == 'Training' else int(fewshot_eval_size/n_labels)/len(choice2_df)
            label1_data = choice1_df.sample(frac=frac1, replace=False)
            label2_data = choice2_df.sample(frac=frac2, replace=False)
            if gpt3_max_eval_size is not None and split != 'Training':
                label1_data = label1_data[:gpt3_max_eval_size // 2]
                label2_data = label2_data[:gpt3_max_eval_size // 2]
            df = pd.concat([label1_data, label2_data])

            data_splits[split_mapping[split]] = [df.iloc[i].to_dict() for i in range(len(df))]
            original_data_splits[split_mapping[split]] = deepcopy(data_splits[split_mapping[split]])

    elif task_name == 'ecqa': 
        for split in ["train", "validation"]: 
            ecqa_data_split = []
            with jsonlines.open(os.path.join(data_path, f"ecqa_{split}.jsonl")) as ecqa_split_reader:
                ecqa_data_split = [item for item in ecqa_split_reader] 
            sample_size = n_shots if split == "train" else int(fewshot_eval_size)
            if gpt3_max_eval_size is not None and split != 'train':
                sample_size = gpt3_max_eval_size
            data_splits[split] = random.sample(ecqa_data_split, sample_size)
        original_data_splits[split] = deepcopy(data_splits[split])
    else: 
        raise ValueError("Unknown task. Currently supported: esnli, sbic, sensemaking, ecqa.")

    if not os.path.exists(os.path.join('feb_data', f'{task_name}')):
        os.makedirs(os.path.join('feb_data', f'{task_name}'))

    with jsonlines.open(os.path.join('feb_data', f'{task_name}', f'{task_name}_train_{seed}.json'), 'w') as writer:
        for item in original_data_splits['train']:
            writer.write(item)
    with jsonlines.open(os.path.join('feb_data', f'{task_name}', f'{task_name}_validation_{seed}.json'), 'w') as writer:
        for item in original_data_splits['validation']:
            writer.write(item)

