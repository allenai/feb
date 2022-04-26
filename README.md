
Code associated with the [Few-Shot Self-Rationalization with Natural Language Prompts](https://arxiv.org/abs/2111.08284) NAACL Findings 2022 paper

## Citation 

```
@inproceedings{marasovic-beltagy-et-al-2022-feb,
    title = "Few-Shot Self-Rationalization with Natural Language Prompts",
    author = "Marasovi{\'c}, Ana and Beltagy, Iz and Downey, Doug and Peters, Matthew E.",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2022",
    url= "https://arxiv.org/abs/2111.08284",
}

```
## Installation

1. Clone the repository

```
git clone https://github.com/allenai/feb
cd feb
```

2. [Download and install Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

3. Create and activate a Conda environment. 

```
conda create -n feb python=3.7
conda activate feb
```

4. Download the requirements. 

```
pip install -r requirements.txt

python -m spacy download en_core_web_sm

wandb offline
```

## Train/Eval Data

### Preferred: Download 

The following command will download the datasets (all except E-SNLI), but doesn't give splits we used for our experiments.

```
wget https://storage.googleapis.com/feb-data/data.zip
unzip data.zip
```

### FEB Splits

For our experiments, we didn't save and load splits, but generated them for every run inside `input_to_label_and_rationale.py`. If you'd like to have the same splits that we used stored locally, you should run the following:

```
mkdir feb_data
python feb_benchmark.py
```


### Replicate Data 

If you want to replicate our preprocessing steps, you can do so with the following commands:

#### SBIC 

```
wget https://homes.cs.washington.edu/~msap/social-bias-frames/SBIC.v2.tgz
tar -xvzf SBIC.v2.tgz
python scripts/preprocess_data.py --dataset sbic --dataset_path <path-to-sbic-folder>`
```
#### SenseMaking 

```
python scripts/preprocess_data.py --dataset sensemaking --dataset_path <path-to-sensemaking-folder>
```

#### ECQA
```
git clone https://github.com/dair-iitd/ECQA-Dataset.git

cd ECQA-Dataset 
mkdir cqa
cd cqa
wget https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl 
wget https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl

cd explain_lm/finetuning
python scripts/preprocess_data.py --dataset ecqa --dataset_path <path to `ECQA-Dataset`>
```

## Run evaluation

### T5/UnfifiedQA

We run our experiments on Google Cloud with N gpus allocated only for this project. All our experiments are done by jointly training models and evaluating them on the dev set. The command below will train and evaluate given models on chosen datasets with 60 random seeds: 

```
python scripts/exp.py --exp_root <path_to_checkpoints_folder> --not_dryrun --model_vals <a string of models to evaluate separated by comma> --dataset_vals <a string of datasets to evaluate on, separated by comma> --n_gpus <number of available GPUs>
```

By default these experiments will be done with IO formats (prompts) that find to work the best (according to the experiments in the paper), but you can play around with different values in `format_dict` in `scripts/exp.py`.

The same command with concrete values: 

```
mkdir checkpoints
python scripts/exp.py --exp_root checkpoints --not_dryrun --model_vals t5-base,t5-large,t5-3b --dataset_vals esnli --n_gpus 4
python scripts/exp.py --exp_root checkpoints --not_dryrun --model_vals allenai/unifiedqa-t5-base,allenai/unifiedqa-t5-large,allenai/unifiedqa-t5-3b --dataset_vals ecqa,sensemaking,sbic --n_gpus 4
```


### GPT3 

```
python scripts/exp.py --exp_root <path_to_checkpoints_folder> --model_vals <a string of models to evaluate separated by comma> --dataset_vals <a string of datasets to evaluate on, separated by comma>  --use_gpt3 --openai_key <your_openai_key>  --not_dryrun
```

### Collect results 

After you're doing with training/eval with 60 seeds, you can collect results (mean, stddev) by running this: 

```
mkdir out
python scripts/exp.py --exp_root <path_to_checkpoints_folder>  --collect_results
```

If you get the assertion error, check which runs have not been trained properly, repeat evaluating only those seeds, and run the above command again. 

### Human evaluation

We sampled and prapered data for human evaluation on Mturk using the following command: 
```
python scripts/samples_for_human_eval.py --output_dir <path_to_checkpoints_folder> --dataset <dataset_name>
```

Our samples are available at `human_eval/batches`, crowdworkers' annotation at `human_eval/results`, and templates for Mturk interface at `human_eval/templates`. We followed [these templates](https://github.com/maximek3/e-ViL/tree/main/eViL_Mturk) for related work to produce our. 

To get results of explanation plausibility (per crowdworkers) and Fleiss' Kappa, run: 

```
python scripts/compute_kappa.py --input_file <a file in human_eval/results/>
```
