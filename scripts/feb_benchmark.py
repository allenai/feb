'''
This script will generate the same data splits as in the paper and store them in `feb_data` folder.
For our experiments, we didn't save and load splits, but generated them for every run inside `input_to_label_and_rationale.py`.
'''

from feb_splits import make_splits

seeds_fewshot = [7004, 3639, 6290, 9428, 7056, 4864, 4273, 7632, 2689, 8219, 4523, 2175, 7356, 8975, 51, 4199, 4182, 1331, 2796, 6341, 7009, 1111, 1967, 1319, 741, 7740, 1335, 9933, 6339, 3112, 1349, 8483, 2348, 834, 6895, 4823, 2913, 9962, 178, 2147, 8160, 1936, 9991, 6924, 6595, 5358, 2638, 6227, 8384, 2769, 4512, 2051, 4779, 2498, 176, 9599, 1181, 5320, 588, 4791]

data_path_dict = {'esnli': None,
                  'cos_e': None,
                  'ecqa': 'data/ECQA-Dataset',
                  'sensemaking': 'data/SenseMaking/',
                  'sbic': 'data/SBIC/'}

for dataset in ['esnli', 'ecqa', 'sbic', 'sensemaking']:
    print (f"Starting to create splits for {dataset}...")
    for seed in seeds_fewshot:
        make_splits(seed, dataset, data_path=data_path_dict[dataset])