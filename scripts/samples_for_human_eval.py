import argparse
import glob
import os
from collections import defaultdict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="path to the validation output dir that has all episodes")
    parser.add_argument("--dataset", type=str, required=True, help="dataset name")
    parser.add_argument("--sample", type=int, default=6, help="sample size per episode")
    args = parser.parse_args()

    assert args.sample % 2 == 0
    assert args.sample % 3 == 0

    if args.dataset in ['sensemaking', 'sbic']:
        num_labels = 2
    elif args.dataset == 'esnli':
        num_labels = 3
    elif args.dataset == 'cose':
        num_labels = 1
    else:
        assert False
    sample_per_label = args.sample // num_labels

    episode_dirs = sorted(glob.glob(f'{args.output_dir}/*'))
    for episode_dir in episode_dirs:
        ls = sorted(glob.glob(f'{episode_dir}/*'))
        assert len(ls) == 2
        analysis_file = f'{ls[0]}/validation_posthoc_analysis.txt'
        assert os.path.isfile(analysis_file)
        # print(analysis_file)
        with open(analysis_file) as f:
            all_lines = [line for line in f]
        i = 0
        label_count = defaultdict(int)
        while i < len(all_lines):
            pred = all_lines[i].strip()
            input = all_lines[i + 1].strip()
            gold = all_lines[i + 2].strip()
            eval = all_lines[i + 3].strip()
            empty_line = all_lines[i + 4].strip()

            assert pred.startswith('Predicted: ')
            pred_l, pred_e = pred.replace('Predicted: ', '').split(' | ', 1)
            assert gold.startswith('Correct: ')
            gold_l, gold_e = gold.replace('Correct: ', '').split(' | ', 1)
            assert eval.startswith('Considered Correct: ')
            eval = eval.replace('Considered Correct: ', '')
            assert eval in ['True', 'False']
            is_correct = True if eval == 'True' else False
            assert empty_line == ""
            if args.dataset == 'sbic':
                pred_l = pred_l.replace('Yes', 'offensive').replace('No', 'not_offensive')
            assert is_correct == (pred_l == gold_l)

            i += 5
            if is_correct:
                if args.dataset == 'cose':
                    gold_l = 'gold'
                if label_count[gold_l] < sample_per_label:
                    label_count[gold_l] += 1
                    print(pred_l, gold_l, pred_e, gold_e)

        found_all = len(label_count) == num_labels
        for label, found_count in label_count.items():
            if found_count < sample_per_label:
                found_all = False
                break

        if not found_all:
            print(analysis_file)
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', label_count)
        else:
            print('====================================')
