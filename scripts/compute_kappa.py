
import argparse
import csv
import numpy as np
from collections import defaultdict, OrderedDict
from statsmodels.stats.inter_rater import fleiss_kappa
import math
from sklearn.metrics import accuracy_score

def calculate_metrics(selected_annotations, label_output=''):
    # To compute kappa, need a matrix with rows as questions and columns as ratings.
    # Compute agreement for all questions with >= 2 annotations.
    # Collapse w_yes, yes -> yes / w_no, no -> no
    label_mapping_iaa = {"w_yes": 1, "yes": 1, "w_no": 0, "no": 0}
    label_mapping_plausibility = {"yes": 1, "w_yes": 2/3, "w_no": 1/3, "no": 0}

    unanimous_votes = 0
    for i, row in enumerate(selected_annotations):
        # row is a list like ['no', 'yes', 'w_yes']
        label_mapped_iaa = [label_mapping_iaa[x] for x in row]
        if len(set(label_mapped_iaa)) == 1:
            unanimous_votes += 1
    print(f"For explanation_type={explanation_type}, {label_output}number_of_unanimous_votes={unanimous_votes}/{len(selected_annotations)}")

    summary = np.zeros((len(selected_annotations), 2), dtype=np.int64)
    for i, row in enumerate(selected_annotations):
        # row is a list like ['no', 'yes', 'w_yes']
        for a in row:
            summary[i, label_mapping_iaa[a]] += 1
    kappa = fleiss_kappa(summary, method='uniform')
    print(f"For explanation_type={explanation_type}, {label_output}fleiss_kappa={kappa}")

    scores = []

    for i, row in enumerate(selected_annotations):
        # row is a list like ['no', 'yes', 'w_yes']
        score = sum([label_mapping_plausibility[x] for x in row]) / len(row)
        scores.append(score)
    print(f"For explanation_type={explanation_type}, {label_output}score_average={np.mean(scores)*100}")
    print(f"For explanation_type={explanation_type}, {label_output}score_std={np.std(scores)*100}")
    print(f"For explanation_type={explanation_type}, {label_output}score_stderr={np.std(scores)/math.sqrt(len(scores))*100}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--label_breakdown", action='store_true', default=False)
    args = parser.parse_args()

    print(f"Analyzing {args.input_file}")

    # Each row has 10 examples.
    # Each example is identified with a question ID, e.g. 'Input.1_ques_id': '198'
    # Columns like Input.gt_1 signify whether gold or model explanation was first, with 0 == gold explanation is e1, 1 == model explanation is e1
    annotations = {}  # question id -> List[annotations]
    label_eval_items = defaultdict(list)

    with open(args.input_file, 'r') as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            for k in range(1, 11):  # the CSV columns are indexed starting at 1
                # get the question ids
                question_id = row[f"Input.{k}_ques_id"]
                # get the answers
                answer_e1 = row[f"Answer.{k}-e1"]
                answer_e2 = row[f"Answer.{k}-e2"]

                gold_explanation_idx = int(row[f'Input.{k}_gt_idx']) + 1
                if gold_explanation_idx == 1:
                    gold_answer = answer_e1
                    pred_answer = answer_e2
                else:
                    gold_answer = answer_e2
                    pred_answer = answer_e1

                if question_id not in annotations:
                    annotations[question_id] = []
                annotations[question_id].append([gold_answer, pred_answer])
                label_eval_items[row[f"Input.gt_{k}"]].append(question_id)

    # Summarize the data.
    num_questions = len(annotations)
    num_annotations = [len(a) for a in annotations.values()]
    print(f"Number questions: {num_questions}")
    print(f"min/mean/max number annotations per question: {min(num_annotations)}, {np.mean(num_annotations)}, {max(num_annotations)}")
    print(f"Using {len([a for a in annotations.values() if len(a) == 3])} questions with 3 annotators to compute kappa.")

    for k, explanation_type in enumerate(["gold", "predicted"]):
        if args.label_breakdown:
            label_eval_items_ordered = OrderedDict(sorted(label_eval_items.items())) 
            for label in label_eval_items_ordered:
                selected_annotations = [[a[k] for a in annotation] for question_id, annotation in annotations.items() if len(annotation) == 3 and question_id in label_eval_items[label]]
                label_output = f'label={label}, '
                calculate_metrics(selected_annotations, label_output)
        else: 
            selected_annotations = [[a[k] for a in annotation] for annotation in annotations.values() if len(annotation) == 3]
            label_output = ''
            calculate_metrics(selected_annotations, label_output)

        
