import pandas as pd 
import spacy 
import argparse
from tqdm import trange
import numpy as np
import os
import jsonlines 

def turn_implied_statements_to_explanations(split, df):
    '''
    This function implements a set of rules to transform annotations of which identity-based group is targeted and what stereotypes of this group are referenced or implied into a single, coherent sentence (explanation).
    For example:
    `targetMinority` == "women"
    `targetStereotype` == "can't drive"
    return: "this posts implies that women can't drive." 

    For attacks on individuals, it will return "this post is a personal attack". 

    For posts that are not offensive, it will return "this post does not imply anything offensive"
    '''
    if df is None:  
        df = pd.read_csv(f"{args.dataset_path}SBIC.v2.{split}.csv")
    df["offensiveYN"] = df["offensiveYN"].apply(lambda x: 'not offensive' if x < 0.5 else 'offensive') 

    group_attack_no_implied_statement = 0
    personal_attack = 0
    not_offensive = 0
    group_offensive = 0
    total = 0

    drop_indices = []
    for i in trange(len(df["targetStereotype"])):
        offensive_label = df.loc[i,"offensiveYN"]

        if offensive_label == 'offensive' and (pd.isna(df.loc[i, "whoTarget"]) or df.loc[i, "whoTarget"]==''):
            drop_indices.append(i)
            continue

        if offensive_label == 'offensive' and df.loc[i,"whoTarget"] == 1.0: # only posts that target a group have annotations of implied statements 
            if pd.isna(df.loc[i, "targetStereotype"]) or df.loc[i, "targetStereotype"]=='': 
                group_attack_no_implied_statement += 1 
                drop_indices.append(i)
                continue
            inferences = df.loc[i, "targetStereotype"].split(' [SEP] ')
            target_minorities = df.loc[i, "targetMinority"].split(' [SEP] ')
            new_inferences = []
            for target_minority, inference in zip(target_minorities,inferences): 
                inference_annotations = [{'token': token.text, 'pos': token.pos_, 'tag': token.tag_} for token in nlp(inference)]
                first_word = inference_annotations[0]
                changed = False 

                if first_word['token'] == 'claims' and first_word['tag'] not in ['VBD', 'VBZ']: 
                    new_inferences.append(f'this post {inference}')
                    changed = True

                if first_word['tag'] in ['VBD', 'VBZ']: # verb, 3rd person singular present or simple past tense (e.g, said, calls, makes, uses, implies, marginalized, advocates, encourages, trivializes, refers)
                    new_inferences.append(f'this post {inference}')
                    changed = True
                
                if (first_word['tag'] == 'VBG' and first_word['token'] not in ['implying', 'calling']) or first_word['token'] == 'being': # gerund or present participle and "being" [infinitive (e.g, using, lying, having, disrespecting, being; BUT NOT implying or calling)]
                    new_inferences.append(f'this post implies that {inference}')
                    changed = True

                if first_word['token'] in ['implying', 'calling']:
                    new_inferences.append(f'this post is {inference}')
                    changed = True
                
                if first_word['token'] == 'rape':
                    new_inferences.append(f'this post implies that {target_minority} {inference}')
                    changed = True
                
                if first_word['tag'] in ['MD', 'VB'] or (first_word['pos'] == 'AUX' and first_word['token'] != 'being'): # verb, modal verb or base form 
                    if (target_minority in ["women", "men"] or "folks" in target_minority or len(target_minority.split(',')) > 1) and first_word['token'] == 'is': # verb does not match targets' number
                        split_inference = ' '.join(inference.split(' ')[1:])
                        new_inferences.append(f'this post implies that {target_minority} are {split_inference}')
                        changed = True
                    else:
                        new_inferences.append(f'this post implies that {target_minority} {inference}')
                        changed = True
                
                if first_word['token'] == "all":
                    new_inferences.append(f'this post implies that {target_minority} are {inference}')
                    changed = True
                
                if not changed: 
                    new_inferences.append(f'this post implies that {inference}')
                group_offensive += 1
            if len(new_inferences) > 1:
                df.loc[i, "targetStereotype"] = ' [SEP] '.join(new_inferences)
            else: 
                df.loc[i, "targetStereotype"] = new_inferences[0]

        if offensive_label == 'offensive' and df.loc[i,"whoTarget"] == 0.0:
            personal_attack += 1
            df.loc[i, "targetStereotype"] = f'this post is a personal attack'

        if offensive_label == 'not offensive':
            not_offensive += 1
            df.loc[i, "targetStereotype"] = f'this post does not imply anything offensive'  

        total += 1
    
    print ("---------------------------------------------------")
    print (f"Split: {split}")    
    print (f"Group attack but no implied statement: {group_attack_no_implied_statement}")
    print (f"Personal attacks: {personal_attack}")
    print (f"Group offensive: {group_offensive}")
    print (f"Not offensive: {not_offensive}")
    print (f"Total: {total}")
    print ("---------------------------------------------------")

    df_filter = df.drop(drop_indices)

    return df_filter



def aggregate_sbic_annotations(split):
    '''
    In the original SBIC csv file, one post occurs multiple times with annotations from different workers. 
    Here, for each post, we aggregate its annotations into a single row (for eval instances) or make multiple train instances. 
    '''
    df = pd.read_csv(f"{args.dataset_path}SBIC.v2.{split}.csv")
    columns = ["post", "offensiveYN", "whoTarget", "targetMinority", "targetStereotype"]
    aggregated_data = []
    visited_posts = []
    for i in trange(len(df["targetStereotype"])):
        post = df.loc[i, "post"]
        if post in visited_posts:
            continue
        visited_posts.append(post)

        # A post is offensive if at least half of the annotators say it is. 
        offensiveYN_frac = sum(df.loc[df["post"]==post]["offensiveYN"]) / float(len(df.loc[df["post"]==post]["offensiveYN"]))
        offensiveYN_label = 1.0 if offensiveYN_frac >= 0.5 else 0.0

        # A post targets a demographic group if at least half of the annotators say it does.
        whoTarget_frac = sum(df.loc[df["post"]==post]["whoTarget"]) / float(len(df.loc[df["post"]==post]["whoTarget"]))
        whoTarget_label = 1.0 if whoTarget_frac >= 0.5 else 0.0

        targetMinority_label = None 
        targetStereotype_label = None
        
        if whoTarget_label == 1.0: # The post targets an identity group; only such posts have annotations of stereotypes of the group that are referenced or implied
            minorities = df.loc[df["post"]==post]["targetMinority"]
            stereotypes = df.loc[df["post"]==post]["targetStereotype"]

            if split in ['dev', 'tst']: # For evaluation, we combine all implied statements into a single string separated by [SEP] 
                targetMinority_labels = []
                targetStereotype_labels = []
                for m, s in zip(minorities, stereotypes):
                    if not pd.isna(s):
                        targetMinority_labels.append(m)
                        targetStereotype_labels.append(s)
                targetMinority_label = ' [SEP] '.join(targetMinority_labels)
                targetStereotype_label = ' [SEP] '.join(targetStereotype_labels)
                aggregated_data.append([post, offensiveYN_label, whoTarget_label, targetMinority_label, targetStereotype_label])
            else: # For training, each implied statement leads to an individual training instance
                for m, s in zip(minorities, stereotypes):
                    if not pd.isna(s):
                        aggregated_data.append([post, offensiveYN_label, whoTarget_label, m, s])
        else: 
            aggregated_data.append([post, offensiveYN_label, whoTarget_label, targetMinority_label, targetStereotype_label])
    df_new = pd.DataFrame(aggregated_data, columns=columns) 
    return df_new

  
def process_senmaking_data(data_path):
    '''
    We can basically use this data as it is originally published.
    '''
    for split in ['Training', 'Dev', 'Test']:
        task_input_filename = 'subtaskA_data_all' if split == 'Training' else f'subtaskA_{split.lower()}_data'
        task_input_file_path = data_path + split + f'_Data/{task_input_filename}.csv'
        task_input_df = pd.read_csv(task_input_file_path)
        if split == 'Training': 
            task_input_df_extended = pd.DataFrame(np.repeat(task_input_df.values, 3, axis=0))
            task_input_df_extended.columns = task_input_df.columns
        
        task_output_filename = 'subtaskA_answers_all' if split == 'Training' else f'subtaskA_gold_answers'
        task_output_file_path = data_path + split + f'_Data/{task_output_filename}.csv'
        task_output_df = pd.read_csv(task_output_file_path, header=None, names=['id', 'NonsensicalSentence'])
        if split == 'Training': 
            task_output_df_extended = pd.DataFrame(np.repeat(task_output_df.values, 3, axis=0))
            task_output_df_extended.columns = task_output_df.columns

        explanations_output_path_file = task_output_file_path.replace('subtaskA', 'subtaskC')
        explanations_output_df = pd.read_csv(explanations_output_path_file, header=None, names=['id', 'CorrectExplanation1', 'CorrectExplanation2', 'CorrectExplanation3'])

        if split == 'Training':
            correct_explanations = []
            for index, row in explanations_output_df.iterrows():
                correct_explanations.extend([row['CorrectExplanation1'], row['CorrectExplanation2'], row['CorrectExplanation3']])
            sent0_list = task_input_df_extended['sent0'].values.tolist()
            sent1_list = task_input_df_extended['sent1'].values.tolist()
            labels = task_output_df_extended['NonsensicalSentence'].values.tolist()
        else: 
            correct_explanations = ['[SEP]'.join(explanations_output_df.loc[i, ['CorrectExplanation1', 'CorrectExplanation2', 'CorrectExplanation3']]) for i in range(len(explanations_output_df))] 
            sent0_list = task_input_df['sent0'].values.tolist()
            sent1_list = task_input_df['sent1'].values.tolist()
            labels = task_output_df['NonsensicalSentence'].values.tolist()
        dataset = list(zip(sent0_list, sent1_list, labels, correct_explanations))

        dataset_df = pd.DataFrame(dataset, columns = ['sent0', 'sent1', 'label', 'explanation'])
        print (f"{data_path}SenMaking.{split}.csv")
        dataset_df.to_csv(f"{data_path}SenMaking.{split}.csv")

def process_ecqa_data(data_path):
    '''
    ECQA dataset introduces explanations for instances in previously published CQA (CommonsenseQA) dataset. 
    ECQA contains not only justifications of the correct answer (`positives`), but also justifications that refute the incorrect answer choices.
    We use only the former since they answer "why is [input] assigned [label]?", just as explanations in other datasets that we have included in FEB.
    '''

    # Collects explanations from ECQA 
    ecqa_annotations_data = {}
    with jsonlines.open(os.path.join(data_path, 'ecqa.jsonl')) as ecqa_annotations_reader: 
        for item in ecqa_annotations_reader: 
            explanation = '. '.join(item['positives'])
            explanation += '.'
            explanation = explanation.replace('..','.')
            ecqa_annotations_data[item['id']] = {'explanation': explanation}

    # Collects task instances from CQA 
    cqa_data = {}
    for split in ['train', 'dev']:
        filename = f'cqa/{split}_rand_split.jsonl' if split != 'test' else f'cqa/{split}_rand_split_no_answers.jsonl' 
        with jsonlines.open(os.path.join(data_path, filename), 'r') as cqa_split_reader:
            for item in cqa_split_reader: 
                if split == 'test':
                    answer = ''
                else:
                    for choice in item['question']['choices']:
                        if choice['label'] == item['answerKey']:
                            answer = choice['text']
                item_id = item['id']
                question = item['question']['stem']
                choices = [choice['text'] for choice in item['question']['choices']]
                explanation = ecqa_annotations_data[item_id]['explanation'] 
                cqa_data[item_id] = {
                                     'question': question,
                                     'answer': answer, 
                                     'choices': choices, 
                                     'explanation': explanation,
                                    }

    # ECQA and CQA have different splits; use ECQA splits
    no_cqa_annotations = 0
    ecqa_count = 0 
    for split in ['train', 'test', 'val']:
        with open(os.path.join(data_path, f'author_split/{split}_ids.txt')) as split_ids_file:
            split_ids = split_ids_file.read().splitlines() 
        split = split.replace('val', 'validation')
        with jsonlines.open(os.path.join(data_path, f'ecqa_{split}.jsonl'), 'w') as ecqa_split_data: 
            for item_id in split_ids:
                ecqa_split_data.write(cqa_data[item_id])       

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset name, supported: sbic")
    parser.add_argument("--dataset_path", type=str, help="path to dataset")
    args = parser.parse_args()

    if args.dataset == "sbic":
        nlp = spacy.load("en_core_web_sm")

        for split in ['trn', 'dev','tst']:
            df_agg = aggregate_sbic_annotations(split)
            df_modified = turn_implied_statements_to_explanations(split, df_agg, False)
            df_modified.to_csv(f"{args.dataset_path}SBIC.v2.{split}.modified.csv")
    elif args.dataset == "sensemaking": 
        process_senmaking_data(args.dataset_path)
    elif args.dataset == "ecqa": 
        process_ecqa_data(args.dataset_path)
    else: 
        raise ValueError('Use "sbic", "sensemaking", or "ecqa".')
                