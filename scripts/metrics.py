from tqdm import tqdm
import os
import pandas as pd
import numpy as np 
import torch
import datasets 
import json
from feature_conversion_methods import unified_qa_esnli_label_mapping, wt5_esnli_label_mapping, unified_qa_sbic_label_mapping


def evaluate(
    save_path,
    dataset,
    model,
    tokenizer,
    split,
    task,
    device,
    explanation_sep,
    rationale_only=False,
    label_only=False,
    generations_file=None,
    io_format=None
):
    fname = os.path.join(save_path, "%s_generations.txt" % split)
    if os.path.isfile(fname):
        fname = fname.split(".txt")[0] + "_1.txt"

    if generations_file is not None: # actual words have already been decoded and saved in `generations_file`
        with open(generations_file, "r") as f: 
            lines = f.readlines()
        generations_list = [l.replace("\n", " ").replace(tokenizer.eos_token, " ").strip()for l in lines] # strip newlines & EOS token (if exists)
    else: # decode output words
        generations_list = []
        with open(fname, "w") as w:
            for i, element in tqdm(enumerate(dataset), total=len(dataset)):
                inpt_tensor = torch.tensor(element["input_ids"], device=device).reshape(1, -1)
                
                out = model.generate(
                    inpt_tensor,
                    max_length=100,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                skip_special_tokens = False if "infilling" in io_format else True
                words = tokenizer.decode(out[0].tolist(), skip_special_tokens=skip_special_tokens)
                if "infilling" in io_format:
                    words = words.replace("<extra_id_1>", f" {explanation_sep}")
                    words = words.replace(tokenizer.pad_token,'')
                    words = words.replace("<extra_id_0>", '')
                    words = words.replace("<extra_id_2>", '')
                    words = ' '.join(words.split())
                words = (words.replace("\n", " ").replace(tokenizer.eos_token, " ").strip())
                w.write(words + "\n")
                generations_list.append(words)

    broken_count = 0

    # for Bertscore
    bertscore = None 
    expl_true = []
    expl_pred = []

    # for accuracy 
    accuracy = None 
    label_mapping = {
        "sbic": {
            "not_offensive": 0,
            "offensive": 1
        }
    }
    label_true = []
    label_pred = []
    acc = []

    # to separate label from explanation
    if explanation_sep in ['explanation', 'explanation_why']:
        explanation_sep = explanation_sep + ':' # when these separators are used we format the input as: `answer explanation: explanation`

    analysis_file = os.path.join(save_path, "%s_posthoc_analysis.txt" % split)
    if os.path.isfile(analysis_file):
        analysis_file = analysis_file.split(".txt")[0] + "_1.txt"

    with open(analysis_file, "w") as g:
        for _, (line, gold) in tqdm(enumerate(zip(generations_list, dataset)), total=len(dataset)):
            broken_generation = False

            if rationale_only:
                pred_l = None 
                pred_e = line.strip()
            elif label_only:
                pred_l = line.strip()
                pred_e = None
            else: 
                line_split = line.split(explanation_sep)
                if len(line_split) > 1:
                    pred_l = line_split[0].strip()
                    pred_e = line_split[1].strip()
                    g.write(f"Predicted: {pred_l} | {pred_e}\n")
                else: 
                    print(f"This line couldn't be processed (most likely due to format issue): {line}")
                    pred_l = line.strip()
                    pred_e = "UNK"
                    broken_count += 1
                    broken_generation = True
                    g.write(f"Predicted: {pred_l}\n")

            if task in ["cos_e", "ecqa"]:
                gold_l = gold["answer"]
                gold_explanations_string = gold["abstractive_explanation"] if task == "cos_e" else gold["explanation"]
                gold_explanations = [gold_explanations_string]
                g.write(gold["question"] + "\n")
                g.write(f"Correct: {gold_l} | {gold_explanations_string}\n")

            elif task == "esnli":
                gold_l = unified_qa_esnli_label_mapping[gold["label"]] if ('unified' in io_format and io_format not in ['unifiedqa_snli_mix_what_with_choices',
                                                                                                                        'unifiedqa_snli_mix_what_v2',
                                                                                                                        'unifiedqa_snli_mix_what_with_choices_v2',
                                                                                                                        'unifiedqa_what_v2']) or \
                                                                        io_format in ['squad', 'squad_nli_mix'] else wt5_esnli_label_mapping[gold["label"]]
                gold_explanations = [gold[f"explanation_{k}"] for k in [1,2,3]] # there can be up to 3 human gold-explanations in E-SNLI. Only first 2 gold explanations are use to compute BLEU in prior works. We follow suit.
                gold_explanations_string = ' [SEP] '.join(gold_explanations)

                g.write(gold["premise"] + " " + gold["hypothesis"] + "\n")
                g.write(f"Correct: {gold_l} | {gold_explanations_string} \n")

            elif task == "sbic":
                if ('unified' in io_format and 'what' not in io_format and 'unifew' not in io_format) or io_format == "squad_yn":
                    gold_l = unified_qa_sbic_label_mapping[gold["offensiveYN"]]
                elif io_format == 't5_fewshot_infilling_bool':
                    gold_l = 'True' if gold["offensiveYN"] == 'offensive' else 'False'
                else:
                    gold_l = gold["offensiveYN"].replace("not offensive", "not_offensive")
                gold_explanations_string = gold["targetStereotype"]
                post = gold['post'].replace('\n', ' ')
                g.write(f"{post}\n")
                if pd.isna(gold_explanations_string) or pd.isna(gold_l):
                    raise ValueError('Gold label or explanation empty...')
                else:
                    g.write("Correct: " + gold_l + " | " + gold_explanations_string + "\n")
                    gold_explanations = gold_explanations_string.split(' [SEP] ')

            elif task == "sensemaking":
                if '_yn' in io_format:
                    gold_l = "yes" if bool(int(gold['label'])) else "no"
                elif io_format in ['copa_bool', 't5_fewshot_infilling_bool']: 
                    gold_l = str(bool(int(gold['label'])))
                else:
                    gold_l_idx = str(int(gold["label"])+1)
                    gold_l = f"choice{gold_l_idx}"
                gold_explanations_string = gold["explanation"]
                gold_explanations = gold_explanations_string.split('[SEP]')

                g.write(f"choice1: {gold['sent0']} choice2: {gold['sent1']}\n")
                g.write(f"Correct: {gold_l} | {gold_explanations_string} \n")

            else:
                raise Exception("Unknown task. Currently supported: esnli, cos_e, sbic, sensemaking, ecqa.")

            if not label_only:
                # for Bertscore 
                expl_pred.append(pred_e)
                expl_true.append(gold_explanations)

            if not rationale_only:
                # for accuracy
                if task == "sbic":
                    gold_key = "offensive" if gold_l.lower() in ["offensive", "yes"] else "not_offensive"
                    if not broken_generation:
                        pred_key = "offensive" if pred_l.lower() in ["offensive", "yes"] else "not_offensive"
                        met = gold_key.lower() == pred_key.lower()
                        if task == "sbic" and pred_key in label_mapping[task].keys():
                            label_pred.append(label_mapping[task][pred_key])
                            label_true.append(label_mapping[task][gold_key])
                        else:
                            print(f"Broken label: {pred_l}")
                            met = False
                            label_pred.append(len(label_mapping[task]))
                            label_true.append(label_mapping[task][gold_key])
                    else:
                        met = False
                        label_pred.append(len(label_mapping[task]))
                        label_true.append(label_mapping[task][gold_key])
                else:
                    met = gold_l.lower() == pred_l.lower()
                acc.append(met)
                g.write("Considered Correct: " + str(met) + "\n")
                g.write("\n")

    results = {}

    if not rationale_only:
        # final accuracy 
        accuracy = sum(acc) / len(acc) * 100

    if not label_only:
        # BERTscore
        bertscore_metric = datasets.load_metric("bertscore")
        bertscores = []
        for pred_expl, list_gold_expl in zip(expl_pred, expl_true):
            instance_bertscores = []
            for gold_expl in list_gold_expl: 
                score = bertscore_metric.compute(predictions=[pred_expl.lower()], references=[gold_expl.lower()], lang="en")["f1"][0]*100
                instance_bertscores.append(score)
            bertscores.append(max(instance_bertscores))

        bertscore = np.mean(bertscores)
        bertscores_correct_prediction = [score for correct_yn, score in zip(acc, bertscores) if correct_yn]
        bertscore_correct_prediction = np.mean(bertscores_correct_prediction)

        bertscores_correct_normalized = [score if correct_yn else 0.0 for correct_yn, score in zip(acc, bertscores)]
        bertscore_correct_normalized = np.mean(bertscores_correct_normalized)

    if split == 'validation':
        split = 'dev'
    (
     results[f"{split}_broken_count"],
     results[f"{split}_acc"],
     results[f"{split}_bertscore"],
     results[f"{split}_bertscore_correct_pred"],
     results[f"{split}_bertscore_correct_normalized"]
    ) = (broken_count, accuracy, bertscore, bertscore_correct_prediction, bertscore_correct_normalized)

    with open(os.path.join(save_path, f"results_{split}.json"), "w") as fp:
        json.dump(results, fp)

    return results 
