from collections import defaultdict
import random
"""
Example-to-Feature conversion methods
Modified from
https://github.com/salesforce/cos-e/blob/master/code/generation/train_commonsenseqa_v1.0.py and ""_v1.11.py (identical)
as well as Tensorflow code for WTF?: 
https://github.com/google-research/google-research/blob/master/wt5/wt5/preprocessors.py
"""
# This code is based on https://github.com/allenai/label_rationale_association/blob/main/feature_conversion_methods.py

unified_qa_esnli_label_mapping = {0: 'yes', 1: 'maybe', 2: 'no'}
unified_qa_esnli_label_mapping_upper = {0: 'Yes', 1: 'Maybe', 2: 'No'} 
wt5_esnli_label_mapping = {0: 'entailment', 1: 'neutral', 2: 'contradiction'} 
unified_qa_sbic_label_mapping = {"offensive": 'Yes', "not offensive": 'No'}

def format_instance(
        example,
        tokenizer,
        explanation_sep,
        max_seq_length=None,
        datasource=None,
        io_format=None, 
):
    assert datasource in {"cos_e", "esnli", "sbic", "sensemaking", "ecqa"}

    if datasource in ["cos_e", "ecqa"]:
        input_string, answer_string = cqa_formatting(example, io_format, explanation_sep, datasource)
    elif datasource == "esnli":
        input_string, answer_string = esnli_formatting(example, io_format, explanation_sep)
    elif datasource == 'sbic':
        input_string, answer_string = sbic_formatting(example, io_format, explanation_sep)
    elif datasource == 'sensemaking':
        input_string, answer_string = sensemaking_formatting(example, io_format, explanation_sep)
    else:
        raise ValueError("Unknown task. Currently supported: esnli, cos_e, sbic, sensemaking, ecqa.")
    
    if 'unified' in io_format and 'unifew' not in io_format:
        input_string += '</s>'

    input_string = ' '.join(input_string.split())
    answer_string = ' '.join(answer_string.split())

    input_string = ' '.join(input_string.split())
    answer_string = ' '.join(answer_string.split())

    encodings = tokenizer.encode_plus(
        input_string,
        max_length=max_seq_length,
        pad_to_max_length=False,
        return_token_type_ids=False,
        return_attention_mask=True,
    )


    # note even with "lm_labels.shift_right()", the decoder attention mask length is still correct since we remove the last token
    dec = tokenizer.encode_plus(
        answer_string,
        max_length=max_seq_length,
        pad_to_max_length=False,
        return_token_type_ids=False,
        return_attention_mask=True,
    )

    encodings["labels"] = dec["input_ids"]
    encodings["decoder_attention_mask"] = dec["attention_mask"]
    encodings["question_encoding"] = encodings["input_ids"]

    #return encodings
    return {**example, **encodings}


def cqa_formatting(item, io_format, explanation_sep, datasource):
    question = item["question"]
    answer = item["answer"]
    abstr_expl = item["abstractive_explanation"].lower() if datasource == 'cos_e' else item["explanation"].lower()


    if io_format == 't5_fewshot_infilling_with_choices':
        input_string = f"explain {datasource} question: {question} choice: " + " choice: ".join(item["choices"]) + f" <extra_id_0> {explanation_sep} <extra_id_1>"
        answer_string = f"<extra_id_0> {answer} <extra_id_1> {abstr_expl} <extra_id_2>"
    elif io_format == 't5_fewshot_infilling_more_natural':
        input_string = f"explain {datasource} question: {question} choice: " + " choice: ".join(item["choices"]) + f" The answer is <extra_id_0> {explanation_sep} <extra_id_1>"
        answer_string = f"<extra_id_0> {answer} <extra_id_1> {abstr_expl} <extra_id_2>"
    elif io_format == "squad": 
        input_string = f"explain {datasource} question: {question} context: " + ', '.join(item['choices']) # explain cos_e question: When getting in shape you need to have this in between workouts? context: give up, period of recovery, jogging
        answer_string = f"{answer} {explanation_sep} {abstr_expl}" # period of recovery because without a period of recovery you will not get any gains.
    elif io_format == "record": 
        # might not work because cos_e doesn't have a passage 
        input_string = f"explain {datasource} query: {question} entities: " + ', '.join(item['choices']) # explain cos_e query: When getting in shape you need to have this in between workouts? entities: give up, period of recovery, jogging
        answer_string = f"{answer} {explanation_sep} {abstr_expl}" # period of recovery because without a period of recovery you will not get any gains.
    elif io_format == 'unifiedqa_matching':
        choice_ids = ['(A)', '(B)', '(C)', '(D)', '(E)']
        input_string = f'explain {question.lower()} \\n'
        for choice_id, choice in zip(choice_ids, item["choices"]):
            input_string += f' {choice_id} {choice.lower()}'
        answer_string = f"{answer.lower()} {explanation_sep} {abstr_expl.lower()}"
        answer_string = answer_string.lower()
    else:
        raise ValueError("The IO format is not supported. Choose `standard` or `masked_cause_generate`.")
    
    return input_string, answer_string


def esnli_formatting(item, io_format, explanation_sep):

    premise = item["premise"]
    hypothesis = item["hypothesis"]
    answer = unified_qa_esnli_label_mapping[item["label"]] if 'unified' in io_format else wt5_esnli_label_mapping[item["label"]]
    abstr_expl = item["explanation_1"].lower() 
    # Dev/test instances have more than one explanation annotated; merge them into one sequence separated by [SEP] 
    for k in [2,3]:
        if f"explanation_{k}" in item and item[f'explanation_{k}']!='': 
            abstr_expl += f" [SEP] {item[f'explanation_{k}'].lower()}"

    if io_format == 'standard':
        input_string = f"explain nli hypothesis: {hypothesis} premise: {premise}"
        answer_string = f"{answer} {explanation_sep} {abstr_expl}"
    elif io_format == 't5_fewshot_infilling':
        input_string = f"explain nli hypothesis: {hypothesis} premise: {premise} <extra_id_0> {explanation_sep} <extra_id_1>"
        answer_string = f"<extra_id_0> {answer} <extra_id_1> {abstr_expl} <extra_id_2>"
    elif io_format == 't5_fewshot_infilling_more_natural':
        input_string = f"explain nli hypothesis: {hypothesis} premise: {premise} This is <extra_id_0> {explanation_sep} <extra_id_1>"
        answer_string = f"<extra_id_0> {answer} <extra_id_1> {abstr_expl} <extra_id_2>"
    elif io_format == "squad": 
        input_string = f"explain nli question: Is this entailment? context: {hypothesis} {premise}"  
        answer_ynm = unified_qa_esnli_label_mapping[item["label"]]
        answer_string = f"{answer_ynm} {explanation_sep} {abstr_expl}" 
    elif io_format == "squad_endswith_what":
        input_string = f"explain nli question: What is this? context: {hypothesis} {premise}"  
        answer_string = f"{answer} {explanation_sep} {abstr_expl}"  
    elif io_format == "squad_nli_mix": 
        input_string = f"explain nli question: Is this entailment? context: hypothesis: {hypothesis} premise: {premise}"  
        answer_ynm = unified_qa_esnli_label_mapping[item["label"]]
        answer_string = f"{answer_ynm} {explanation_sep} {abstr_expl}"  
    elif io_format == "squad_nli_mix_endswith_what":  
        input_string = f"explain nli question: What is this? context: hypothesis: {hypothesis} premise: {premise}"  
        answer_string = f"{answer} {explanation_sep} {abstr_expl}"   
    elif io_format == 'unifiedqa_unifew':
        hypothesis = hypothesis.lower().rstrip('.')
        unified_qa_esnli_label_mapping_upper = {0: 'Yes', 1: 'Maybe', 2: 'No'}
        answer = unified_qa_esnli_label_mapping_upper[item["label"]]
        input_string = f'explain {premise} Is {hypothesis}? \\n (A) Yes (B) Maybe (C) No'
        answer_string = f"{answer} {explanation_sep} {abstr_expl}"  
    elif io_format == 'unifiedqa_unifew_nli_mix':
        premise = premise.lower().rstrip('.')
        unified_qa_esnli_label_mapping_upper = {0: 'Yes', 1: 'Maybe', 2: 'No'}
        input_string = f'explain hypothesis: {hypothesis} Is premise: {premise}? \\n (A) Yes (B) Maybe (C) No'
        answer_string = f"{answer} {explanation_sep} {abstr_expl}"  
    elif io_format == 'unifiedqa_ynm': 
        input_string = f'explain is this entailment? \\n {hypothesis.lower()} {premise.lower()}'  
        answer = unified_qa_esnli_label_mapping[item["label"]]
        answer_string = f"{answer} {explanation_sep} {abstr_expl.lower()}"  
    elif io_format == 'unifiedqa_snli_mix_ynm': 
        input_string = f'explain is this entailment? \\n hypothesis: {hypothesis.lower()} premise: {premise.lower()}' 
        answer = unified_qa_esnli_label_mapping[item["label"]]
        answer_string = f"{answer} {explanation_sep} {abstr_expl.lower()}"  
    elif io_format == 'unifiedqa_snli_mix_ynm_with_choices': 
        input_string = f'explain is this entailment? \\n (A) yes (B) maybe (C) no \\n hypothesis: {hypothesis.lower()} premise: {premise.lower()}'  
        answer = unified_qa_esnli_label_mapping[item["label"]]
        answer_string = f"{answer} {explanation_sep} {abstr_expl.lower()}"  
    elif io_format == 'unifiedqa_what_v2': 
        input_string = f'explain what is this? \\n {hypothesis.lower()} {premise.lower()}'  
        answer = wt5_esnli_label_mapping[item["label"]]
        answer_string = f"{answer} {explanation_sep} {abstr_expl.lower()}"  
    elif io_format == 'unifiedqa_snli_mix_what_v2': 
        input_string = f'explain what is this? \\n hypothesis: {hypothesis.lower()} premise: {premise.lower()}'  
        answer = wt5_esnli_label_mapping[item["label"]]
        answer_string = f"{answer} {explanation_sep} {abstr_expl.lower()}"  
    elif io_format == 'unifiedqa_snli_mix_what_with_choices_v2': 
        input_string = f'explain what is this? \\n (A) entailment (B) neutral (C) contradiction \\n hypothesis: {hypothesis.lower()} premise: {premise.lower()}'  
        answer = wt5_esnli_label_mapping[item["label"]]
        answer_string = f"{answer} {explanation_sep} {abstr_expl.lower()}"     
    else:
        raise ValueError("The IO format is not supported.")

    return input_string, answer_string


def sbic_formatting(item, io_format, explanation_sep):
    # We pre-processed the SBIC dataset such that we join multiple implied statements with the [SEP] token for dev/test instances 
    # Each annotation in the training split is a separate instance
    post = item['post']
    answer = unified_qa_sbic_label_mapping[item["offensiveYN"]] if 'unified' in io_format else item["offensiveYN"].replace("not offensive", "not_offensive")
    abstr_expl = item["targetStereotype"]
    

    if io_format == 'cola':
        input_string = f"explain sbic sentence: {post}"
        answer_string = f"{answer} {explanation_sep} {abstr_expl}"  
    elif io_format == 't5_fewshot_infilling':
        input_string = f"explain sbic sentence: {post} <extra_id_0> {explanation_sep} <extra_id_1>"
        answer_string = f"<extra_id_0> {answer} <extra_id_1> {abstr_expl} <extra_id_2>"
    elif io_format == 't5_fewshot_infilling_bool':
        answer_bool = 'True' if item["offensiveYN"] == 'offensive' else 'False'
        input_string = f"explain sbic sentence: {post} It is <extra_id_0> that this is offensive {explanation_sep} <extra_id_1>"
        answer_string = f"<extra_id_0> {answer} <extra_id_1> {abstr_expl} <extra_id_2>"
    elif io_format == 't5_fewshot_infilling_more_natural':
        input_string = f"explain sbic sentence: {post} This is <extra_id_0> {explanation_sep} <extra_id_1>"
        answer_string = f"<extra_id_0> {answer} <extra_id_1> {abstr_expl} <extra_id_2>"
    elif io_format == 'squad_yn': 
        input_string = f"explain sbic question: Is this offensive? context: {post}" # explain sbic question: Is this offensive? context: It's only happened to you like two times.
        answer_yn = unified_qa_sbic_label_mapping[item["offensiveYN"]] 
        answer_string = f"{answer_yn} {explanation_sep} {abstr_expl}" #  No because this post does not imply anything offensive
    elif io_format == 'squad_yn_with_tags': 
        input_string = f"explain sbic question: Is this offensive? context: post: {post}" # explain sbic question: Is this offensive? context: It's only happened to you like two times.
        answer_yn = unified_qa_sbic_label_mapping[item["offensiveYN"]] 
        answer_string = f"{answer_yn} {explanation_sep} {abstr_expl}" #  No because this post does not imply anything offensive
    elif io_format == 'squad_what': 
        input_string = f"explain sbic question: What is this? context: {post}" # explain sbic question: This is what? context: It's only happened to you like two times.
        answer = item["offensiveYN"].replace("not offensive", "not_offensive")        
        answer_string = f"{answer} {explanation_sep} {abstr_expl}" # not_offensive because this post does not imply anything offensive    
    elif io_format == 'squad_what_with_tags': 
        input_string = f"explain sbic question: What is this? context: post: {post}" # explain sbic question: This is what? context: It's only happened to you like two times.
        answer = item["offensiveYN"].replace("not offensive", "not_offensive")        
        answer_string = f"{answer} {explanation_sep} {abstr_expl}" # not_offensive because this post does not imply anything offensive    
    elif io_format == 'unified_qa_yn':
        input_string = f'explain is this offensive? \\n {post.lower()}'
        answer_string = f"{answer} {explanation_sep} {abstr_expl.lower()}" 
    elif io_format == 'unified_qa_yn_with_tags':
        input_string = f'explain is this offensive? \\n post: {post.lower()}'
        answer_string = f"{answer} {explanation_sep} {abstr_expl.lower()}"  
    elif io_format == 'unified_qa_yn_with_choices':
        input_string = f'explain is this offensive? \\n (A) yes (B) no \\n {post.lower()}'
        answer_string = f"{answer} {explanation_sep} {abstr_expl.lower()}"
    elif io_format == 'unified_qa_yn_with_choices_and_tags':
        input_string = f'explain is this offensive? \\n (A) yes (B) no \\n post: {post.lower()}'
        answer_string = f"{answer} {explanation_sep} {abstr_expl.lower()}"  
    elif io_format == 'unified_qa_what':
        input_string = f'explain what is this? \\n {post.lower()}'
        answer = item["offensiveYN"].replace("not offensive", "not_offensive")
        answer_string = f"{answer} {explanation_sep} {abstr_expl.lower()}"  
    elif io_format == 'unified_qa_what_with_tags':
        input_string = f'explain what is this? \\n post: {post.lower()}'
        answer = item["offensiveYN"].replace("not offensive", "not_offensive")
        answer_string = f"{answer} {explanation_sep} {abstr_expl.lower()}"  
    elif io_format == 'unified_qa_what_with_choices':
        input_string = f'explain what is this? \\n (A) offensive (B) not_offensive \\n {post.lower()}'
        answer = item["offensiveYN"].replace("not offensive", "not_offensive")
        answer_string = f"{answer} {explanation_sep} {abstr_expl.lower()}"
    elif io_format == 'unified_qa_what_with_choices_and_tags':
        input_string = f'explain what is this? \\n (A) offensive (B) not_offensive \\n post: {post.lower()}'
        answer = item["offensiveYN"].replace("not offensive", "not_offensive")
        answer_string = f"{answer} {explanation_sep} {abstr_expl.lower()}"  
    elif io_format == 'unifiedqa_unifew':
        input_string = f"Topic? \\n (A) offensive (B) not_offensive \\n {post}"
        answer = item["offensiveYN"].replace("not offensive", "not_offensive")
        answer_string = f"{answer} {explanation_sep} {abstr_expl}"
    else:
        raise ValueError("The IO format is not supported. Choose `standard` or `masked_cause_generate`.")

    input_string = ' '.join(input_string.split())
    answer_string = ' '.join(answer_string.split())
    return input_string, answer_string

def sensemaking_formatting(item, io_format, explanation_sep):
    # TODO: explore whether removing periods makes difference? 
    sent0 = item['sent0']
    sent1 = item['sent1']
    nonsensical_sentence = str(int(item['label'])+1)
    explanation = item['explanation'].lower()

    if io_format == 'copa_with_question':
        input_string = f"explain sensemaking choice1: {sent0} choice2: {sent1} question: nonsensical"
        answer_string = f"choice{nonsensical_sentence} {explanation_sep} {explanation}"
    elif io_format == 'copa_bool':  
        answer_bool = str(bool(int(item['label']))) # True if choice2 is more nonsensical    
        input_string = f"explain sensemaking choice1: {sent0} choice2: {sent1} Less common is choice2"
        answer_string = f"{answer_bool} {explanation_sep} {explanation}"
    elif io_format == 't5_fewshot_infilling':  
        input_string = f"explain sensemaking choice1: {sent0} choice2: {sent1} <extra_id_0> {explanation_sep} <extra_id_1>"
        answer_string = f"<extra_id_0> choice{nonsensical_sentence} <extra_id_1> {explanation} <extra_id_2>"
    elif io_format == 't5_fewshot_infilling_bool':  
        answer_bool = str(bool(int(item['label']))) # True if choice2 is more nonsensical    
        input_string = f"explain sensemaking choice1: {sent0} choice2: {sent1} It is <extra_id_0> that choice2 is less common {explanation_sep} <extra_id_1>"
        answer_string = f"<extra_id_0> {answer_bool} <extra_id_1> {explanation} <extra_id_2>"
    elif io_format == "squad_yn": 
        input_string = f"explain sensemaking question: Is choice2 more nonsensical? context: choice1: {sent0} choice2: {sent1}" # explain sensemaking question: What is nonsensical, choice1 or choice2? context: choice1: All state flowers are the scarlet carnation. choice2: The New Jersey state flower is the scarlet carnation
        answer = "Yes" if bool(int(item['label'])) else "No"
        answer_string = f"{answer} {explanation_sep} {explanation}" #  choice1 because state flowers are unique to each state.  
    elif io_format == "squad_yn_no_tags": 
        input_string = f"explain sensemaking question: Is choice2 more nonsensical? context: {sent0} {sent1}" # explain sensemaking question: What is nonsensical, choice1 or choice2? context: choice1: All state flowers are the scarlet carnation. choice2: The New Jersey state flower is the scarlet carnation
        answer = "Yes" if bool(int(item['label'])) else "No"
        answer_string = f"{answer} {explanation_sep} {explanation}" #  choice1 because state flowers are unique to each state.  
    elif io_format == "squad_what": 
        input_string = f"explain sensemaking question: What is more nonsensical? context: choice1: {sent0} choice2: {sent1}" # explain sensemaking question: What is nonsensical, choice1 or choice2? context: choice1: All state flowers are the scarlet carnation. choice2: The New Jersey state flower is the scarlet carnation
        answer_string = f"choice{nonsensical_sentence} {explanation_sep} {explanation}" #  choice1 because state flowers are unique to each state.  
    elif io_format == "squad_what_no_tags": 
        input_string = f"explain sensemaking question: What is more nonsensical? context: {sent0} {sent1}" # explain sensemaking question: What is nonsensical, choice1 or choice2? context: choice1: All state flowers are the scarlet carnation. choice2: The New Jersey state flower is the scarlet carnation
        answer_string = f"choice{nonsensical_sentence} {explanation_sep} {explanation}" #  choice1 because state flowers are unique to each state.  
    elif io_format == "record": 
        input_string = f"explain sensemaking query: What is more nonsensical? entities: choice1, choice2 passage: choice1: {sent0} choice2: {sent1}" # explain sensemaking query: What is nonsensical? entities: choice1, choice2 passage: choice1: All state flowers are the scarlet carnation. choice2: The New Jersey state flower is the scarlet carnation.
        answer_string = f"choice{nonsensical_sentence} {explanation_sep} {explanation}" # choice1 because state flowers are unique to each state.
    elif io_format == 'unifiedqa_yn_with_choices':
        answer = "yes" if bool(int(item['label'])) else "no"
        input_string = f'explain is choice2 more nonsensical? \\n (A) yes (B) no \\n choice1: {sent0.lower()} choice2: {sent1.lower()}'
        answer_string = f"{answer} {explanation_sep} {explanation.lower()}" 
    elif io_format == 'unifiedqa_yn':
        answer = "yes" if bool(int(item['label'])) else "no"
        input_string = f'explain is choice2 more nonsensical? \\n choice1: {sent0.lower()} choice2: {sent1.lower()}'
        answer_string = f"{answer} {explanation_sep} {explanation.lower()}"  
    elif io_format == 'unifiedqa_yn_no_tags':
        answer = "yes" if bool(int(item['label'])) else "no"
        input_string = f'explain is choice2 more nonsensical? \\n {sent0.lower()} {sent1.lower()}'
        answer_string = f"{answer} {explanation_sep} {explanation.lower()}"  
    elif io_format == 'unifiedqa_what_with_choices':
        nonsensical_sentence = str(int(item['label'])+1)
        input_string = f'explain what is more nonsensical? \\n (A) choice1 (B) choice2 \\n choice1: {sent0.lower()} choice2: {sent1.lower()}'
        answer_string = f"choice{nonsensical_sentence} {explanation_sep} {explanation.lower()}"  # use " BECAUSE "
    elif io_format == 'unifiedqa_what':
        nonsensical_sentence = str(int(item['label'])+1)
        input_string = f'explain what is more nonsensical? \\n choice1: {sent0.lower()} choice2: {sent1.lower()}'
        answer_string = f"choice{nonsensical_sentence} {explanation_sep} {explanation.lower()}"  # use " BECAUSE "
    elif io_format == 'unifiedqa_what_no_tags':
        nonsensical_sentence = str(int(item['label'])+1)
        input_string = f'explain what is more nonsensical? \\n {sent0.lower()} {sent1.lower()}'
        answer_string = f"choice{nonsensical_sentence} {explanation_sep} {explanation.lower()}"  # use " BECAUSE "


    input_string = ' '.join(input_string.split())
    answer_string = ' '.join(answer_string.split())
    return input_string, answer_string