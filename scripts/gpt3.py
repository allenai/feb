import os
import random
import transformers


gpt3_esnli_label_mapping = {0: 'Yes', 1: 'Maybe', 2: 'No'}  # Tried True, False, Neither but this works better.
gpt3_t5_esnli_label_mapping = {'Yes': 'entailment', 'Maybe': 'neutral', 'No': 'contradiction'}

gpt3_sbic_label_mapping = {"offensive": 'Yes', "not offensive": 'No'}


def _get_task_description(task):
    if task == 'sbic':
        return "Is the following post offensive? and why?\n"
    elif task == 'esnli':
        return "Answer the question and provide a reason why the answer is correct.\n"
    elif task == 'cos_e':
        return "Answer the question from the provided choices, and provide a reason why the answer is correct.\n"
    elif task == 'sensemaking':
        return "Which of the two choices makes more sense? and why?\n"
    else:
        assert False


def _convert_instance_to_prompt(instance, task):
    if task == 'sbic':
        post = instance['post'].replace('\n', ' ')  # a few posts have new lines
        input = f"Post: {post}\n" \
                 "Answer:"
        output = f" {gpt3_sbic_label_mapping[instance['offensiveYN']]}\n" \
                 f"Reason: {instance['targetStereotype']}\n" \
                  "###\n"
    elif task == 'esnli':
        input = f"{instance['premise']}\n" \
                f"Question: Is {instance['hypothesis'].lower().replace('.', '')}?\n" \
                 "Answer:"
        output = f" {gpt3_esnli_label_mapping[instance['label']]}\n" \
                 f"Reason: {instance['explanation_1']}\n" \
                  "###\n"
    elif task == 'cos_e':
        choices = ', '.join(instance['choices'])
        input = f"Question: {instance['question']}\n" \
                f"Choices: {choices}\n" \
                 "Answer:"
        output = f" {instance['answer']}\n" \
                 f"Reason: {instance['abstractive_explanation']}\n" \
                  "###\n"
    elif task == 'sensemaking':
        input = f"choice1: {instance['sent0']}\n" \
                f"choice2: {instance['sent1']}\n" \
                 "Answer:"
        # flip choices to select the one that makes "more" sense. GPT3 works much better with this prompt
        label = 1 if instance['label'] == 1 else 2
        output = f" choice{label}\n" \
                 f"Reason: {instance['explanation']}\n" \
                  "###\n"
    else:
        assert False

    return input, output


def _parse_response(response_text, task):
    if task in ['cos_e', 'esnli', 'sensemaking', 'sbic']:
        if 'Reason:' not in response_text:
            print('Invalid output format')  # rarely happens
            pred_answer = 'wrong'
            pred_explanation = response_text.replace('\n', '')
        else:
            splits = response_text.split('Reason:', 1)
            pred_answer = splits[0].strip()
            pred_explanation = splits[1].strip()
            pred_explanation = pred_explanation.split('\n')[0]
            if task == 'esnli':
                pred_answer = gpt3_t5_esnli_label_mapping.get(pred_answer) or pred_answer
            if task == 'sensemaking':
                # flip it back
                pred_answer = pred_answer.replace('1', '2') if '1' in pred_answer else pred_answer.replace('2', '1')
    else:
        assert False

    return pred_answer, pred_explanation


def run_gpt3(train_data, test_data, task, generations_file, explanation_sep, save_path):
    import openai
    openai.api_key = os.environ['OPENAI_KEY']
    gpt3_version = "davinci-instruct-beta"  # davinci
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
    max_output_tokens = 75  # covers ~=99% of the examples

    gpt3_log_dir = os.path.join(save_path, 'gpt3_log')
    if os.path.isdir(gpt3_log_dir):
        raise False
    else:
        os.mkdir(gpt3_log_dir)

    task_description = _get_task_description(task)
    train_prompts = [_convert_instance_to_prompt(instance, task) for instance in train_data]
    test_prompts = [_convert_instance_to_prompt(instance, task) for instance in test_data]

    with open(generations_file, 'w') as w:
        total_num_tokens = []
        output_num_tokens = []
        count_correct_predictions = 0
        for j, (test_instance_input, test_instance_output) in enumerate(test_prompts):

            # Each test instance uses a different set of training instances
            random.shuffle(train_prompts)
            header = task_description

            # Pack as many training examples as possible in the prompt (usually 30-45 examples depending on the dataset)
            #  start with seqlen enough for: task description + test instance input + maximum output length
            #  then add training examples until the 2049 are filled
            # We don't know for sure what tokenizer gpt3 is using but it is likely transformers.GPT2TokenizerFast
            current_seqlen = len(tokenizer.tokenize(f'{header}{test_instance_input}')) + max_output_tokens
            for i, (train_instance_input, train_instance_output) in enumerate(train_prompts):
                train_instance_seqlen = len(tokenizer.tokenize(f'{train_instance_input}{train_instance_output}'))
                if current_seqlen + train_instance_seqlen > 2045:  # max seqlen of gpt3 is 2049
                    break
                header = header + train_instance_input + train_instance_output
                current_seqlen += train_instance_seqlen

            prompt = f'{header}{test_instance_input}'
            output_num_tokens.append(len(tokenizer.tokenize(test_instance_output)))
            total_num_tokens.append(len(tokenizer.tokenize(prompt)) + max_output_tokens)

            # Number of training examples, prompt sequence length, input, gold output
            print(f'{i}-{current_seqlen}-{test_instance_input}{test_instance_output}')
            response = openai.Completion.create(engine=gpt3_version, prompt=prompt, max_tokens=max_output_tokens, temperature=0.0)
            response_text = response['choices'][0]['text']
            print(response_text)  # predicted output
            with open(os.path.join(gpt3_log_dir, f'{j}.txt'), 'w') as flog:
                flog.write(prompt)
                flog.write('''\n\n''')
                flog.write(response_text)
            pred_answer, pred_explanation = _parse_response(response_text, task)
            w.write(f'{pred_answer}{explanation_sep}{pred_explanation}\n')

            # # The following is a hack to only run on `neutral` labels of `esnli` to get data for human eval
            # gold_answer, gold_explanation = _parse_response(test_instance_output, task)
            # if gold_answer.lower() == pred_answer.lower():
            #     count_correct_predictions += 1
            #     print(f'=========={count_correct_predictions} / {j} = {count_correct_predictions / j}, {len(test_prompts)}')
            #     if count_correct_predictions == 120:
            #         break
