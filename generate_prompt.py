import argparse
import json
import os

from utils.get_label_set import get_label_set


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_root", type=str, default="./data")
    parser.add_argument("-p", "--prompt_json_path", type=str, default="./prompt.json")
    parser.add_argument("-t", "--task", type=str, choices=["conll2003", "bc5cdr", "wnut2017", "mitmovie"], required=True)
    
    args = parser.parse_args()
    return args


def extract_prompts_from_sentence(sentence, ners, label_set):
    prompts = []
    prompt_ners = []
    
    for (start, end, label) in ners:
        if label in label_set:
            prompt = ["A", "photo", "of"] + sentence[start : end + 1] + ["."]
            ner = [3, 3 + (end - start), label]
            
            prompts.append(prompt)
            prompt_ners.append(ner)
    
    return prompts, prompt_ners
        

def generate_prompt(data_path, label_set):
    with open(data_path) as f:
        data_jsons = [json.loads(line) for line in f.readlines()]
    
    all_prompts = []
    all_prompt_ners = []
    for data_json in data_jsons:
        sentences = data_json['sentences']
        ners_list = data_json['ner']
        
        for sentence, ners in zip(sentences, ners_list):
            prompts, prompt_ners = extract_prompts_from_sentence(sentence, ners, label_set)
            
            all_prompts += prompts
            all_prompt_ners += prompt_ners
    
    lines = [
        json.dumps({'sentence': prompt, 'ner': ner})
        for prompt, ner in zip(all_prompts, all_prompt_ners)
    ]
    
    return lines


def main(args):
    data_root = args.data_root
    prompt_json_path = args.prompt_json_path
    task = args.task
    
    label_set = get_label_set(task)
    
    data_path = os.path.join(data_root, task, f'all_data.json')
    lines = generate_prompt(data_path, label_set)
        
    with open(prompt_json_path, 'w+') as f:
        f.write('\n'.join(lines))


if __name__ == "__main__":
    args = get_args()
    main(args)