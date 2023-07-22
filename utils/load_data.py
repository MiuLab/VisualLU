import copy
import json

from PIL import Image
from tqdm import tqdm


def get_data_from_json(json_path, label_dict, max_length=77):
    print(f'Read data from {json_path}')

    with open(json_path, 'r') as f:
        data_jsons = json.load(f)
        
    data_lines = []
    for i in range(len(data_jsons)):
        data_json = data_jsons[f'{i}']
        sentence = data_json['sentence']
        start = data_json['start']
        end = data_json['end']
        label = data_json['label']
        image_path = data_json['image_path']
        
        data_lines.append([sentence, (start, end), label, image_path])
    
    
    data = []
    for i, line in tqdm(enumerate(data_lines)):
        try:
            sentence, (start, end), label, image_path = line
        except:
            print(i, line)

        if label not in label_dict:
            continue
        
        if len(sentence) >= max_length:
            prompt = ' '.join(sentence[:max_length - 1]) + '.'
        else:
            prompt = sentence
        image = Image.open(image_path)
        
        data.append({
            'text': prompt,
            'label': label_dict[label],
            'image': copy.deepcopy(image)
        })

        image.close()

    return data


def clip_batchify(data, batch_size):
    batches = []
    
    for i in range(0, len(data), batch_size):
        text_batch = [d['text'] for d in data[i : i + batch_size]]
        image_batch = [d['image'] for d in data[i : i + batch_size]]
        label_batch = [d['label'] for d in data[i : i + batch_size]]
        
        batches.append((text_batch, image_batch, label_batch))
    
    return batches