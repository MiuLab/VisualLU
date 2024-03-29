import argparse
import json
import os
import sys

import numpy as np
import torch
from easydict import EasyDict as edict
from tqdm import tqdm
from transformers import (CLIPProcessor, CLIPTextModel, CLIPTokenizer,
                          CLIPVisionModel)

from clustering.kmeans import get_kmeans
from utils.load_data import clip_batchify, get_data_from_json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("-m", "--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("-d", "--data_json_path", default="./data.json")
    parser.add_argument("-t", "--task", choices=['conll2003', "bc5cdr", "wnut2017", "mitmovie"], required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--kmeans_states", nargs='+', type=int, default=[0, 1, 2, 3, 4])

    args = parser.parse_args()
    return args


def get_confusion_mean(confusions):
    acc_mean = sum([c.acc() for c in confusions]) / len(confusions)
    nmi_mean = sum([c.clusterscores()['NMI'] for c in confusions]) / len(confusions)
    return acc_mean, nmi_mean


def get_features(
    data,
    text_encoder,
    text_tokenizer,
    image_encoder,
    image_processor,
    config
):
    all_results = {}
    for text_feat in ['', 'pooler']:
        for image_feat in ['', 'pooler']:
            if text_feat == '' and image_feat == '':
                continue
            all_results[f'{text_feat}-{image_feat}'] = []
    all_results['labels'] = []

    with torch.no_grad():
        for batch in tqdm(clip_batchify(data, config.batch_size), 
                          ncols=100, file=sys.stdout):
            text_batch, image_batch, label_batch = batch
            
            text_inputs = text_tokenizer(text_batch, padding=True, truncation=True,
                                return_tensors="pt", max_length=config.max_length).to(config.device)
            text_outputs = text_encoder(**text_inputs)
            
            image_inputs = image_processor(images=image_batch,
                                return_tensors="pt").to(config.device)
            image_outputs = image_encoder(**image_inputs)

            for text_feat in ['', 'pooler']:
                for image_feat in ['', 'pooler']:
                    if text_feat == '' and image_feat == '':
                        continue
                    feat = get_feat(text_outputs, image_outputs, text_feat, image_feat)
                    all_results[f'{text_feat}-{image_feat}'].append(feat)
            all_results['labels'] += label_batch

    for k, v in all_results.items():
        if k != 'labels':
            all_results[k] = torch.cat(v, dim=0)
    all_results['labels'] = torch.LongTensor(all_results['labels'])

    return all_results


def get_feat(text_outputs, image_outputs, text_feat, image_feat):
    feats = []
    if text_feat == 'pooler':
        feats.append(text_outputs.pooler_output)
    if image_feat == 'pooler':
        feats.append(image_outputs.pooler_output)
    output = torch.cat(feats, dim=1)
    return output.squeeze().detach().cpu()


def run_kmeans(all_features, kmeans_states, num_classes):
    results = {}
    for text_feat in ['', 'pooler']:
        for image_feat in ['', 'pooler']:
            if text_feat == '' and image_feat == '':
                continue
            features = all_features[f'{text_feat}-{image_feat}'].clone()
            labels = all_features['labels'].clone()

            confusions = [
                get_kmeans(features, labels, num_classes, ks)
                for ks in kmeans_states
            ]
            results[f'{text_feat}-{image_feat}-acc'] = get_confusion_mean(confusions)

    return results


def main(args):
    device = args.device
    task = args.task
    batch_size = args.batch_size
    data_json_path = args.data_json_path
    kmeans_states = args.kmeans_states    
    
    text_encoder = CLIPTextModel.from_pretrained(args.model)
    text_tokenizer = CLIPTokenizer.from_pretrained(args.model)
    image_encoder = CLIPVisionModel.from_pretrained(args.model)
    image_processor = CLIPProcessor.from_pretrained(args.model)
    
    text_encoder.to(device)
    text_encoder.eval()
    image_encoder.to(device)
    image_encoder.eval()

    if task == 'conll2003':
        label_dict = {'PER':0, 'LOC':1, 'ORG':2}
    elif task == 'bc5cdr':
        label_dict = {'Chemical': 0, 'Disease': 1}
    elif task == 'mitmovie':
        label_dict = {'person': 0, 'title': 1}
    elif task == 'wnut2017':
        label_dict = {'corporation': 0, 'creative_work':1, 'group': 2,
                      'location': 3, 'person': 4, 'product': 5}
    else:
        raise NotImplementedError
    num_classes = len(label_dict)

    data = get_data_from_json(data_json_path, label_dict)
    
    config = edict({'batch_size': batch_size, 'device': device, 'max_length': 77})
    all_features = get_features(data, text_encoder, text_tokenizer,
                        image_encoder, image_processor, config)
    results = run_kmeans(all_features, kmeans_states, num_classes)
    
    print(results)


if __name__ == '__main__':
    args = get_args()
    main(args)