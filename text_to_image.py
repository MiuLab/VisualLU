import os
import json
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
import csv

import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def generate_valid_image(
    pipe,
    num_inference_steps,
    prompt,
    image_path,
    seed
):  
    valid_image = False
    while not valid_image:
        set_seed(seed)
        output = pipe(prompt, num_inference_steps=num_inference_steps)
        image = output.images[0]
        nsfw_content_detected = output.nsfw_content_detected
        if not nsfw_content_detected:
            valid_image = True
            image.save(image_path)
        seed += 1


def main(args):
    device = args.device
    num_inference_steps = args.num_inference_steps

    scheduler = EulerDiscreteScheduler.from_pretrained(args.model_id,
                                                       subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(args.model_id,
                                                   scheduler=scheduler,
                                                   revision="fp16",
                                                   torch_dtype=torch.float16)
    pipe = pipe.to(device)
    
    with open(args.prompt_json_path, encoding="utf-8") as f:
        prompt_jsons = [json.loads(line) for line in f.readlines()]
    
    data = {}
    for idx, prompt_json in tqdm(enumerate(prompt_jsons), desc="Generate images"):
        prompt = prompt_json["sentence"]
        ner = prompt_json["ner"]
        start, end, label = ner
        image_path = f"{args.output_image_dir}/{idx}.png"
        
        if not os.path.isfile(image_path):
            generate_valid_image(pipe, num_inference_steps, prompt, image_path, args.seed)

        data[idx] = {
            "sentence": prompt,
            "start": start,
            "end": end,
            "label": label,
            "image_path": image_path
        }
    
    with open(args.data_json, 'w+') as f:    
        json.dump(data, f, indent=4)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--prompt_json_path",
        type=Path,
        help="Path to the text data.",
        default="./prompt.json",
    )
    parser.add_argument(
        "-m",
        "--model_id",
        type=str,
        help="Pretrain model name.",
        default="stabilityai/stable-diffusion-2-base",
    )
    parser.add_argument(
        "-o",
        "--output_image_dir",
        type=Path,
        help="Directory to the output images.",
        default="./image_root",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="random seed",
        default=0
    )
    parser.add_argument(
        "-d",
        "--device",
        type=torch.device,
        help="cpu, cuda, cuda:0, cuda:1",
        default="cuda"
    )
    parser.add_argument(
        "-n",
        "--num_inference_steps",
        type=int,
        default=50
    )
    parser.add_argument(
        "--data_json",
        type=Path,
        default="./data.json"
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        required=True
    )
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.output_image_dir.mkdir(exist_ok=True, parents=True)
    main(args)