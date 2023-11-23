#!/usr/bin/env python3

# Based on the README.md in Qwen-VL
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import argparse
from peft import AutoPeftModelForCausalLM

        
# Set seed for reproducibility
torch.manual_seed(1234)

def parse():
    # default model is the Int4 model
    default_model = "Qwen/Qwen-VL-Chat-Int4"

    parser = argparse.ArgumentParser(description='Multi Modal demo for Qwen-VL')
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument('--model', type=str, default=default_model, help='Model name')
    model_group.add_argument('--lora-model', type=str, default=None, help='Model name for LoRA (Adapter model)')
    parser.add_argument('--prompt', type=str, default='What is this?', help='Prompt to use')
    parser.add_argument('image_files', nargs='*', help='Image URLs to explain')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse()

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if len(args.image_files) == 0:
        print("No image files specified, exiting after model load")
        exit(0)

    # use cuda device
    # NOTE: cuda device is required for Multi Modal queries
    model = None
    if args.lora_model is None:
        # normal loader
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map='cuda', trust_remote_code=True).eval()
    else:
        # use LoRA with peft loader
        model = AutoPeftModelForCausalLM.from_pretrained(args.lora_model, device_map="cuda", trust_remote_code=True).eval()


    # Specify hyperparameters for generation
    model.generation_config = GenerationConfig.from_pretrained(args.model, trust_remote_code=True)

    for img in args.image_files:
        # Encode query
        query = tokenizer.from_list_format([
            {'image': img}, # Either a local path or an url
            {'text': args.prompt},
        ])

        # Generate response
        response, _ = model.chat(tokenizer, query=query, history=None)
        print(response)