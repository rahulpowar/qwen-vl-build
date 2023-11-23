#!/usr/bin/env python3

# Based on the README.md in Qwen-VL
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch

# Set seed for reproducibility
torch.manual_seed(1234)

if __name__ == '__main__':
    model_name = "Qwen/Qwen-VL-Chat-Int4"

    # Load model and tokenizer, use the Int4 variant
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # use bf16
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
    
    # use fp16
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
    
    # use cpu only
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
    
    # use cuda device
    # NOTE: cuda device is required for Multi Modal queries
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", trust_remote_code=True).eval()

    # Specify hyperparameters for generation
    model.generation_config = GenerationConfig.from_pretrained(model_name, trust_remote_code=True)

    # Encode query
    query = tokenizer.from_list_format([
        {'image': 'https://www.looper.com/img/gallery/heres-who-played-darth-vader-without-his-helmet/intro-1566225818.jpg'}, # Either a local path or an url
        {'text': 'What is this?'},
    ])

    # Generate response
    response, _ = model.chat(tokenizer, query=query, history=None)
    print(response)