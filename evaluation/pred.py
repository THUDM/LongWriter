import requests
import time, os, json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import random
import codecs
import argparse
from copy import deepcopy
from tqdm import tqdm
import traceback
import re
import torch.distributed as dist
import torch.multiprocessing as mp

def count_words(text):
    chinese_characters = re.findall(r'[\u4e00-\u9fff]', text)
    english_words = re.findall(r'\b[a-zA-Z]+\b', text)
    
    chinese_char_count = len(chinese_characters)
    english_word_count = len(english_words)
    
    total_count = chinese_char_count + english_word_count
    
    return total_count

def get_pred(rank, world_size, data, path, max_new_tokens, temperature, tokenizer, fout):
    device = torch.device(f'cuda:{rank}')
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    model = model.eval()
    for dt in data:
        prompt = dt['prompt']
        if "llama" in path.lower():
            prompt = f"[INST]{prompt}[/INST]"
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            context_length = input.input_ids.shape[-1]
            output = model.generate(
                **input,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=True,
                temperature=temperature,
            )[0]
            response = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        else:
            response, history = model.chat(tokenizer, prompt, history=[], max_new_tokens=max_new_tokens, temperature=temperature)
        dt["response_length"] = count_words(response)
        dt["response"] = response
        fout.write(json.dumps(dt, ensure_ascii=False)+'\n')
        fout.flush()
        print(response)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    seed_everything(42)
    model = 'LongWriter-glm4-9b' # LongWriter-llama3.1-8b
    path = "THUDM/LongWriter-glm4-9b" # THUDM/LongWriter-llama3.1-8b
    os.makedirs(f"models/{model}", exist_ok=True)
    fout = open(f"models/{model}/pred.jsonl", 'w', encoding='utf-8')

    max_new_tokens = 32768
    temperature = 0.5
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    world_size = torch.cuda.device_count()

    with open('longbench_write.jsonl', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    data_subsets = [data[i::world_size] for i in range(world_size)]
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], path, max_new_tokens, temperature, tokenizer, fout))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()