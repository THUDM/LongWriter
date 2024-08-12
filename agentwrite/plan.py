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

GPT4_API_KEY = ''
GPT_MODEL = 'gpt-4o-2024-05-13'
def get_response_gpt4(prompt, max_new_tokens=1024, temperature=1.0, stop=None):
    tries = 0
    while tries < 10:
        tries += 1
        try:
            headers = {
                'Authorization': "Bearer {}".format(GPT4_API_KEY),
            }
            messages = [
                {'role': 'user', 'content': prompt},
            ]
            resp = requests.post("https://api.openai.com/v1/chat/completions", json = {
                "model": GPT_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_new_tokens,
                "stop": stop,
            }, headers=headers, timeout=600)
            if resp.status_code != 200:
                raise Exception(resp.text)
            resp = resp.json()
            break
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            if "maximum context length" in str(e):
                raise e
            elif "triggering" in str(e):
                return 'Trigger OpenAI\'s content management policy'
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
    else:
        print("Max tries. Failed.")
        return "Max tries. Failed."
    try:
        return resp["choices"][0]["message"]["content"]
    except: 
        return ''

def get_pred(rank, world_size, data, max_new_tokens, fout, template):
    for item in tqdm(data):
        prompt = item['prompt']
        prompt = template.replace('$INST$', prompt)
        try:
            response = get_response_gpt4(prompt, max_new_tokens)
            item["plan"] = response
            fout.write(json.dumps(item, ensure_ascii=False)+'\n')
            fout.flush()
        except Exception as e:
            print(e)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    # input format: {"prompt": "xxx", ...}
    # output format: {"prompt": "xxx", "plan": "xxx", ...}
    in_file = 'instructions.jsonl'
    out_file = 'plan.jsonl'
    seed_everything(42)
    max_new_tokens = 4096
    world_size = 8
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["prompt"]: 0 for line in f}
    fout = open(out_file, 'a', encoding='utf-8')
    data = []
    with open(in_file, encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if item["prompt"] not in has_data:
                data.append(item)
    template = open('prompts/plan.txt', encoding='utf-8').read()

    data_subsets = [data[i::world_size] for i in range(world_size)]
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], max_new_tokens, fout, template))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()