from transformers import AutoTokenizer, AutoModel, LlamaTokenizer
import copy
import torch
import json, os, random
import multiprocessing
from tqdm import tqdm
import traceback
import numpy as np
import argparse

tokenizer = AutoTokenizer.from_pretrained("THUDM/LongWriter-llama3.1-8b", trust_remote_code=True, use_fast=False)
max_length = 32768
PAD_ID = 128004
BOS_ID = tokenizer.bos_token_id
EOS_ID = tokenizer.eos_token_id
skip_exceed_length_case = True
truncate_side = 'right'

PROC_NUM = 64
save_dir = 'multiprocess_data'

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="llama", type=str)
    return parser.parse_args(args)

def process_file(lines, rank, args):
    def build_input(conversations, tokenizer, args):
        zero_width_chars = ["\u200b", "\u200c", "\u200d", "\ufeff"] # filter null characters
        for conv in conversations:
            if conv['role'] == "assistant":
                for char in zero_width_chars:
                    conv['content'] = conv['content'].replace(char, '')

        if len(conversations) == 0:
            return None

        inputs = torch.full((1,), BOS_ID, dtype=torch.int64)
        starts = []
        ends = []
        for item in conversations:
            content = item["content"]
            role = item["role"]
            if role == 'system':
                cur_inputs = tokenizer(f"<<SYS>>\n{content}\n<</SYS>>\n\n", return_tensors="pt")['input_ids'][0]
            elif role == "user":
                cur_inputs = tokenizer(f"[INST]{content}[/INST]", return_tensors="pt")['input_ids'][0]
            else:
                starts.append(inputs.shape[0])
                cur_inputs = tokenizer(content, return_tensors="pt")['input_ids'][0]
                ends.append(inputs.shape[0] + cur_inputs.shape[0])
            inputs = torch.cat([inputs, cur_inputs], dim=0)
            
        inputs = torch.cat([inputs, torch.tensor([EOS_ID])], dim=0)
        labels = torch.full_like(inputs, -100)
        for start, end in zip(starts, ends):
            labels[start:end] = inputs[start:end]
            labels[end] = EOS_ID

        if inputs.shape[0] > max_length:
            print("exceed_length")
            if skip_exceed_length_case:
                return None
            if truncate_side == 'right':
                inputs = inputs[:max_length]
                labels = labels[:max_length]
            elif truncate_side == 'left':
                cut_num = inputs.shape[0] - max_length
                inputs = torch.cat([inputs[:2], inputs[2 + cut_num:]], dim=0)
                labels = torch.cat([labels[:2], labels[2 + cut_num:]], dim=0)
            else:
                raise ValueError('truncate_side must be "right" or "left"')
        return inputs, labels

    try:
        final_inputs = torch.full((len(lines), max_length), PAD_ID, dtype=torch.int64)
        final_labels = torch.full((len(lines), max_length), -100, dtype=torch.int64)
        pass_data_num = 0

        for line in tqdm(lines):
            conversations = json.loads(line)['messages']
            tmp = build_input(conversations, tokenizer, args)
            if tmp is None:
                continue
            inputs, labels = tmp
            final_inputs[pass_data_num, :inputs.shape[0]] = inputs
            final_labels[pass_data_num, :labels.shape[0]] = labels
            pass_data_num += 1
        final_inputs = final_inputs[:pass_data_num]
        final_labels = final_labels[:pass_data_num]
        torch.save(final_inputs, os.path.join(save_dir, f'inputs_{rank}.pt'))
        torch.save(final_labels, os.path.join(save_dir, f'labels_{rank}.pt'))
    except Exception:
        with open('error.txt', 'a') as f:
            traceback.print_exc(file=f)

def main(args):
    final_dir = f'data/llama3/longwriter'
    os.system('rm -r {}'.format(save_dir))
    os.makedirs(final_dir, exist_ok=True)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    manager = multiprocessing.Manager()
    lines = manager.list()

    general = open('general.jsonl', encoding='utf-8').readlines() # TODO: your general sft data
    longwriter = open('LongWriter-6k.jsonl', encoding='utf-8').readlines()
    lines = general + longwriter
    random.shuffle(lines)
    total_lines = len(lines)
    print(total_lines)

    pool = multiprocessing.Pool(processes=PROC_NUM)
    lines_per_process = total_lines // PROC_NUM

    for i in range(PROC_NUM):
        start_line = i * lines_per_process
        end_line = None if i == PROC_NUM - 1 else (i + 1) * lines_per_process
        pool.apply_async(process_file, args=(lines[start_line:end_line], i, args))

    pool.close()
    pool.join()

    inputs, labels = [], []
    for i in tqdm(range(PROC_NUM)):
        inputs.append(torch.load(os.path.join(save_dir, f'inputs_{i}.pt')))
        labels.append(torch.load(os.path.join(save_dir, f'labels_{i}.pt')))
    inputs = torch.cat(inputs, dim=0)
    labels = torch.cat(labels, dim=0)

    input_ids = inputs.numpy().astype(np.int64)
    labels = labels.numpy().astype(np.int64)
    filtered_rows = np.where(~np.all(labels == -100, axis=1))[0]
    input_ids = input_ids[filtered_rows]
    labels = labels[filtered_rows]

    print(labels.shape)
    np.save(os.path.join(final_dir, 'inputs.npy'), input_ids)
    np.save(os.path.join(final_dir, 'labels.npy'), labels)

if __name__ == '__main__':
    main(parse_args())