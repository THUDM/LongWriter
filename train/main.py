import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset import LMDataset, LMSortDataset, LMPackDataset
from trainer import TrainerNoShuffle

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="THUDM/glm-4-9b")
    pack_loss: bool = field(default=False)

@dataclass
class DataArguments:
    train_file: str = field(default=None, metadata={"help": "Path to the training data."})
    validation_file: str = field(default=None, metadata={"help": "Path to the training data."})
    preprocessing_num_workers: Optional[int] = field(
        default=1,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    prompt_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    response_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    batch_method: str = field(default="naive")

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")

@dataclass
class DataCollatorForLMDataset(object):

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key].unsqueeze(0) for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.cat(input_ids, dim=0)
        labels = torch.cat(labels, dim=0)
        eos_indices = input_ids.argmin(dim=1) - 1
        max_position = eos_indices.max()
        if max_position < 0:
            return dict(
                input_ids=input_ids,
                labels=labels
            )
        return dict(
            input_ids=input_ids[:, :max_position+1],
            labels=labels[:, :max_position+1]
        )

@dataclass
class DataCollatorForLMPackDataset(object):

    def __call__(self, instances):
        input_ids, attention_masks = tuple([instance[key].unsqueeze(0) for instance in instances] for key in ["input_ids", "attention_mask"])
        batch_seq_num = instances[0]["labels"][2]
        labels = ([instance["labels"][0].unsqueeze(0) for instance in instances], [instance["labels"][1].unsqueeze(0) for instance in instances])
        input_ids = torch.cat(input_ids, dim=0)
        labels = (torch.cat(labels[0], dim=0), torch.cat(labels[1], dim=0))
        labels = (labels[0], labels[1].sum()/30)
        max_length = input_ids.shape[1]
        attention_mask = attention_masks[0].squeeze()
        acc_length = max_length
        for new_attention_mask in attention_masks[1:]:
            new_attention_mask = new_attention_mask.squeeze()
            attention_mask = torch.cat([attention_mask, new_attention_mask[1:]+acc_length], dim=0)
            acc_length += max_length
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

def make_supervised_data_module(data_args) -> Dict:
    print("loading data...")
    if data_args.batch_method == "naive":
        train_dataset = LMDataset(data_args.train_file)
        data_collator = DataCollatorForLMDataset()
    elif data_args.batch_method == "pack":
        train_dataset = LMPackDataset(data_args.train_file)
        data_collator = DataCollatorForLMPackDataset()
    elif data_args.batch_method == "sort":
        train_dataset = LMSortDataset(data_args.train_file)
        data_collator = DataCollatorForLMDataset()
    print("finish loading data")
    return dict(train_dataset=train_dataset, data_collator=data_collator)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if "chatglm" in model_args.model_name_or_path.lower() or "longalign-6b" in model_args.model_name_or_path.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True, empty_init=False
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, 
                                          torch_dtype=torch.bfloat16, 
                                          trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                                  trust_remote_code=True)
    if model_args.pack_loss:
        model.pack_loss = True
    data_module = make_supervised_data_module(data_args=data_args)

    trainer = TrainerNoShuffle(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        **data_module
    )

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model()

if __name__ == "__main__":
    train()
