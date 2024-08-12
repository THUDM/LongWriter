import torch
import os
import json
import numpy as np
from torch.utils.data import DataLoader

class LMDataset(torch.utils.data.Dataset):
    def __init__(self, filepath):
        self.input_ids, self.labels = self.process_data(filepath)
        self.input_ids = self.input_ids
        self.labels = self.labels

    def process_data(self, filepath):
        input_ids = torch.from_numpy(np.load(os.path.join(filepath, 'inputs.npy')))
        labels = torch.from_numpy(np.load(os.path.join(filepath, 'labels.npy')))
        return input_ids, labels

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return self.input_ids.size(0)

class LMSortDataset(torch.utils.data.Dataset):
    def __init__(self, filepath):
        self.input_ids, self.labels = self.process_data(filepath)
        self.input_ids = self.input_ids
        self.labels = self.labels
    
    def process_data(self, filepath):
        input_ids = torch.from_numpy(np.load(os.path.join(filepath, 'inputs_sort.npy')))
        labels = torch.from_numpy(np.load(os.path.join(filepath, 'labels_sort.npy')))
        return input_ids, labels

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return self.input_ids.size(0)

class LMPackDataset(torch.utils.data.Dataset):
    def __init__(self, filepath):
        self.input_ids, self.attention_masks, self.labels, self.weights, self.nums = self.process_data(filepath)
        self.num_gpus = torch.cuda.device_count()
        
    def process_data(self, filepath):
        input_ids = torch.from_numpy(np.load(os.path.join(filepath, 'inputs_pack.npy')))
        labels = torch.from_numpy(np.load(os.path.join(filepath, 'labels_pack.npy')))
        weights = torch.from_numpy(np.load(os.path.join(filepath, 'weights_pack.npy')))
        attention_masks = json.load(open(os.path.join(filepath, 'attention_masks_pack.json')))
        num_gpus = torch.cuda.device_count()
        l = (input_ids.size(0) // num_gpus) * num_gpus
        input_ids, labels, weights, attention_masks = input_ids[:l, :], labels[:l, :], weights[:l, :], attention_masks[:l]
        nums = [weights[i*num_gpus:(i+1)*num_gpus, :].sum() for i in range(l//num_gpus)]
        return input_ids, attention_masks, labels, weights, nums

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.int32),
            'labels': (self.labels[idx], self.weights[idx], self.nums[idx//self.num_gpus])
        }

    def __len__(self):
        return self.input_ids.size(0)