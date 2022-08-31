import json
import torch
from torch.utils.data import Dataset
from operator import itemgetter

import numpy as np
import random

import pandas as pd
import csv

from tqdm import tqdm

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Seq2Seq_dataset(Dataset):

    def __init__(self, data_path, tokenizer,test=False):
        self.data = []
        self.tokenizer = tokenizer
        self.test = test
        with open(data_path) as f:
            for line in f:
                one_data = json.loads(line)

                input_text = one_data['text']

                spos = one_data['spo_list']

                output_texts = []

                for spo in spos:
                    head = spo['h']['name']
                    tail = spo['t']['name']
                    relation = spo['relation']
                    output_texts.append("头:" + head + "$$关系:" + relation + "$$尾:" + tail)

                for output_text in output_texts:
                    if self.test:
                        self.data.append({
                            'condition' : input_text,
                            'context': output_text,
                        })
                    else:
                        self.data.append({
                            'condition' : input_text,
                            'context': output_text,
                        })

    def __len__(self):
        return  len(self.data)

    def __getitem__(self,index):
        return self.data[index]

    def collate_fn(self,batch):
        source = self.tokenizer([data['condition'] for data in batch], return_tensors='pt', padding=True)
        target = self.tokenizer([data['context'] for data in batch], return_tensors='pt', padding=True) if not self.test else None


        # Convert inputs to PyTorch tensors
        source_tokens = source['input_ids']
        source_masks = source['attention_mask']

        target_tokens = target['input_ids'] if not self.test else None
        target_masks = target['attention_mask'] if not self.test else None

        tensor_data = {
            "source_tokens": source_tokens,
            "source_masks": source_masks,
            "target_tokens": target_tokens,
            "target_masks": target_masks
        }

        return {"meta_data":batch, "tensor_data":tensor_data}