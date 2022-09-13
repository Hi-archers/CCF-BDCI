import json
import torch
from torch.utils.data import Dataset

import numpy as np
import random

import pandas as pd
import csv

from tqdm import tqdm
import re
import string
import collections
# from evaluate import f1_score

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
                
                if self.test:
                    self.data.append({
                        'condition' : input_text,
                        'context': None,
                        'id':one_data['ID']
                    })
                    continue

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
                            'id':one_data['ID']
                        })
                    else:
                        self.data.append({
                            'condition' : input_text,
                            'context': output_text,
                            'id':one_data['ID']
                        })

    def __len__(self):
        return  len(self.data)

    def __getitem__(self,index):
        return self.data[index]

    def collate_fn(self,batch):
        source = self.tokenizer([data['condition'] for data in batch], return_tensors='pt',padding='max_length',truncation=True, max_length = 510)
        target = self.tokenizer([data['context'] for data in batch], return_tensors='pt',padding='max_length',truncation=True, max_length = 510) if not self.test else None


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



class L_dataset(Dataset):
    def __init__(self, data_path, tokenizer,test = False):
        self.data = []
        self.tokenizer = tokenizer
        self.test = test

        with open(data_path, 'r', encoding='utf-8') as f:
            input_data = csv.reader(f)
            First = False
            for line in input_data:
                if not First:
                    First = True
                    continue
                one_data = {}

                one_data['sentence'] = line[0]
                one_data['label'] = min(int(line[1]),1)

                self.data.append(one_data)

    def __len__(self):
        return  len(self.data)

    def __getitem__(self,index):
        return self.data[index]

    def collate_fn(self,batch):
        source = self.tokenizer([data['sentence'] for data in batch], return_tensors = 'pt', padding = True)
        target = torch.LongTensor([data['label'] for data in batch])
        # Convert inputs to PyTorch tensors
        source_tokens = source['input_ids']
        source_masks = source['attention_mask']

        tensor_data = {
            "source_tokens": source_tokens,
            "source_masks": source_masks,
            "target": target,
        }

        return {"meta_data":batch, "tensor_data":tensor_data}

def get_F1_score(train_answers_path,train_predict_path):
    answers = []
    predict = []

    # train_answers_path = '/root/CCF-BDCI-main/CCF-BDCI-main/data/split_train.json'
    # train_predict_path = '/root/CCF-BDCI-main/CCF-BDCI-main/code/submission.json'

    with open(train_answers_path, 'r', encoding='utf-8') as f:
        for line in f:
            one_data = json.loads(line)
            answers.append(one_data)

    with open(train_predict_path, 'r', encoding='utf-8') as f:
        for line in f:
            one_data = json.loads(line)
            predict.append(one_data)

    print(f"train answer len = {len(answers)}  train predict len = {len(predict)}")

    flag = True

    for i in range(len(answers)):
        ans = answers[i]
        pre = predict[i]
        if ans["ID"] != pre["ID"]:
            print(f"line {i} ans ID = {ans['ID']} pre ID = {pre['ID']}")
            flag = False
        if ans["text"] != pre["text"]:
            print(f"line {i} ans text = {ans['text']} pre text = {pre['text']}")
            flag = False

    print(50*"-")
    print(f"Data format is {flag}")
    print(50*"-")

    if flag == False:
        exit()

    label1_len = 0  ##部件故障
    label2_len = 0  ##性能故障
    label3_len = 0  ##检测工具
    label4_len = 0  ##组成
    label1_true_len = 0  ##部件故障
    label2_true_len = 0  ##性能故障
    label3_true_len = 0  ##检测工具
    label4_true_len = 0  ##组成关系
    label1_pre_len = 0  ##部件故障
    label2_pre_len = 0  ##性能故障
    label3_pre_len = 0  ##检测工具
    label4_pre_len = 0  ##组成
    for i in range(len(answers)):
        ans = answers[i]
        pre = predict[i]

        ans_spo_list = ans['spo_list']
        pre_spo_list = pre['spo_list']
        pre_flag = False
        for j in ans_spo_list:
            if pre_spo_list != [] and j['t']['name'] == pre_spo_list[0]['t']['name'] and j['t']['pos'] == pre_spo_list[0]['t']['pos'] and j['h']['name'] == pre_spo_list[0]['h']['name'] and j['h']['pos'] == pre_spo_list[0]['h']['pos'] and j['relation'] == pre_spo_list[0]['relation']:
                pre_flag = True
                if j['relation'] == "部件故障":
                    label1_len+=1
                    label1_true_len+=1
                elif j['relation'] == "性能故障":
                    label2_len+=1
                    label2_true_len+=1
                elif j['relation'] == "检测工具":
                    label3_len+=1
                    label3_true_len+=1
                elif j['relation'] == "组成":
                    label4_len+=1
                    label4_true_len+=1
                continue
            if j['relation'] == "部件故障":
                label1_len+=1
            elif j['relation'] == "性能故障":
                label2_len+=1
            elif j['relation'] == "检测工具":
                label3_len+=1
            elif j['relation'] == "组成":
                label4_len+=1
        if not pre_flag and pre_spo_list != []:
            if pre_spo_list[0]['relation'] == "部件故障":
                label1_pre_len+=1
            elif pre_spo_list[0]['relation'] == "性能故障":
                label2_pre_len+=1
            elif pre_spo_list[0]['relation'] == "检测工具":
                label3_pre_len+=1
            elif pre_spo_list[0]['relation'] == "组成":
                label4_pre_len+=1

    label1_acc = label1_true_len*1.0/label1_pre_len if label1_pre_len != 0 else 0        
    label2_acc = label2_true_len*1.0/label2_pre_len if label2_pre_len != 0 else 0     
    label3_acc = label3_true_len*1.0/label3_pre_len if label3_pre_len != 0 else 0          
    label4_acc = label4_true_len*1.0/label4_pre_len if label4_pre_len != 0 else 0      

    label1_recall = label1_true_len*1.0/label1_len if label1_len != 0 else 0        
    label2_recall = label2_true_len*1.0/label2_len if label2_len != 0 else 0       
    label3_recall = label3_true_len*1.0/label3_len if label3_len != 0 else 0        
    label4_recall = label4_true_len*1.0/label4_len if label4_len != 0 else 0     

    label1_f1 = 2*(label1_acc*label1_recall)/(label1_acc+label1_recall) if label1_acc+label1_recall !=0 else 0       
    label2_f1 = 2*(label2_acc*label2_recall)/(label2_acc+label2_recall) if label2_acc+label2_recall !=0 else 0    
    label3_f1 = 2*(label3_acc*label3_recall)/(label3_acc+label3_recall) if label3_acc+label3_recall !=0 else 0    
    label4_f1 = 2*(label4_acc*label4_recall)/(label4_acc+label4_recall) if label4_acc+label4_recall != 0 else 0

    F1_Score_Micro = (label1_f1*label1_len+label1_f1*label2_len+label3_f1*label3_len+label4_f1*label4_len)/(label1_len+label2_len+label3_len+label4_len)
    print(F1_Score_Micro)