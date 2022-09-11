import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from torch import nn
from transformers import BertTokenizer, BartForConditionalGeneration
import copy
import re
import json
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from utlis import Seq2Seq_dataset, setup_seed

#from tokenizers import SpTokenizer

import matplotlib.pyplot as plt

import logging

logging.basicConfig(level=logging.INFO)

setup_seed(42)

#from data_parallel import BalancedDataParallel
import os

# OUTPUT_DIR = '/root/autodl-tmp/output/'
# if not os.path.exists(OUTPUT_DIR):
#     os.makedirs(OUTPUT_DIR)
def infer(args, logger):
    model_name = args.model_name
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    train_path = args.train_path
    val_path = args.val_path
    #save_file = args.save_file
    epoch_number = args.epoch_number
    learning_rate = args.learning_rate
    multi_gpu = args.multi_gpu
    model_path = args.model_path
    infer_path  = args.infer_path

    acc_grad = 1
    

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
    state = torch.load(model_path,
                       map_location=torch.device('cpu'))
    model.load_state_dict(state['model'])

    preds = []
    model.eval()


    infer_dataset = Seq2Seq_dataset(infer_path, tokenizer,test=True)

   
    infer_dataloader = DataLoader(dataset=infer_dataset, batch_size=val_batch_size, num_workers=1, collate_fn=infer_dataset.collate_fn)

    for i,batch in enumerate(tqdm(infer_dataloader)):
        # print(batch['meta_data'])
        # ids = batch['meta_data']['id']
        # texts = batch['meta_data']['condition']
        tensor_data = batch['tensor_data']
        source_tokens = tensor_data['source_tokens'].to(device)
        source_masks = tensor_data['source_masks'].to(device)

        # if source_tokens.shape[1] >= 512:
        #     continue

        outputs = model.generate(
            source_tokens,
            attention_mask = source_masks,
            max_length = 500,
            num_beams = 10,
            do_sample = False
        )
        outputs_decode=[]
        for i in range(len(outputs)):
            output = ''.join(tokenizer.decode(outputs[i], skip_special_tokens = True)).replace(' ','')
            outputs_decode.append([batch['meta_data'][i]['id'],batch['meta_data'][i]['condition'],output])
        
        preds.extend(outputs_decode)

        
    return preds



def Get_Entity_Pos(text,entity):
    position = text.find(entity)
    if position == -1:
        return [-1,0]
    else:
        return [position,position+len(entity)]
    
def Get_Entity(pred):
    head_entity = re.findall(r'头:(.+?)\$\$关系:',pred)
    tail_entity = re.findall(r'\$\$尾:(.+?)',pred)
    relation = re.findall(r'\$\$关系:(.+?)\$\$尾:',pred)
    if len(head_entity) != 0:
        head_entity = head_entity[0]
    if len(tail_entity) != 0:
        tail_entity = tail_entity[0]
    if len(relation) != 0:
        relation = relation[0]
    return head_entity,relation,tail_entity
def to_submission(predictions):
    file = open(saved_dir+'submission.json', 'w')
    # predictions_list=[]
    for pred in predictions:
        dict_tmp={}
        dict_tmp['ID']=pred[0]
        dict_tmp['text']=pred[1]
        dict_tmp['spo_list']=[]
        head_entity,relation,tail_entity=Get_Entity(pred[2])
        
        if head_entity == [] or relation == [] or tail_entity == []:
            # predictions_list.append(dict_tmp)
            dict_tmp = json.dumps(dict_tmp,ensure_ascii=False)
            file.write(dict_tmp)
            file.write('\n')
            continue
        
        head_pos = Get_Entity_Pos(pred[1],head_entity)
        tail_pos = Get_Entity_Pos(pred[1],tail_entity)
        tmp={}
        tmp['h']={'name':head_entity,'pos':head_pos}
        tmp['t']={'name':tail_entity,'pos':tail_pos}
        tmp['relation']=relation
        dict_tmp['spo_list'].append(tmp)
        # predictions_list.append(dict_tmp)
        dict_tmp = json.dumps(dict_tmp,ensure_ascii=False)
        file.write(dict_tmp)
        file.write('\n')
    # predictions_list_json = json.dumps(predictions_list,ensure_ascii=False)
    # file.write(predictions_list_json)
    file.close()
        
# class args:
#     model_name='fnlp/bart-base-chinese'
#     train_batch_size = 32
#     val_batch_size = 32
#     train_path='/root/CCF-BDCI-main/CCF-BDCI-main/data/split_train.json'
#     val_path='/root/CCF-BDCI-main/CCF-BDCI-main/data/split_valid.json'
#     infer_path='/root/CCF-BDCI-main/CCF-BDCI-main/data/split_train.json'
#     model_path = '/root/autodl-tmp/output/fnlp-bart-base-chinese_best.pth'
#     epoch_number=100
#     learning_rate=2e-5
#     multi_gpu=False

def parse_args():
    parser = argparse.ArgumentParser('Training args...')
    parser.add_argument('--model_name', default='../../model/bart-base-chinese', help='Model name.')
    parser.add_argument('--train_batch_size', default = 32, help='Traning set batch size.')
    parser.add_argument('--val_batch_size', default=1, help='Validation set batch size.')
    parser.add_argument('--train_path', default='../dataset/train.json', help='Traning set file.')
    parser.add_argument('--val_path', default=  '../dataset/.txt', help='Validation set file.')
    parser.add_argument('--save_file', default= './model/NEW_MBART', help='Save model path.')
    parser.add_argument('--epoch_number', default = 200, help='Epoch number.')
    parser.add_argument('--learning_rate', default = 2e-5, help='Learning rate.')
    #parser.add_argument('--beam_size', default=10, help='Size of beam search.')
    #parser.add_argument('--model_parall', default=False, help='Multi-GPU.')
    parser.add_argument('--multi_gpu', default = False, help='Multi-GPU.')

    return parser.parse_args()

import logging
from torch.utils.tensorboard import SummaryWriter
from logging.handlers import TimedRotatingFileHandler

root_logger = logging.getLogger()
root_logger.handlers = []

def get_log(name, log_file=None, console=True, when="D", interval=7):
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level=logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if log_file:
        handler = TimedRotatingFileHandler(log_file, when=when, interval=interval, backupCount=1, encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if console:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)

    return logger

if __name__ == '__main__':

    args = parse_args()

    writer = SummaryWriter("logs/log")
    logger = get_log(__name__, "logs/log.txt")

    logger.info("Start Training！")
    logger.info(f"Optimizer Setting : learning rate = {args.learning_rate} batch size = {args.train_batch_size}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    predictions = infer(args, logger)
    to_submission(predictions)

