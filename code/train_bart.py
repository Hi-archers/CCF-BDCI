import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from torch import nn
from transformers import BertTokenizer, BartForConditionalGeneration

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

def train(args, logger):
    model_name = args.model_name
    train_batch_size = args.train_batch_size
    #val_batch_size = args.val_batch_size
    train_path = args.train_path
    #val_path = args.val_path
    #save_file = args.save_file
    epoch_number = args.epoch_number
    learning_rate = args.learning_rate
    multi_gpu = args.multi_gpu

    acc_grad = 1

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name).cuda()

    model.train()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    train_dataset = Seq2Seq_dataset(train_path, tokenizer)

    if False and multi_gpu:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size = train_batch_size, shuffle = False, num_workers=8,
                                      sampler = train_sampler,collate_fn=train_dataset.collate_fn)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size = train_batch_size, shuffle = True, num_workers=8,
                                      collate_fn=train_dataset.collate_fn,drop_last=True)


    # val_dataset = Seq2Seq_dataset(val_path, tokenizer)
    # val_dataloader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, num_workers=8, collate_fn=val_dataset.collate_fn)

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    LOSS = []

    for epoch in range(epoch_number):
        loss_epoch = 0
        step = 0
        print("Epoch",epoch)
        #score = model_eval(model, val_dataloader, tokenizer, model_type)

        loss_train = []

        tmp = ''
        model.train()
        for i,batch in enumerate(tqdm(train_dataloader)):

            # meta_data = batch['meta_data']


            tensor_data = batch['tensor_data']
            source_tokens = tensor_data['source_tokens'].cuda()
            source_masks = tensor_data['source_masks'].cuda()
            target_tokens = tensor_data['target_tokens'].cuda()
            target_masks = tensor_data['target_masks'].cuda()

            #print(source_tokens.shape)
            #print(target_tokens.shape)

            #print(source_tokens.shape[1])
            #print(target_tokens.shape[1])

            #exit()

            target_input = target_tokens[:, :-1].contiguous()
            target_output = target_tokens[:, 1:].contiguous()


            if source_tokens.shape[1] >= 512 or target_tokens.shape[1] >= 512:
                continue

            # print(f"source token shape = {source_tokens.shape}")
            # print(f"target token shape = {target_tokens.shape}")

            optimizer.zero_grad()

            # Run forward pass and calculate loss labels = target_output
            outputs = model(source_tokens, attention_mask = source_masks, decoder_input_ids = target_input)[0]

            loss = loss_fn(outputs.view(-1, outputs.size(-1)), target_output.view(-1))

            if multi_gpu:
                loss = loss.mean()

            loss.backward()
            optimizer.step()

            loss_train.append(loss.item())

            if len(tmp) == 0:
                tmp = outputs.cpu().detach().numpy()
                tmp = tmp[0]
                tmp = np.argmax(tmp,axis = 1)
                print(f"\n生成结果\n{tokenizer.decode(tmp, skip_special_tokens=True)}\n实际结果\n{tokenizer.decode(target_tokens[0], skip_special_tokens=True)}")
                logger.info(tokenizer.decode(tmp, skip_special_tokens=True))

        logger.info(f"epoch = {epoch} train loss = {sum(loss_train)/len(loss_train)}")


        # model.save_pretrained(os.path.join(save_file,str(epoch) + '_ckpt'))
        # logger.info("Success save the model ckpt in " + str(epoch))

        loss_val = []

        """
        model.eval()
        for i,batch in enumerate(tqdm(val_dataloader)):

            meta_data = batch['meta_data']
            tensor_data = batch['tensor_data']
            source_tokens = tensor_data['source_tokens'].cuda()
            source_masks = tensor_data['source_masks'].cuda()
            target_tokens = tensor_data['target_tokens'].cuda()
            target_masks = tensor_data['target_masks'].cuda()

            target_input = target_tokens[:, :-1].contiguous()
            target_output = target_tokens[:, 1:].contiguous()

            if target_tokens.shape[1] >= 503:
                continue

            optimizer.zero_grad()

            # Run forward pass and calculate loss labels = target_output
            outputs = model(source_tokens, attention_mask = source_masks, decoder_input_ids = target_input)[0]

            loss = loss_fn(outputs.view(-1, outputs.size(-1)), target_output.view(-1))

            if multi_gpu:
                loss = loss.mean()

            loss_val.append(loss.item())

            if len(tmp) == 0:
                tmp = outputs.cpu().detach().numpy()
                tmp = tmp[0]
                tmp = np.argmax(tmp,axis = 1)
                print(tokenizer.decode(tmp, skip_special_tokens=True))
                
        logger.info(f"epoch = {epoch} valid loss = {sum(loss_val)/len(loss_val)} valid ppl = {math.exp(sum(loss_val)/len(loss_val))}")
        """



    return LOSS


def eval():
    token_name = '../model/bart-large-chinese'
    model_name = './model/NEW_MBART/6_ckpt'
    #token_name = '../model/mt5-small'

    tokenizer = BertTokenizer.from_pretrained(token_name)
    model = BartForConditionalGeneration.from_pretrained(model_name).cuda()

    model.eval()

    Str = ' 中文名@赛貂蝉##长相@粗犷##住处@长安###'

    source = tokenizer.encode(Str, return_tensors='pt').cuda()

    print(source)
    print(tokenizer.batch_decode(source))

    output = model.generate(
        source,
        max_length = 500,
        num_beams = 10,
        do_sample = True
    )

    print(output.shape,output)

    output = ''.join(tokenizer.decode(output[0], skip_special_tokens = True)).replace(' ','')

    print(Str)
    print(output)

def parse_args():
    parser = argparse.ArgumentParser('Training args...')
    parser.add_argument('--model_name', default='../../model/bart-base-chinese', help='Model name.')
    parser.add_argument('--train_batch_size', default = 32, help='Traning set batch size.')
    #parser.add_argument('--val_batch_size', default=1, help='Validation set batch size.')
    parser.add_argument('--train_path', default='../dataset/train.json', help='Traning set file.')
    #parser.add_argument('--val_path', default=  '../dataset/.txt', help='Validation set file.')
    #parser.add_argument('--save_file', default= './model/NEW_MBART', help='Save model path.')
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

    train_ls = train(args, logger)

    train_min = np.argmin(train_ls)

    show_train_min = '[' + str(train_min) + ' ' + str(train_ls[train_min]) + ']'

    plt.plot(train_min, train_ls[train_min], 'ko')
    plt.annotate(show_train_min, xy=(train_min, train_ls[train_min]), xytext=(train_min, train_ls[train_min]))

    plt.plot(train_ls, label='training mse loss')
    plt.show()
