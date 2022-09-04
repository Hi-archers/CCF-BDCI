import torch
import csv

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from tqdm import tqdm

from sklearn.metrics import precision_score, recall_score, accuracy_score

import torch.nn.functional as F
import torch.nn as softmax
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

from utlis import L_dataset

def safe_concat(A, B):
    if A == None:
        return B
    else :
        return torch.cat((A,B))

from transformers import BertTokenizer
from roformer import RoFormerForSequenceClassification,RoFormerConfig

def Get_model_token():
    model_path = "../../model/roformer_v2_chinese_char_large"

    model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path = model_path,
                                              num_labels = 2)

    # Get model's tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = model_path)

    # Get the actual model.
    print('Loading model...')
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path,
                                                               config=model_config).cuda()

    return model, tokenizer

def Get_roformer():
    model_path = "../../model/roformer_v2_chinese_char_large"

    config = RoFormerConfig.from_pretrained(
        model_path,
        num_labels=2,
    )

    # Get model's tokenizer.
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path = model_path)

    # Get the actual model.
    print('Loading model...')
    model = RoFormerForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_path,
        config=config,
    ).cuda()

    return model, tokenizer


def train(model, tokenize):
    train_path = "../dataset/bert_train.csv"
    val_path = "../dataset/bert_val.csv"

    batch_size = 8
    lr = 3e-5

    Train_data = L_dataset(data_path = train_path, tokenizer = tokenize, test = False)
    Train_dataloader = DataLoader(dataset = Train_data, batch_size=batch_size, shuffle = True, drop_last=True, collate_fn=Train_data.collate_fn)
    val_data = L_dataset(data_path = val_path, tokenizer = tokenize, test = True)
    val_dataloader = DataLoader(dataset = val_data, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=val_data.collate_fn)

    optimizer = torch.optim.AdamW(params = model.parameters(), lr = lr)

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        # 3% of Total Steps
        num_warmup_steps=int(0.1 * 40 * len(Train_data) / 24),
        num_training_steps=int(40 * len(Train_data) / 24),
    )

    logger = get_log(__name__, "logs/bert_log.txt")

    logger.info("Start Training！")
    logger.info(f"Model = roformer_v2_chinese_char_large number label = {2} ")
    logger.info(f"Optimizer Setting : learning rate = {lr} batch size = {batch_size}")


    num_epoch = 40
    for epoch in range(num_epoch):
        model.train()
        losses = []
        ans = None
        pre = None
        for i,batch in tqdm(enumerate(Train_dataloader), total = len(Train_dataloader)):
            tensor_data = batch['tensor_data']
            source_tokens = tensor_data['source_tokens'].cuda()
            source_masks = tensor_data['source_masks'].cuda()
            target_label = tensor_data['target'].cuda()

            output = model(input_ids = source_tokens, attention_mask = source_masks).logits
            # output = model(input_ids = source_tokens, attention_mask = source_masks).logits[0]

            # print(output)

            # print(f"source_token shape = {source_tokens.shape}")
            # print(f"output shape = {output.shape}")
            # exit()

            # print(f"output shape = {output.shape}")
            # print(f"target label shape = {target_label.shape}")
            # print(target_label)

            optimizer.zero_grad()
            loss = F.cross_entropy(output, target_label)

            with torch.no_grad():
                losses.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()

            output = torch.argmax(output, dim = 1)

            pre = safe_concat(pre, output)
            ans = safe_concat(ans, target_label)

        pre = ~pre.cpu().numpy()
        ans = ~ans.cpu().numpy()
        pre = ~pre
        ans = ~ans


        logger.info(f"epoch = {epoch} Train losses = {sum(losses) / len(losses)} Train acc = {accuracy_score(ans, pre)} precision = {precision_score(ans, pre)} recall = {recall_score(ans, pre)}")

        ans = None
        pre = None

        losses = []
        model.eval()
        for i,batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            tensor_data = batch['tensor_data']
            source_tokens = tensor_data['source_tokens'].cuda()
            source_masks = tensor_data['source_masks'].cuda()
            target_label = tensor_data['target'].cuda()

            with torch.no_grad():
                output = model(input_ids = source_tokens, attention_mask = source_masks)['logits']

                loss = F.cross_entropy(output, target_label)
                losses.append(loss.item())

                output = torch.argmax(output, dim = 1)

                pre = safe_concat(pre, output)
                ans = safe_concat(ans, target_label)


        pre = ~pre.cpu().numpy()
        ans = ~ans.cpu().numpy()

        pre = ~pre
        ans = ~ans

        # model.save_pretrained(os.path.join("./data/bert-base", 'best_ckpt'))

        logger.info(f"epoch = {epoch} eval losses = {sum(losses) / len(losses)} Eval acc = {accuracy_score(ans, pre)} precision = {precision_score(ans, pre)} recall = {recall_score(ans, pre)}")

def Pre(model, tokenize):
    train_path = "../data/train.txt"
    val_path = "../data/val.txt"

    Train_data = E_dataset(data_path = train_path, tokenizer = tokenize, test = True)

    print(f"Train len = {len(Train_data)}")

    Train_dataloader = DataLoader(dataset = Train_data, batch_size=256, shuffle = False, drop_last=False, collate_fn=Train_data.collate_fn)
    val_data = E_dataset(data_path = val_path, tokenizer = tokenize, test = True)

    print(f"Val len = {len(val_data)}")

    val_dataloader = DataLoader(dataset = val_data, batch_size=256, shuffle=False, drop_last=False, collate_fn=val_data.collate_fn)

    Train_label = None

    model.eval()

    for i, batch in tqdm(enumerate(Train_dataloader), total=len(Train_dataloader)):
        tensor_data = batch['tensor_data']
        source_tokens = tensor_data['source_tokens'].cuda()
        source_masks = tensor_data['source_masks'].cuda()

        output = model(input_ids=source_tokens, attention_mask=source_masks)['logits']
        output = torch.argmax(output, dim=1)
        Train_label = safe_concat(Train_label, output)

    val_label = None
    for i, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        tensor_data = batch['tensor_data']
        source_tokens = tensor_data['source_tokens'].cuda()
        source_masks = tensor_data['source_masks'].cuda()

        output = model(input_ids=source_tokens, attention_mask=source_masks)['logits']
        output = torch.argmax(output, dim=1)
        val_label = safe_concat(val_label, output)

    with open("../data/train_label.txt", 'w+', encoding='utf-8') as f:
        for i in Train_label:
            f.write(str(i.item()))
            f.write('\n')

    with open("../data/val_label.txt", 'w+', encoding='utf-8') as f:
        for i in val_label:
            f.write(str(i.item()))
            f.write('\n')

import logging
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

if __name__ == "__main__":
    model, tokenize = Get_roformer()
    train(model, tokenize)
    # Pre(model, tokenize)
