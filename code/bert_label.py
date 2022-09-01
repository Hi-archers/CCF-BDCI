import torch
import csv

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'

from tqdm import tqdm

import torch.nn.functional as F
import torch.nn as softmax
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

from utlis import L_dataset,E_dataset

def safe_concat(A, B):
    if A == None:
        return B
    else :
        return torch.cat((A,B))

def Get_model_token():
    model_path = "../../model/bert-base-uncased/"

    model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path = model_path,
                                              num_labels = 6)

    # Get model's tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = model_path)

    # Get the actual model.
    print('Loading model...')
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path,
                                                               config=model_config).cuda()

    return model, tokenizer

def train(model, tokenize):
    train_path = "../data/bert_train.csv"
    val_path = "../data/bert_val.csv"

    Train_data = L_dataset(data_path = train_path, tokenizer = tokenize, test = False)
    Train_dataloader = DataLoader(dataset = Train_data, batch_size=256, shuffle = True, drop_last=True, collate_fn=Train_data.collate_fn)
    val_data = L_dataset(data_path = val_path, tokenizer = tokenize, test = True)
    val_dataloader = DataLoader(dataset = val_data, batch_size=256, shuffle=False, drop_last=False, collate_fn=val_data.collate_fn)

    optimizer = torch.optim.Adam(params = model.parameters(), lr = 3e-5)

    num_epoch = 5
    for epoch in range(num_epoch):
        model.train()
        losses = []
        ans = None
        for i,batch in tqdm(enumerate(Train_dataloader), total = len(Train_dataloader)):
            tensor_data = batch['tensor_data']
            source_tokens = tensor_data['source_tokens'].cuda()
            source_masks = tensor_data['source_masks'].cuda()
            target_label = tensor_data['target'].cuda()

            output = model(input_ids = source_tokens, attention_mask = source_masks)['logits']

            optimizer.zero_grad()
            loss = F.cross_entropy(output, target_label)

            with torch.no_grad():
                losses.append(loss.item())
            loss.backward()
            optimizer.step()

        print(f"epoch = {epoch} Train losses = {sum(losses) / len(losses)}")
        losses = []
        model.eval()
        for i,batch in enumerate(val_dataloader):
            tensor_data = batch['tensor_data']
            source_tokens = tensor_data['source_tokens'].cuda()
            source_masks = tensor_data['source_masks'].cuda()
            target_label = tensor_data['target'].cuda()

            output = model(input_ids = source_tokens, attention_mask = source_masks)['logits']

            loss = F.cross_entropy(output, target_label)
            with torch.no_grad():
                losses.append(loss.item())

            output = torch.argmax(output, dim = 1)

            tmp = output == target_label
            ans = safe_concat(ans, tmp)

        model.save_pretrained(os.path.join("../data/bert-base", 'best_ckpt'))

        print(f"epoch = {epoch} eval losses = {sum(losses) / len(losses)} eval acc = {torch.sum(ans) / ans.shape[0]}")

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


if __name__ == "__main__":
    model, tokenize = Get_model_token()
    train(model, tokenize)
    Pre(model, tokenize)
