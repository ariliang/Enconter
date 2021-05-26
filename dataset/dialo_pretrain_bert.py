#! /usr/bin/env python3
# self contained


# pretrain a bert model using MedDG dataset

import os
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from transformers import BertForMaskedLM, BertTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader


# global
##############

class Argument:

    # environment
    LOCAL_DATA = '/home/ariliang/Local-Data/'
    ROOT = LOCAL_DATA + 'Enconter/'
    OUTPUT = ROOT + 'output/dialo/'
    PWD = ROOT + 'dataset/'

    # pretrained model and tokenizer
    MODEL = LOCAL_DATA + 'models_datasets/bert-base-chinese/'
    MODEL_TRAINED = OUTPUT + 'model/'
    TOKENIZER = LOCAL_DATA + 'models_datasets/bert-base-chinese/'

    # raw train file and preprocessed
    RAW_TRAIN = OUTPUT + 'train_ent.txt'
    RAW = False
    TRAIN = OUTPUT + 'train_ent_test'

    # hyper parameters
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MAX_LEN = 512
    BATCH_SIZE = 8
    BATCH_SIZE_EVAL = 16
    EPOCH = 5
    WORKERS = 4
    LR = 1e-5
    SEED = 2021

args = Argument
min_loss = float('inf')


class MedDataset:

    def __init__(self, inputs):
        self.raw = inputs['raw']
        self.input_ids = inputs['input_ids']
        self.attention_mask = inputs['attention_mask']

    def __getitem__(self, i):
        return (self.raw[i], self.input_ids[i], self.attention_mask[i])

    def __len__(self):
        return len(self.input_ids)


# functions
#########################

# preprocessing
def preprocessing(tokenizer):
    print(f'Clean dataset file: {args.TRAIN}')
    os.system(f'rm -f {args.TRAIN}')

    # read train set
    train = []
    with open(args.RAW_TRAIN, 'r') as fr:
        data = fr.read().strip()
        fr.close()

    # document to list of list of sentences
    for dialo in data.split('\n\n'):
        train.append(dialo.split('\n'))

    inputs = {
        'raw': [],
        'input_ids': [],
        'attention_mask': []
    }

    # preprocess train
    for dialo in train:
        for i in range(1, len(dialo), 2):
            inputs['raw'].append(dialo[i].split('[ENT]')[0])

    print('Tokenizing, this will take a long time...')
    encoded = tokenizer(inputs['raw'], padding='max_length', truncation=True, max_length=args.MAX_LEN)

    inputs['input_ids'] = encoded['input_ids']
    inputs['attention_mask'] = encoded['attention_mask']

    print(f'Writing processed train data to {args.TRAIN}')
    with open(args.TRAIN, 'wb') as fwb:
        pickle.dump(inputs, fwb)

# get dataset
def get_dataset(tokenizer):

    # preprocessing
    if args.RAW or not os.path.exists(args.TRAIN):
        preprocessing(tokenizer)
    assert os.path.exists(args.TRAIN)

    # inputs = {'raw': [...], 'input_ids': [...], 'attention_mask': [...]}
    inputs = {}
    with open(args.TRAIN, 'rb') as frb:
        inputs = pickle.load(frb)
        frb.close()

    inputs['input_ids'] = torch.tensor(inputs['input_ids'])
    inputs['attention_mask'] = torch.tensor(inputs['attention_mask'], dtype=torch.float)

    med_dataset = MedDataset(inputs)
    return med_dataset

# get model
def get_model(path, embed=None):
    print('Loading pretrained bert model...')
    model = BertForMaskedLM.from_pretrained(args.MODEL)
    if embed:
        pass
    return model

def random_mask(dataloader, tokenizer):
    pass

def compute_loss(logits, labels, loss_func, tokenizer):
    num_targets = labels.ne(tokenizer.pad_token_id).long().sum().item()
    loss = loss_func(
        logits.view(-1, logits.shape[-1]),
        labels.view(-1)
    )
    return loss/num_targets

def train():
    tokenizer = BertTokenizer.from_pretrained(args.TOKENIZER)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENT]']})

    dataset = get_dataset(tokenizer)
    train_set, eval_set = train_test_split(dataset, test_size=0.2, shuffle=True, random_state=args.SEED)
    print(f'Train set size: {len(train_set)}, eval set size: {len(eval_set)}')
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.BATCH_SIZE,
        num_workers=args.WORKERS
    )
    eval_loader = DataLoader(
        dataset=eval_set,
        batch_size=args.BATCH_SIZE,
        num_workers=args.WORKERS
    )

    model = get_model(args.MODEL)
    model = torch.nn.DataParallel(model)
    model = model.to(args.DEVICE)
    optimizer = AdamW(model.parameters(), lr=args.LR)

    loss_func = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=tokenizer.pad_token_id)

    model.train()
    total_loss = []
    for epoch in range(args.EPOCH):
        losses = []
        for batch in tqdm(train_loader, desc=f'Train epoch {epoch+1}/{args.EPOCH}'):
            # batch = (raw, input_ids, attention_mask)
            input_ids, attention_mask = map(lambda x: x.to(args.DEVICE), batch[1:])

            logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids).logits

            loss = compute_loss(logits, input_ids, loss_func, tokenizer)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f'Train loss: {np.mean(losses):.3f}')
        total_loss.extend(losses)

        evaluate(model, eval_loader, loss_func, tokenizer)

    # save loss change
    plt.plot(np.arange(1, len(total_loss)+1), total_loss, '-o')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(args.OUTPUT + 'loss.png')


def evaluate(model, loader, loss_func, tokenizer):

    global min_loss
    model.eval()

    losses = []
    for batch in tqdm(loader, desc=f'Evaluating'):
        input_ids, attention_mask = map(lambda x: x.to(args.DEVICE), batch[1:])

        logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids).logits
        loss = compute_loss(logits, input_ids, loss_func, tokenizer)

        losses.append(loss.item())

    print(f'Eval loss: {np.mean(losses):.3f}')

    # save best epoch
    if np.mean(losses) < min_loss:
        min_loss = np.mean(losses)
        torch.save(model.module.state_dict(), args.OUTPUT + 'trained.pth')


def main():
    train()


if __name__ == '__main__':
    main()
