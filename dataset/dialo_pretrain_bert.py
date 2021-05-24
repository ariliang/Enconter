#! /usr/bin/env python3
# self contained


# pretrain a bert model using MedDG dataset

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, BertTokenizer
from tqdm import tqdm


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
    TRAIN = OUTPUT + 'train_ent'

    # hyper parameters
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MAX_LEN = 512
    BATCH_SIZE = 8
    EPOCH = 10
    WORKERS = 4
    LR = 1e-5

args = Argument


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
    model = BertForMaskedLM.from_pretrained(args.MODEL)
    if embed:
        pass
    return model


def main():
    tokenizer = BertTokenizer.from_pretrained(args.TOKENIZER)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENT]']})

    dataset = get_dataset(tokenizer)
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.BATCH_SIZE,
        num_workers=args.WORKERS
    )

    model = get_model(args.MODEL)
    model = torch.nn.DataParallel(model)
    model = model.to(args.DEVICE)

    optimizer = AdamW(model.parameters(), lr=args.LR)

    min_loss = float('inf')
    total_loss = []
    for epoch in range(args.EPOCH):
        losses = []
        for batch in tqdm(loader, desc=f'Epoch {epoch+1}/{args.EPOCH}...'):
            # batch = (raw, input_ids, attention_mask)
            input_ids, attention_mask = map(lambda x: x.to(args.DEVICE), batch[1:])

            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids).loss.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f'avg loss: {np.mean(losses):.3f}')
        total_loss.extend(losses)

        # save best epoch
        if np.mean(losses) < min_loss:
            min_loss = np.mean(losses)
            torch.save(model.state_dict(), args.OUTPUT + 'trained.pth')

    # save loss change
    plt.plot(np.arange(1, len(total_loss)+1), total_loss, '-o')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(args.OUTPUT + 'loss.png')


if __name__ == '__main__':
    main()