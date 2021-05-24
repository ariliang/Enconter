#! /usr/bin/env python3
# self contained


# pretrain a bert model using MedDG dataset

import os
import pickle

import torch
from transformers import BertForMaskedLM, BertTokenizer


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
    TOKENIZER = LOCAL_DATA + 'models_datasets/bert-base-chinese/'

    # raw train file and preprocessed
    RAW_TRAIN = OUTPUT + 'train_ent.txt'
    RAW = True
    TRAIN = OUTPUT + 'train_ent'

    # hyper arguments
    MAX_LEN = 256

args = Argument


# functions
#########################

# preprocessing
def preprocessing(tokenizer):
    print('Clean dataset dir')

    # read train set
    train = []
    with open(args.RAW_TRAIN, 'r') as fr:
        pass

    # preprocess train
    for item in train:
        pass

    # read dev set
    dev = []
    with open(args.RAW_TRAIN, 'r') as fr:
        pass

# get dataset
def get_dataset(tokenizer):

    if args.raw or not os.path.exists(args.TRAIN):
        preprocessing(tokenizer)

# get model
def get_model(path, embed=None):
    model = BertForMaskedLM.from_pretrained(args.MODEL)
    if embed:
        pass
    return model


def main():
    tokenizer = BertTokenizer.from_pretrained(args.TOKENIZER)
    dataset = get_dataset(args.TRAIN)


if __name__ == '__main__':
    main()