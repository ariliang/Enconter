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

    LOCAL_DATA = '/home/ariliang/Local-Data/'
    ROOT = LOCAL_DATA + 'Enconter/'
    OUTPUT = ROOT + 'output/'
    PWD = ROOT + 'dataset/'

    MODEL = LOCAL_DATA + 'models_datasets/bert-base-chinese/'
    TOKENIZER = LOCAL_DATA + 'models_datasets/bert-base-chinese/'

    RAW_DIR = LOCAL_DATA + 'models_datasets/ccks21/'
    RAW = True
    DATASET = PWD + 'pretrain_bert/'

    MAX_LEN = 256

args = Argument


# functions
#########################

# preprocessing
def preprocessing(tokenizer):
    print('Clean dataset dir')
    os.system(f'rm -rf {args.DATASET}')
    os.system(f'mkdir -p {args.DATASET}')

    # read train set
    train = []
    with open(args.RAW_DIR + 'train.pk', 'rb') as frb:
        pickle.load(frb)
        frb.close()

    # preprocess train
    for item in train:
        pass

    # read dev set
    dev = []
    with open(args.RAW_DIR + 'dev.pk', 'rb') as frb:
        pickle.load(frb)
        frb.close()

# get dataset
def get_dataset(tokenizer):

    if args.raw or not os.path.exists(args.DATASET):
        preprocessing(tokenizer)

# get model
def get_model(path, embed=None):
    model = BertForMaskedLM.from_pretrained(args.MODEL)
    if embed:
        pass
    return model


def main():
    tokenizer = BertTokenizer.from_pretrained(args.TOKENIZER)
    dataset = get_dataset(args.DATASET)


if __name__ == '__main__':
    main()