#! /usr/bin/env python3
# self contained


# pretrain a bert model using MedDG dataset

import torch

from transformers import BertForMaskedLM, BertTokenizer

def main():
    tokenizer = BertTokenizer.from_pretrained('/home/ariliang/Local-Data/models_datasets/bert-base-chinese/')
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENT]']})

    model = BertForMaskedLM.from_pretrained('/home/ariliang/Local-Data/models_datasets/bert-base-chinese/')
    model.load_state_dict(torch.load('/home/ariliang/Local-Data/Enconter/output/dialo/trained.pth', map_location='cpu'))

    sent = '您好，[MASK]这种状况'
    encoded = tokenizer.encode(sent)

    input_ids = torch.tensor(encoded).unsqueeze(0)

    logits = model(input_ids).logits
    output = torch.argmax(logits, dim=2).flatten().tolist()

    print(tokenizer.decode(output))

if __name__ == '__main__':
    main()
