#! /usr/bin/env python3
# self contrained

import pickle
from transformers import BertTokenizer
from tqdm import tqdm
import pkuseg


# LOCAL_DATA = '/home/ariliang/Local-Data/'
LOCAL_DATA = 'E:/Local-Data/'
ROOT = LOCAL_DATA + 'Enconter/'
RAW_DIR = LOCAL_DATA + 'models_datasets/ccks21/'
OUTPUT = ROOT + 'output/'

POSTAG = True


# recurrsive sampling
def process_train_helper(data):

    concated_dialog = []

    for i, dialog in enumerate(data):
        group = [[0]]
        for i in range(1, len(dialog)):
            if dialog[i].get('id') == dialog[i-1].get('id'):
                group[-1].append(i)
            else:
                group.append([i])

        new_dialog = []
        for grp in group:
            utters = []
            for idx in grp:
                new_entity = []
                for ents in list(dialog[idx].values())[2:]:
                    new_entity.extend(ents)
                utters.append(''.join([ '[ENT]'+ent+'[ENT]'  for ent in new_entity ]) + dialog[idx].get('Sentence'))
                # utters.append(dialog[idx].get('Sentence'))
            # new_dialog.append('[SEP]'.join(utters))
            new_dialog.append('[CONT]'.join(utters))

        if dialog[-1].get('id') == 'Patient':
            new_dialog = new_dialog[:-1]

        concated_dialog.append(new_dialog)


    with open(OUTPUT + 'dialo/train_ent.txt', 'w', encoding='utf8') as fw:
        for i, dialog in enumerate(concated_dialog):
            # fw.write(str(i))
            # fw.write('\n')
            for line in dialog:
                fw.write(line)
                fw.write('\n')
            fw.write('\n')

    # recurrsive
    # with open(OUTPUT + 'dialo/train_ent_processed.txt', 'w') as fw:
    #     for dialog in concated_dialog:
    #         for i in range(1, len(dialog), 2):
    #             for line in dialog[:i]:
    #                 fw.writelines(line)
    #                 fw.write('\n')
    #             fw.write('\n')

def process_train():
    # load train data
    with open(RAW_DIR + 'train.pk','rb') as f:
        train = pickle.load(f)

    process_train_helper(train)


def process_dev():
    # load test data
    with open(RAW_DIR + 'dev.pk','rb') as f:
        test_data = pickle.load(f)


    with open(OUTPUT + 'dialo/test_processed_grouped.txt', 'w') as fw:
        for dialog in test_data:
            for utter in dialog.get('history'):
                fw.write(utter+'[ENT][ENT]')
                fw.write('\n')
            fw.write('\n')
        fw.close()


def process_raw():
    # load train data
    with open(RAW_DIR + 'train.pk','rb') as f:
        data = pickle.load(f)

    pos = POSTAG
    if pos:
        seg = pkuseg.pkuseg('medicine', postag=True, user_dict=ROOT+'dataset/all_ents.txt')


    # result = [
    #            [
    #               {'pat': {'utter': str, 'ents': [], 'attr':[]},
    #               'doc': {'utter': str, 'ents': [], 'attr':[]}},
    #              ...
    #            ]
    #              ...
    #          ]
    results = []

    ents_list = ['Diease', 'Test', 'Medicine', 'Symptom']

    for dialog in tqdm(data):
        group = [[0]]
        for i in range(1, len(dialog)):
            if dialog[i].get('id') == dialog[i-1].get('id'):
                group[-1].append(i)
            else:
                group.append([i])

        if dialog[group[0][0]].get('id') == 'Doctor':
            group = group[1:]

        if dialog[group[-1][0]].get('id') == 'Patient':
            group = group[:-1]

        # pd pd pd ...
        assert len(group)%2 == 0

        new_dialog = []
        for i in range(0, len(group), 2):

            pair = {}
            pair['pat'] = {}
            pair['doc'] = {}
            pair['pat']['ents'], pair['pat']['attr'] = [], []
            pair['doc']['ents'], pair['doc']['attr'] = [], []

            pair['pat']['utter'] = '[CONT]'.join([dialog[idx].get('Sentence') for idx in group[i]])
            pair['doc']['utter'] = '[CONT]'.join([dialog[idx].get('Sentence') for idx in group[i+1]])

            for idx in group[i]:
                for k, v in dialog[idx].items():
                    if k in ents_list:
                        pair['pat']['ents'].extend(v)
                    if k == 'Attribute':
                        pair['pat']['attr'].extend(v)
            for idx in group[i+1]:
                for k, v in dialog[idx].items():
                    if k in ents_list:
                        pair['doc']['ents'].extend(v)
                    if k == 'Attribute':
                        pair['doc']['attr'].extend(v)

            if pos:
                # pair['pat']['pos'] = seg.cut(pair['pat']['utter'])
                pair['doc']['pos'] = seg.cut(pair['doc']['utter'])

            new_dialog.append(pair)

        results.append(new_dialog)

    #####################

    with open(OUTPUT+'dialo/results.pk', 'wb') as fw:
        pickle.dump(results, fw)
        fw.close()

def process_tokenization():

    tokenizer = BertTokenizer.from_pretrained('E:/Local-Data/models_datasets/bert-base-chinese')
    tokenizer.add_special_tokens({'additional_special_tokens': ['[NOI]', '[CONT]']})
    tokenizer.save_pretrained(OUTPUT+'dialo/')

    with open(OUTPUT+'dialo/results.pk', 'rb') as frb:
        data = pickle.load(frb)


    results = []
    for dialo in data:
        for pair in dialo:
            pass


def main():
    # process_train()
    # process_dev()
    # process_raw()
    process_tokenization()


if __name__ == '__main__':
    main()