#! /usr/bin/env python3
# self contrained

import pickle

from tqdm import tqdm
import numpy as np
from scipy.special import softmax
from sklearn.model_selection import train_test_split

from transformers import BertTokenizer
import pkuseg


# LOCAL_DATA = '/home/ariliang/Local-Data/'
LOCAL_DATA = 'E:/Local-Data/'
ROOT = LOCAL_DATA + 'Enconter/'
RAW_DIR = LOCAL_DATA + 'models_datasets/ccks21/'
OUTPUT = ROOT + 'output/'

POSTAG = True
MAX_LEN = 512


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

            pair['pat']['utter'] = ''.join([dialog[idx].get('Sentence') for idx in group[i]])
            pair['doc']['utter'] = ''.join([dialog[idx].get('Sentence') for idx in group[i+1]])

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
    tokenizer.add_special_tokens({'additional_special_tokens': ['[NOI]', '\n', '[BOS]', '[EOS]']})
    tokenizer.bos_token = '[BOS]'
    tokenizer.eos_token = '[EOS]'
    # tokenizer.save_pretrained(OUTPUT+'dialo/')

    with open(OUTPUT+'dialo/results.pk', 'rb') as frb:
        data = pickle.load(frb)

    max_len = MAX_LEN
    pat_len = int(max_len*0.8) - int(max_len*0.8)%100
    doc_len = max_len - pat_len

    results = []
    for dialo in tqdm(data):
        for pair_idx, pair in enumerate(dialo):
            masked_span = np.array([], dtype=int)
            score = np.array([], dtype=float)
            pat_tokens = []
            doc_tokens = []

            # add patient utterance
            # patent utter...[SEP]
            pat_tokens.extend(tokenizer.tokenize(pair['pat']['utter']))

            loc = 0
            for bpe, pos in pair['doc']['pos']:

                if len(doc_tokens) > doc_len:
                    break

                for i, tkn in enumerate(bpe):
                    if i == 0:
                        doc_tokens.append(tkn)
                    else:
                        doc_tokens.append('##'+tkn)
                    masked_span = np.append(masked_span, 0)
                    score = np.append(score, 0.0)

                if bpe in pair['doc']['ents']:
                    masked_span[range(loc, loc+len(bpe))] = 1
                    score[range(loc, loc+len(bpe))] += 1.5
                if pos in ['n']:
                    score[range(loc, loc+len(bpe))] += 1.3
                if pos in ['v']:
                    score[range(loc, loc+len(bpe))] += 1.1
                if pos in ['a', 'r']:
                    score[range(loc, loc+len(bpe))] += 0.9
                if pos in ['d']:
                    score[range(loc, loc+len(bpe))] += 0.7

                loc += len(bpe)

            score = softmax(score)

            # add sep token at the end of utterance
            doc_tokens.append(tokenizer.eos_token)
            masked_span = np.append(masked_span, 1)
            score = np.append(score, 0.0)

            tokens = [tokenizer.cls_token] + pat_tokens[:max_len-len(doc_tokens)-2] + [tokenizer.sep_token] + doc_tokens

            if pair_idx > 0:
                tokens[0] = tokenizer.bos_token
                tokens = [tokenizer.cls_token] + \
                            tokenizer.tokenize(dialo[pair_idx-1]['pat']['utter']+tokenizer.sep_token+dialo[pair_idx-1]['doc']['utter'])[:max_len-len(tokens)-2] + \
                            tokens

            results.append({
                'tokens': tokens,
                'score': score,
                'masked_span': masked_span
            })

    with open(OUTPUT+'dialo/results_pos.pk', 'wb') as fwb:
        pickle.dump(results, fwb)
        fwb.close()

# Use softmax as masks
def generate_distance(start, end):
    left_bound, right_bound = start - 1, end + 1
    distance = [min(i - left_bound, right_bound - i) for i in range(start, end+1)]
    return distance

def process_prepare():
    with open(OUTPUT+'dialo/results_pos.pk', 'rb') as frb:
        data = pickle.load(frb)
        frb.close()

    # data = data[:200]
    training_data = []
    for pair in tqdm(data):
        tokens, score, masked_span = pair.values()
        tkns = tokens[-len(masked_span)-1:]

        masked_span = np.insert(masked_span, 0, 1)
        score = np.insert(score, 0, 0.0)

        # while not all(masked_span):
        #     cursor = 0
        #     start, end = None, None

        #     insert_index = []
        #     # 在非重要的span中(masked=0)选择一个较大的score的序号(greedy体现)，加入insert_index
        #     # e.g.: masked_span = [1, 1, 1, 0,   0,   0, 1, 1]
        #     #       score   = [4, 4, 1, 1, 0.5, 0.5, 4, 1]
        #     #                          span{↑          }
        #     # insert_indx.appned(3)
        #     while cursor < len(masked_span):
        #         if masked_span[cursor] == 0:
        #             if start is None:
        #                 start = cursor
        #                 end = cursor
        #             else:
        #                 end = cursor
        #         elif end is not None:
        #             overall_score = score[start:end+1]
        #             # greedy 在这里
        #             insert_index.append(start + overall_score.argmax())
        #             # Clear span
        #             start, end = None, None
        #         cursor += 1
        #     train, label = [], []
        #     select_cursor = 0
        #     # 若1, 1，则label插入'[NOI]'。代表两个重要字相邻，则不在之间插入
        #     # 若1, 0，则label插入0起始的span里，insert_index所指的token(非重要字中score较大)
        #     # 结束后把所有insert_index代表的位置置1，直到所有masked_span里为1，就结束
        #     for i, m, r in zip(range(len(masked_span)), masked_span, tkns):
        #         if m == 1:
        #             train.append(r)
        #             if i + 1 < len(masked_span) and masked_span[i + 1] == 0:
        #                 label.append(tkns[insert_index[select_cursor]])
        #                 select_cursor += 1
        #             else:
        #                 label.append("[NOI]")
        #     training_data.append((train, label))
        #     for i_idx in insert_index:
        #         masked_span[i_idx] = 1
        # # 添加x^k, y^k
        # training_data.append((tkns, ["[NOI]"] * len(tkns)))

        while not all(masked_span):
            cursor = 0
            start, end = None, None

            insert_index = []
            while cursor < len(masked_span):
                if masked_span[cursor] == 0:
                    if start is None:
                        start = cursor
                        end = cursor
                    else:
                        end = cursor
                elif end is not None:
                    overall_score = score[start:end+1]
                    softmax_score = softmax(generate_distance(start, end))
                    if softmax_score.max() - softmax_score.min() != 0:
                        overall_score *= (softmax_score - softmax_score.min()) / (softmax_score.max() - softmax_score.min())
                    insert_index.append(start + overall_score.argmax())
                    # Clear span
                    start, end = None, None
                cursor += 1
            train, label = [], []
            select_cursor = 0
            for i, m, r in zip(range(len(masked_span)), masked_span, tkns):
                if m == 1:
                    train.append(r)
                    if i + 1 < len(masked_span) and masked_span[i + 1] == 0:
                        label.append(tkns[insert_index[select_cursor]])
                        select_cursor += 1
                    else:
                        label.append("[NOI]")
            # training_data.append((train, label))
            # training_data.append((tokens[:-len(tkns)] + train, ['[NOI]']*(len(tokens)-len(tkns)) + label, [0]*(len(tokens)-len(tkns))+[1]*len(train)))
            training_data.append((tokens[:-len(tkns)] + train, tokens[:-len(tkns)] + label, [0]*len(tokens[:-len(tkns)]) + [1]*len(train)))
            for i_idx in insert_index:
                masked_span[i_idx] = 1
        # training_data.append((tkns, ["[NOI]"] * len(tkns)))
        training_data.append((tokens, tokens[:-len(tkns)] + ["[NOI]"] * len(tkns), [0]*len(tokens[:-len(tkns)]) + [1]*len(tkns)))


    dialo_train, dialo_eval = train_test_split(training_data, test_size=0.2)

    with open(OUTPUT+'dialo/dialo_train', 'wb') as fwb:
        pickle.dump(dialo_train, fwb)
        fwb.close()

    with open(OUTPUT+'dialo/dialo_eval', 'wb') as fwb:
        pickle.dump(dialo_eval, fwb)
        fwb.close()

def main():
    # process_train()
    # process_dev()

    # process_raw()
    process_tokenization()
    # process_prepare()


if __name__ == '__main__':
    main()