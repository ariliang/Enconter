from tqdm import tqdm
import pickle as pk
from collections import Counter
from scipy.special import softmax
import numpy as np


DATASET = 'dataset/'
OUTPUT = 'output/CoNLL/'

# Read CoNLL
##############

result = []
# Read train as well as dev
with open(DATASET + "eng.train") as ftrain, open(DATASET + "eng.testa") as fdev:
    document = {}
    lines = []
    for line in ftrain.readlines():
        line = line.split()
        if len(line) > 0:
            if line[1] == '-X-':
                document['conll'] = lines
                document['content'] = ' '.join([line[0] for line in lines])
                result.append(document)
                lines = []
                document = {}
            else:
                lines.append(line)
    for line in fdev.readlines():
        line = line.split()
        if len(line) > 0:
            if line[1] == '-X-':
                document['conll'] = lines
                document['content'] = ' '.join([line[0] for line in lines])
                result.append(document)
                lines = []
                document = {}
            else:
                lines.append(line)


# YAKE!
###########

import yake

language = "en"
max_key_word_ngram_size = 3
deduplication_thresold = 0.9
deduplication_algo = 'seqm'
windowSize = 1
numOfKeywords = 20
custom_kw_extractor = yake.KeywordExtractor(lan=language,
                                            n=max_key_word_ngram_size,
                                            dedupLim=deduplication_thresold,
                                            dedupFunc=deduplication_algo,
                                            windowsSize=windowSize,
                                            top=numOfKeywords,
                                            features=None)

for r in tqdm(result):
    keywords = custom_kw_extractor.extract_keywords(r['content'])
    keys = [key for key, value in keywords]
    scores = np.array([value for key, value in keywords])
    # change score from low -> high to high -> low
    scores = 1 - (scores - scores.min()) / (scores.max() - scores.min())
    # Convert list to dict
    keywords = { key : value for key, value in zip(keys, scores)}
    r['keywords'] = keywords

with open(OUTPUT + "CoNLL-2003", "wb") as fout:
    pk.dump(result, fout)

# Load keyword extracted CoNLL

with open(OUTPUT + "CoNLL-2003", "rb") as fin:
    result = pk.load(fin)

'''results = [
    {
        'conll': [['EU', 'NNP', 'I-NP', 'I-ORG'], ...],
        'content': 'EU rejects German...',
        'keywords': {
            'mad cow disease': 1.0,
            ...
        }
    }
]
'''


# tf-idf
#############

# TF-ID
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import numpy as np

vectorizer = TfidfVectorizer(stop_words='english')

# whole document
corpus = []
for r in result:
    corpus.append(r['content'])

X = vectorizer.fit_transform(corpus)
tfidf = X.todense()
# scale score to 0 ~ 1
tfidf = tfidf / tfidf.max(axis=0)
word_array = vectorizer.get_feature_names()
word_dict = vectorizer.vocabulary_
reverse_dict = {v : k for k, v in word_dict.items()}

# Append tf-idf score
for r in tqdm(result):
    for token in r["conll"]:
        if word_dict.get(token[0].lower()) is not None:
            token.append(tfidf[0, word_dict[token[0].lower()]])
        else:
            token.append(0)
    for token in r["conll"]:
        assert len(token) == 5

# append td-idf score in the end of [{'conll': [['EU', 'NNP', 'I-NP', 'I-ORG', td-idf score], ...]}]


# YAKE!
#############

max_key_word_ngram_size = 3
# Append yake score
for r in tqdm(result):
    length, token_counter = len(r["conll"]), 0
    keywords = r["keywords"]
    while token_counter < length:
        old_c = token_counter
        # Try to match keyword within keyword ngram
        for key_word_len in range(max_key_word_ngram_size, 0, -1):
            # Find skills
            keyword = " ".join([s[0] for s in r["conll"][token_counter:token_counter+key_word_len]]).lower()
            if (token_counter + key_word_len - 1) < length and keyword in keywords:
                # Tag skills
                for i in range(key_word_len):
                    r['conll'][token_counter + i].append(keywords[keyword])
                # Move pointer
                token_counter += key_word_len
                break
        # No skill is found
        if old_c == token_counter:
            r['conll'][token_counter].append(0)
            token_counter += 1
    for token in r["conll"]:
        assert len(token) == 6

# pattern match, match keywords
# append yake score after iftdf: {'conll': [['Rare', 'NNP', 'I-NP', 'O', 0.0, 0], ...]}


# Load tokenizer
from transformers import BertTokenizer, BertConfig
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
tokenizer.add_special_tokens({"additional_special_tokens" : ["[NOI]", "\n"]})
BertConfig.from_pretrained("bert-base-cased")


# Split by BPE
###############

import copy
for r in tqdm(result):
    r["tokenize"] = []
    r["tokenize"].append(('[CLS]', "", "", "start_tkn", 0, 0))
    for t in r["conll"]:
        for i, bpe in enumerate(tokenizer.tokenize(t[0])):
            temp = copy.deepcopy(t)
            # Deal with skill tag
            temp[0] = bpe
            r['tokenize'].append(temp)
    r["tokenize"].append(('[SEP]', "", "", "end_tkn", 0, 0))

# tokenized by bpe, this extends {'conll': [...]}


# Dynamic programming solution for house robber
#################################################

class Solution:
    def explore(self, index, nums, seen):
        if index >= len(nums):
            return (0, [])

        if index in seen:
            return seen[index]

        # include current house
        val1, path1 = self.explore(index + 2, nums, seen)
        val1 += nums[index]

        # exclude current house
        val2, path2 = self.explore(index + 1, nums, seen)

        if val1 > val2:
            seen[index] = (val1, [index] + path1)
        else:
            seen[index] = (val2, path2)
        return seen[index]

    def rob(self, nums) -> int:
        seen = {}
        val, path = self.explore(0, nums, seen)
        return val, path

s = Solution()


# POINTER-E
#############

def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

# rqm only
training_data = []
noi_ratio = np.zeros(shape=len(result))
for index, r in tqdm(enumerate(result), total=len(result)):
    score_arr = np.zeros(len(r['tokenize']))
    masked_span = np.zeros(len(r['tokenize']))
    if len(r['tokenize']) > 512:
        continue
    for bpe_index, bpe in enumerate(r['tokenize']):
        score, mask_value = 0, 0
        # skill and special token
        if bpe[3] != 'O':
            score = 4
            mask_value = 2
        else:
            # words other than skills and start token
            added = False
            for pos_tag in ["NN", "JJ", "VB"]:
                if pos_tag in bpe[1]:
                    score += 1
                    added = True
                    break
            if not added:
                score += 0.5
            # tf-idf score
            score += bpe[4]
            # yake score
            score += bpe[5]
            mask_value = 0
        # Turn score into negative
        score_arr[bpe_index] = 4-score
        masked_span[bpe_index] = mask_value
    rqm_tkn = [t[0] for t in r['tokenize']]
    training_data.append((rqm_tkn, ["[NOI]"] * len(rqm_tkn)))
    while True:
        zero_ranges = zero_runs(masked_span)
        if len(zero_ranges) == 0:
            break
        for span in zero_ranges:
            start, end = span[0], span[1]
            # One house to rob
            if span[1] - span[0] == 1:
                masked_span[start] = 1
            # Two house to rob
            elif span[1] - span[0] == 2:
                if score_arr[start] > score_arr[end-1]:
                    masked_span[start] = 1
                else:
                    masked_span[end-1] = 1
            # More than two house to rob
            else:
                value, path = s.rob(score_arr[start:end])
                for p in path:
                    masked_span[start+p] = 1

        train, label = [], []
        select_cursor = 0
        for i in range(len(masked_span)):
            if masked_span[i] != 1:
                train.append(rqm_tkn[i])
                if i + 1 < len(masked_span) and masked_span[i + 1] == 1:
                    label.append(rqm_tkn[i+1])
                    select_cursor += 1
                else:
                    label.append("[NOI]")
        training_data.append((train, label))
        rqm_tkn = train
        score_arr = score_arr[masked_span != 1]
        masked_span = masked_span[masked_span != 1]
        assert len(masked_span) == len(rqm_tkn) == len(score_arr)
    noi_ratio[index] = Counter(training_data[-1][1])['[NOI]'] / len(training_data[-1][1])

with open(OUTPUT + "CoNLL_pointer_e", "wb") as fout:
    pk.dump(training_data, fout)


# Greedy Enconter
##################

# rqm only
# Use softmax as a mask (multiply)
training_data = []
for r in tqdm(result):
    score_arr = []
    masked_span = []
    if len(r['tokenize']) > 512:
        continue
    for bpe in r['tokenize']:
        score, mask_value = 0, 0
        # POS tag score
        # Skill and special token
        if bpe[3] != 'O':
            score = 4
            mask_value = 1
        else:
            # words other than skills and start token
            added = False
            for pos_tag in ["NN", "JJ", "VB"]:
                if pos_tag in bpe[1]:
                    score += 1
                    added = True
                    break
            if not added:
                score += 0.5
            # tf-idf score
            score += bpe[4]
            # yake score
            score += bpe[5]
            mask_value = 0
        # Turn score into negative
        score_arr.append(score)
        masked_span.append(mask_value)
    score_arr = np.array(score_arr)
    tkns = [t[0] for t in r['tokenize']]
    while not all(masked_span):
        cursor = 0
        start, end = None, None
        # 这两个没用
        max_reward, max_reward_idx = float('-inf'), None
        insert_index = []
        # 在非重要的span中(masked=0)选择一个较大的score的序号(greedy体现)，加入insert_index
        # e.g.: masked_span = [1, 1, 1, 0,   0,   0, 1, 1]
        #       score_arr   = [4, 4, 1, 1, 0.5, 0.5, 4, 1]
        #                          span{↑          }
        # insert_indx.appned(3)
        while cursor < len(masked_span):
            if masked_span[cursor] == 0:
                if start is None:
                    start = cursor
                    end = cursor
                else:
                    end = cursor
            elif end is not None:
                overall_score = score_arr[start:end+1]
                # greedy 在这里
                insert_index.append(start + overall_score.argmax())
                # Clear span
                start, end = None, None
            cursor += 1
        train, label = [], []
        select_cursor = 0
        # 若1, 1，则label插入'[NOI]'。代表两个重要字相邻，则不在之间插入
        # 若1, 0，则label插入0起始的span里，insert_index所指的token(非重要字中score较大)
        # 结束后把所有insert_index代表的位置置1，直到所有masked_span里为1，就结束
        for i, m, r in zip(range(len(masked_span)), masked_span, tkns):
            if m == 1:
                train.append(r)
                if i + 1 < len(masked_span) and masked_span[i + 1] == 0:
                    label.append(tkns[insert_index[select_cursor]])
                    select_cursor += 1
                else:
                    label.append("[NOI]")
        training_data.append((train, label))
        for i_idx in insert_index:
            masked_span[i_idx] = 1
    # 添加x^k, y^k
    training_data.append((tkns, ["[NOI]"] * len(tkns)))

with open(OUTPUT + "CoNLL_greedy_enconter", "wb") as fout:
    pk.dump(training_data, fout)


# BBT Enconter
##############

# Use softmax as masks
def generate_distance(start, end):
    left_bound, right_bound = start - 1, end + 1
    distance = [min(i - left_bound, right_bound - i) for i in range(start, end+1)]
    return distance

# rqm only
# Use softmax as a mask (multiply)
training_data = []
for r in tqdm(result):
    score_arr = []
    masked_span = []
    if len(r['tokenize']) > 512:
        continue
    for bpe in r['tokenize']:
        score, mask_value = 0, 0
        # POS tag score
        # Skill and special token
        if bpe[3] != 'O':
            score = 4
            mask_value = 1
        else:
            # words other than skills and start token
            added = False
            for pos_tag in ["NN", "JJ", "VB"]:
                if pos_tag in bpe[1]:
                    score += 1
                    added = True
                    break
            if not added:
                score += 0.5
            # tf-idf score
            score += bpe[4]
            # yake score
            score += bpe[5]
            mask_value = 0
        # Turn score into negative
        score_arr.append(score)
        masked_span.append(mask_value)
    score_arr = np.array(score_arr)
    tkns = [t[0] for t in r['tokenize']]
    while not all(masked_span):
        cursor = 0
        start, end = None, None
        max_reward, max_reward_idx = float('-inf'), None
        insert_index = []
        while cursor < len(masked_span):
            if masked_span[cursor] == 0:
                if start is None:
                    start = cursor
                    end = cursor
                else:
                    end = cursor
            elif end is not None:
                overall_score = score_arr[start:end+1]
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
        training_data.append((train, label))
        for i_idx in insert_index:
            masked_span[i_idx] = 1
    training_data.append((tkns, ["[NOI]"] * len(tkns)))

with open(OUTPUT + "CoNLL_bbt_enconter", "wb") as fout:
    pk.dump(training_data, fout)


# Testing data
#################

#Prepare testing data for requirement insertion only

with open(DATASET + "eng.testb") as fin:
    testing, lines = [], []
    for line in fin.readlines():
        line = line.split()
        if len(line) > 0:
            if line[1] == '-X-':
                testing.append(lines)
                lines = []
            else:
                lines.append(line)

testing_data = []
for test in tqdm(testing):
    gt = ' '.join([line[0] for line in test])
    content = ' '.join([line[0] for line in test if line[-1] != 'O'])
    testing_data.append((" [CLS] " + content + " [SEP] ", gt))

with open(OUTPUT + "CoNLL_test", "wb") as fout:
    pk.dump(testing_data, fout)

#Prepare testing data for requirement only but with span inference the phrase

with open(DATASET + "eng.testb") as fin:
    testing, lines = [], []
    for line in fin.readlines():
        line = line.split()
        if len(line) > 0:
            if line[1] == '-X-':
                testing.append(lines)
                lines = []
            else:
                lines.append(line)

entities = []
for test in tqdm(testing):
    entity = []
    tmp_arr = []
    last_tag = ''
    for line in test:
        if len(tmp_arr) != 0 and last_tag != line[3]:
            entity.append(tmp_arr)
            tmp_arr = []
        if line[3] != 'O':
            tmp_arr.append(line[0])
        last_tag = line[3]
    entities.append(entity)

testing_data = []
for test, entity in tqdm(zip(testing, entities), total=len(testing)):
    gt = ' '.join([line[0] for line in test])
    out_str = []
    phrase_num = []
    for i, e in enumerate(entity):
        tokens = tokenizer.tokenize(' '.join(e))
        out_str.append(' '.join(e))
        phrase_num += [i] * len(tokens)
    # add begin and end token
    phrase_num = [-1] + phrase_num + [-1]
    testing_data.append((" [CLS] " + " ".join(out_str) + " [SEP] ", gt, phrase_num))

with open(OUTPUT + "CoNLL_test_esai", "wb") as fout:
    pk.dump(testing_data, fout)
