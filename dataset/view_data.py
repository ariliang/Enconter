# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import pickle


ROOT = '/home/ariliang/Local-Data/Enconter/'
DATASET = ROOT + 'dataset/'


# %%
# pointer_e processed

os.chdir(DATASET)

pointer_e = pickle.load(open('CoNLL_pointer_e', 'rb'))

print('length: ', len(pointer_e))
start = 1
truncate = 20
for tpl in pointer_e[start:start+1]:
    first, second = tpl
    print(first[:truncate])
    print()
    print(second[:truncate])
    print()

os.chdir(ROOT)


# %%
# greedy_enconter

os.chdir(DATASET)

greedy_enconter = pickle.load(open('CoNLL_greedy_enconter', 'rb'))

print('length: ', len(greedy_enconter))
start = 1
truncate = 20
for tpl in greedy_enconter[start:start+1]:
    first, second = tpl
    print(first[:truncate])
    print()
    print(second[:truncate])

os.chdir(ROOT)


# %%
# bbt_enconter

os.chdir(DATASET)

bbt_enconter = pickle.load(open('CoNLL_bbt_enconter', 'rb'))

print('length: ', len(bbt_enconter))
start = 1
truncate = 20
for tpl in bbt_enconter[start:start+1]:
    first, second = tpl
    print(first[:truncate])
    print()
    print(second[:truncate])

os.chdir(ROOT)
