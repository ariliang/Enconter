#! /usr/bin/env python3
# self contrained

import pickle


LOCAL_DATA = '/home/ariliang/Local-Data/'
ROOT = LOCAL_DATA + 'Enconter/'
RAW_DIR = LOCAL_DATA + 'models_datasets/ccks21/'
OUTPUT = ROOT + 'output/'


# recurrsive sampling
def process_train_helper(data):

    concated_dialog = []

    for dialog in data:

        new_dialog = [dialog[0].get('Sentence')]
        new_entity = []
        for ents in list(dialog[0].values())[2:]:
            new_entity.extend(ents)
            # new_entity = reversed(list(set(new_entity)))

        for i in range(1, len(dialog)):

            if dialog[i-1].get('id') == dialog[i].get('id'):
                new_dialog[-1] += dialog[i].get('Sentence')
                # add entities
                for ents in list(dialog[i].values())[2:]:
                    new_entity.extend(ents)
                    # new_entity = reversed(list(set(new_entity)))

            else:
                new_dialog[-1] += ''.join([ '[ENT]'+ent+'[ENT]'  for ent in set(new_entity) ])

                new_entity = []
                for ents in list(dialog[i].values())[2:]:
                    new_entity.extend(ents)
                    # new_entity = reversed(list(set(new_entity)))
                new_dialog.append(dialog[i].get('Sentence'))

        new_dialog[-1] += ''.join([ '[ENT]'+ent+'[ENT]'  for ent in set(new_entity) ])

        if dialog[-1].get('id') == 'Patients':
            new_dialog = new_dialog[:-1]

        concated_dialog.append(new_dialog)


    with open(OUTPUT + 'dialo/train_ent.txt', 'w') as fw:
        for i, dialog in enumerate(concated_dialog):
            # fw.write(str(i))
            # fw.write('\n')
            for line in dialog:
                fw.write(line)
                fw.write('\n')
            fw.write('\n')

    # recurrsive
    # with open(OUTPUT + 'dialo/train_ent_grouped_phaseb.txt', 'w') as fw:
    #     for dialog in concated_dialog:
    #        # fw.write(str(i))
    #        # fw.write('\n')
    #         for i in range(2, len(dialog), 2):
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


def main():
    process_train()
    # process_dev()
    pass


if __name__ == '__main__':
    main()