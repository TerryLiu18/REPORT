import json
import random

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def read_dict_from_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print("{} successfully loaded!".format(path))
    return data


def _split_train_test(index2label):
    # random.seed(0)
    label2index = dict()
    train_indices = []
    test_indices = []
    ratio = 0.8
    for index, label in index2label.items():
        label2index.setdefault(label, []).append(index)
    # print(label2index[0])
    for label in label2index:
        index_list = label2index[label]
        # print(index_list)
        k = int(len(index_list) * ratio)
        # print("k", k)
        train_indices += index_list[:k]
        test_indices += index_list[k:]
    
    # random.shuffle(train_indices)
    # random.shuffle(test_indices)
    # print("train", sorted(train_indices))
    # print('-'*89)
    # print(test_indices)
    # print(len(train_indices))
    return train_indices, test_indices
