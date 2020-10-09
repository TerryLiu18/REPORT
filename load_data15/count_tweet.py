from tools import read_dict_from_json, save_dict_to_json
from task import FILTER_NUM, SOURCE_TWEET_NUM

def get_tail(num):
    """
    :param num:'0_3'
    :return: 3
    """
    return int(num.split('_')[1])


def int2rec(node_id, num):
    """
    :param node_id: 0
    :param num: 3
    :return: '0_3'
    """
    return str(node_id)+'_'+str(num)


def sort_record(rec_list):
    """
    :param rec_list:  ['0_3', '0_1', '0_2', '0_0']
    :return: ['0_0','0_1', '0_2', '0_3', ]
    """
    if not rec_list:  # empty list
        return rec_list
    node_id = rec_list[0].split('_')[0]
    rec_list = [get_tail(i) for i in rec_list]
    max_idx = max(rec_list)
    sorted_record_list = [int2rec(node_id, i) for i in range(max_idx+1)]
    return sorted_record_list


no_sort_tree_dict = read_dict_from_json('tree_dictionary.json')
for num in no_sort_tree_dict.keys():
    one_tree_record = no_sort_tree_dict[num]
    for node_id, child_list in one_tree_record.items():
        if len(child_list) != 0:
            child_list.sort(key=lambda i: int(i.split('_')[1]))
save_dict_to_json(no_sort_tree_dict, 'tree_dictionary.json')


tweet_num = 0
tweet_set = set()
tree_dict = read_dict_from_json('tree_dictionary.json')
for num in tree_dict.keys():
    for node_id, child_list in tree_dict[num].items():
        tweet_set.add(node_id)
        if len(child_list) != 0:
            tweet_set.update(child_list)
        tweet_num += len(child_list)
tweet_num += SOURCE_TWEET_NUM
print(tweet_num)

import pandas as pd
df = pd.read_csv('comments_text_encode1.csv')
tweet_list = df['tree_id'].tolist()
tweet_set2 = set(tweet_list)
print(len(tweet_set2))

loss_set = tweet_set2-tweet_set
print(loss_set)