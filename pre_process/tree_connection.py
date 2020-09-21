"""
processed files form:

** 0. two kinds of id:
node_id: 0, 1, 2, 3... 1311
tree_id: tree_ids are unique identifier for tweets in a tree:
tree_id = node_id + '_' + record_num,   eg: 0_0, 0_1, 0_2... 1_0, 1_1...1311_0, 1311_1...

** 1. tree structure (dict[dict[list]]):
structure of tree_dictionary:  node_id[node_id][tree_id][tree_id] = child
example:
    {
      0:{0_0:[0_1, 0_2, 0_3, 0_5], 0_1:[0_4]}
      1:{1_0:[1_1, 1_2, 1_4], 0_2:[0_3]}
      2: ....
      ...
      1311:{1311_0:[]...}
    }

** 2. tweet text (.csv)
tree_id, user_id, text(string)
0_0:  2312313212, "this is a string"
0_1:  242342222,  "this is a string"
...
1_0: ...
...
1311_0... "there will be a map between tree_id and tweet_id"
"""

import os
import os.path as pth
import csv
import pandas as pd
import numpy as np
from tools import txt2iterable, iterable2txt, save_dict_to_json, read_dict_from_json
from tqdm import tqdm
from pre_process import tree_checker


input_tree_dir = pth.join('../datasets/twitter16/processed_data/processed_tweet_tree')
input_tree_list = os.listdir(input_tree_dir)
tweet2tree_id_dict = txt2iterable('../datasets/twitter16/processed_data/tweet2tree_id_dict.txt')
tweet2node_dict = read_dict_from_json(pth.join('../datasets/twitter16/raw_data/tweet2matrix_idx.json'))
print(len(tweet2tree_id_dict))

tree_dictionary = dict()
tree_dictionary_save_path = pth.join('../load_data16/backup/tree_dictionary_backup.json')
input_comments_text_dir = pth.join('../datasets/twitter16/processed_data/processed_tweet_profile')
output_comments_text_path = pth.join('../load_data16/comments_text.csv')

# structure of tree_dictionary:  node_id[node_id][tree_id][tree_id] = child
"""
{
    0: {0_0: [0_1, 0_2, 0_3, 0_5], 0_1: [0_4]}
    1: {1_0: [1_1, 1_2, 1_4], 0_2: [0_3]}
    1311: {1311_0: []...}
}
"""


def _get_tweet_record(tweet_id, df, recorded_list):
    if str(tweet_id) not in recorded_list:
        raise Exception("tweet_id not in recorded_list: {}, {}".format(tweet_id, recorded_list))

    idx = df[df['tweet_id'].astype(str) == str(tweet_id)].index.tolist()
    # df_new = df.set_index('tweet_id', drop=True, append=False, inplace=False, verify_integrity=False)

    if len(idx) != 1:
        raise Exception("Error! index:{}, tweet_id:{}, recorded_list: {}"
                        .format(idx, tweet_id, recorded_list))
    idx = idx[0]
    # u_id, tex, retweet_cnt, favorite_cnt = \
    #     df.loc[idx, 'user_id'], df.loc[idx, 'text'], df.loc[idx, 'retweet_count'], df.loc[idx, 'favorite_count']
    # return str(u_id), str(tex), retweet_cnt, favorite_cnt
    # print(df.columns)
    t_id, tex = df.loc[idx, 'tweet_id'], df.loc[idx, 'text']
    return str(t_id), str(tex)


for file in tqdm(input_tree_list):
    tree_record = dict()  #  {0_0: [0_1, 0_2, 0_3, 0_5], 0_1: [0_4]}
    source_id = file.split('.')[0]
    node_id = tweet2node_dict[source_id]
    node_id = str(node_id)  # "0, 1, 2... 1311"
    written_tree_list = []
    with open(pth.join(input_tree_dir, file), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        parent_record_dict = tree_checker.tree_line_checker(lines, source_id)
        for line in lines:
            line = line.rstrip()
            _, _, tweet0, tweet1 = tree_checker.parse_record(line)
            if tweet0 not in parent_record_dict or not parent_record_dict[tweet0]:
                continue
            if tweet1 not in parent_record_dict or not parent_record_dict[tweet1]:
                continue
            if tweet0 == tweet1:
                continue
            tree_id0 = tweet2tree_id_dict[tweet0]
            tree_id1 = tweet2tree_id_dict[tweet1]

            if str(tree_id0) not in tree_record.keys():
                tree_record[tree_id0] = []
                tree_record[tree_id0].append(tree_id1)
                written_tree_list.append(tree_id1)
            if str(tree_id0) in tree_record.keys():
                tree_record[tree_id0].append(tree_id1)
                written_tree_list.append(tree_id1)

            # if str(tree_id0) in written_tree_list and str(tree_id0) not in tree_record.keys():
            #     # child becomes a new parent node
            #     tree_record[tree_id0] = []
            #     written_tree_list.append(tree_id0)
            # if str(tree_id0) in tree_record.keys() and str(tree_id0) in written_tree_list:
            #     # there is a tree_record[tree_id0] = [] already
            #     tree_record[tree_id0].append(tree_id1)
            #     written_tree_list.append(tree_id1)
        tree_dictionary[node_id] = tree_record


save_dict_to_json(tree_dictionary, tree_dictionary_save_path)
print(tree_dictionary_save_path)
# user_id = first_record[0]
print("--------------------finish task1 ----------------------------")

with open(output_comments_text_path, 'w', encoding='utf-8', newline='') as wcsv:
    csv_write = csv.writer(wcsv)
    csv_write.writerow(['tree_id', 'tweet_id', 'text'])

    for source_id, tree_id in tweet2tree_id_dict.items():
        if tree_id.split('_')[1] == '0':  # is source, change recorded_list and df
            node_id = tree_id.split('_')[0]
            text_content = pd.read_csv(pth.join(input_comments_text_dir, source_id + '.csv'), encoding='utf-8')
            df = pd.DataFrame(text_content)
            df = df.where(df.notnull(), None)
            recorded_comments_list = df['tweet_id'].astype(str).tolist()

        t_id, text = _get_tweet_record(source_id, df, recorded_comments_list)
        newrow = [tree_id, t_id, text]
        # print(newrow)
        csv_write.writerow(newrow)

print('write to {}'.format(output_comments_text_path))
print("------------------finish task2--------------------")