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
# from utils import processed_data_dir
from tqdm import tqdm
from pre_process import tree_checker
# from pre_process.tree_profile_processing import _get_tweet_record

project_name = 'MyGAT'
loc = os.path.abspath(os.path.dirname(__file__)).split(project_name)[0]
root_path = pth.join(loc, project_name)
# dataset_path = pth.join(root_path, 'datasets', 'twitter15')
dataset_path = pth.join(root_path, 'datasets', 'twitter16')
processed_data_dir = pth.join(dataset_path, 'processed_data')
auxiliary_data_dir = pth.join(dataset_path, 'auxiliary_data')

input_tree_dir = pth.join(processed_data_dir, 'processed_tweet_tree')
input_tree_list = os.listdir(input_tree_dir)
tweet2tree_id_dict = txt2iterable(pth.join(processed_data_dir, 'tweet2tree_id_dict.txt'))
# tweet2node_dict = read_dict_from_json(pth.join(auxiliary_data_dir, 'tweet2node_dict.json'))
tweet2node_dict = read_dict_from_json(pth.join('../datasets/twitter16/raw_data/tweet2matrix_idx.json'))
print(len(tweet2tree_id_dict))

tree_dictionary = dict()
tree_dictionary_save_path = pth.join(processed_data_dir, 'tweet_connection_structure', 'tree_dictionary.txt')
tree_dictionary_save_path2 = pth.join(root_path, 'load_data', 'tree_dictionary.json')
tree_dictionary_save_path2 = pth.join(root_path, 'load_data16', 'tree_dictionary.json')

input_comments_text_dir = pth.join(processed_data_dir, 'processed_tweet_tree_profile')
output_comments_text_path = pth.join('../load_data16/comments_text.csv')
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
    tweet_id = file.split('.')[0]
    node_id = tweet2node_dict[tweet_id]
    # print(tweet_id, node_id)
    node_id = str(node_id)  # "0, 1, 2... 1311"
    written_tree_list = []
    f_path = pth.join(input_tree_dir, file)
    node_set = tree_checker.get_node_set(f_path)
    with open(f_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # todo: adpat to the file if there is no 'ROOT' in the first line
        # firstline = lines[1].rstrip()
        # record_num = 0
        # two_list = firstline.split('-->')
        #
        # first_record = eval(two_list[1])
        # tweet_id = first_record[1]
        # tree_id = tweet2tree_id_dict[tweet_id]  # 0_0, 1_0 ...
        # tree_record[tree_id] = []
        # written_tree_list.append(tree_id)

        for line in lines[2:]:
            line = line.rstrip()
            two_record = line.split('-->')
            record0 = eval(two_record[0])
            record1 = eval(two_record[1])
            tweet0 = record0[1]
            tweet1 = record1[1]
            if tweet0 not in node_set or tweet1 not in node_set:
                continue
            tree_id0 = tweet2tree_id_dict[tweet0]
            tree_id1 = tweet2tree_id_dict[tweet1]
            if str(tree_id1) in tree_record.keys():
                raise Exception("tree_id1 in tree_record.keys()")

            if str(tree_id0) not in tree_record.keys() and str(tree_id0) not in written_tree_list:
                # print(line)
                raise Exception("tree_id0 not in tree_record.keys()")
            if str(tree_id0) in written_tree_list and str(tree_id0) not in tree_record.keys():
                # child becomes a new parent node
                tree_record[tree_id0] = []
                written_tree_list.append(tree_id0)
            if str(tree_id0) in tree_record.keys() and str(tree_id0) in written_tree_list:
                # there is a tree_record[tree_id0] = [] already
                tree_record[tree_id0].append(tree_id1)
                written_tree_list.append(tree_id1)
        # print(node_id)
        tree_dictionary[node_id] = tree_record
# print(tree_dictionary)

save_dict_to_json(tree_dictionary, tree_dictionary_save_path2)
print(tree_dictionary_save_path2)
# user_id = first_record[0]
print("--------------------finish task1 ----------------------------")

with open(output_comments_text_path, 'w', encoding='utf-8', newline='') as wcsv:
    csv_write = csv.writer(wcsv)
    csv_write.writerow(['tree_id', 'tweet_id', 'text'])

    for tweet_id, tree_id in tweet2tree_id_dict.items():
        if tree_id.split('_')[1] == '0':  # is source, change recorded_list and df
            node_id = tree_id.split('_')[0]
            text_content = pd.read_csv(pth.join(input_comments_text_dir,\
                                                tweet_id + '.csv'), encoding='utf-8')
            df = pd.DataFrame(text_content)
            df = df.where(df.notnull(), None)
            recorded_comments_list = df['tweet_id'].astype(str).tolist()

        # u_id, text, retweet_cnt, favorite_cnt = _get_tweet_record(tweet_id, df, recorded_comments_list)
        # newrow = [tree_id, tweet_id, u_id, text, retweet_cnt, favorite_cnt]
        #
        t_id, text = _get_tweet_record(tweet_id, df, recorded_comments_list)
        newrow = [tree_id, t_id, text]
        # print(newrow)
        csv_write.writerow(newrow)

print("------------------finish task2--------------------")