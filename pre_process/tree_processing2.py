"""
this file aims to eliminate re-tweets in twitter15 trees
file output: 1. processed_data/proccessed_tweet_tree
             2. tweet2tree_id (dictionary)
file form:
[tweet_id, user_id, text(string)] -> [tweet_id, user_id, text(string)]
"""
import csv
import os
import os.path as pth
import pandas as pd
from tqdm import tqdm
from pre_process import tools
from collections import defaultdict
from pre_process import tree_checker
from tree_checker import parse_record
from tqdm import tqdm
from pre_process.tools import save_dict_to_json, read_dict_from_json

def get_tweet_record(tweet_id, df, recorded_list):
    if tweet_id not in recorded_list:
        raise Exception("tweet_id not in recorded_list")
    idx = df[df['tweet_id'].astype(str) == str(tweet_id)].index.tolist()
    if len(idx) != 1:
        raise Exception(
            "Error! index:{}, tweet_id:{}, recorded_list: {}".format(idx, tweet_id, recorded_list)
        )
    idx = idx[0]
    u_id, tex = df.loc[idx, 'tweet_id'], df.loc[idx, 'text']
    return str(u_id), str(tex)



def build_graph(lines, source_id):
    parent2child = defaultdict(list)
    parent2child[source_id] = []
    for line in lines:
        if '->' not in line:
            continue
        line = line.rstrip()
        _, _, parent_tweet, child_tweet = parse_record(line)
        if parent_tweet != child_tweet and child_tweet not in parent2child[parent_tweet]:
            parent2child[parent_tweet].append(child_tweet)
    return parent2child


def dfs(cur):
    visit.add(cur)
    print('visit add: {}'.format(cur))
    if cur not in parent2child.keys():
        return None
    for y in parent2child[cur]:
        # if y != father and y in visit:
        if y in visit:
            del_edge.add((cur, y))  # parent2child[cur].remove(y)
        else:
            print('goto y:', y)
            dfs(y)


tree_tree_dir = pth.join('../datasets/twitter16/raw_data/tree_tree')
tweet2mat_dict = read_dict_from_json('../datasets/twitter16/raw_data/tweet2matrix_idx.json')


node = 0
tweet2tree_id_dict = dict()


for f in tqdm(os.listdir(tree_tree_dir)):
    source_id = f.split('.')[0]
    tree_num = tweet2mat_dict[source_id]
    tree_id = str(tree_num) + '_0'
    print(tree_id)

    fr = open(pth.join(tree_tree_dir, f), 'r')
    lines = fr.readlines()
    fr.close()

    del_edge = set()
    del_node = set()
    # build graph
    parent2child = build_graph(lines, source_id)
    # print('original p2c', len(parent2child))
    visit = set()
    dfs(source_id)

    for parent, child_list in parent2child.items():
        if parent not in visit:
            del_node.add(parent)
        for child in child_list:
            if child not in visit:
                del_edge.add((parent, child))

    for (parent, child) in del_edge:
        parent2child[parent].remove(child)
    for false_node in del_node:
        del parent2child[false_node]

    print('p2c after dfs', sum([len(i) for i in parent2child.values()]))
    length = 0
    for i, lt in parent2child.items():
        length += len(lt)
    if length != len(visit)-1:
        print('length', length)
        print('visit', len(visit))
        raise ValueError('source_id', source_id)

    idx = 0
    tree_id = str(tree_num) + '_' + str(idx)
    tweet2tree_id_dict[tree_id] = source_id
    numbered_list = []
    numbered_list.append(source_id)
    for p, child_list in parent2child.items():
        # if p in tweet2tree_id_dict.keys():
        #     pass
        # if p not in visit: # not connected to root
        #     pass
        if p not in numbered_list and p in visit:
            idx += 1
            tree_id = str(tree_num) + '_' + str(idx)
            # print('add parent tree_id: {}'.format(tree_id))
            tweet2tree_id_dict[tree_id] = p

        for child in child_list:
            if child not in visit or child in numbered_list:
                continue
            idx += 1
            tree_id = str(tree_num) + '_' + str(idx)
            # print('add tree_id: {}'.format(tree_id))
            tweet2tree_id_dict[tree_id] = child
            numbered_list.append(child)


tweet2tree_id_dict_path = pth.join('../datasets/twitter16/auxiliary_data/tweet2tree_dict.json')
save_dict_to_json(tweet2tree_id_dict, tweet2tree_id_dict_path)





























#
#
#
#
#
# tools.save_dict_to_json(tree_dictionary, tree_dictionary_save_path)
# tools.iterable2txt(tweet2tree_id_dict, tweet2tree_id_path)
#
#
#
#






#
#
#
#
#
# print("--------------------finish task1 ----------------------------")
#
# with open(output_comments_text_path, 'w', encoding='utf-8', newline='') as wcsv:
#     csv_write = csv.writer(wcsv)
#     csv_write.writerow(['tree_id', 'tweet_id', 'text'])
#
#     for tweet_id, tree_id in tweet2tree_id_dict.items():
#         if tree_id.split('_')[1] == '0':  # is source, change recorded_list and df
#             node_id = tree_id.split('_')[0]
#             text_content = pd.read_csv(pth.join(input_comments_text_dir, tweet_id + '.csv'), encoding='utf-8')
#             df = pd.DataFrame(text_content)
#             df = df.where(df.notnull(), None)
#             recorded_comments_list = df['tweet_id'].astype(str).tolist()
#
#         t_id, text = get_tweet_record(tweet_id, df, recorded_comments_list)
#         newrow = [tree_id, t_id, text]
#         # print(newrow)
#         csv_write.writerow(newrow)
#
# print('write to {}'.format(output_comments_text_path))
# print("------------------finish task2--------------------")
#
# # print(len(tweet2tree_id_dict))
# # print("abnormal_count: {}".format(abnormal_count))
# # print("abnormal_list:{}".format(abnormal_list))
