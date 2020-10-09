"""help check record of tree file"""

import os
import os.path as pth
import pandas as pd
from collections import defaultdict
from pre_process import tools
from tqdm import tqdm

original_tree_dir = pth.join('../raw_data/tweet_tree')
tweet_dir = pth.join('../raw_data/tweet_profile')
user_dir = pth.join('../raw_data/user_profile')
available_tweet_set = set()
available_user_set = set()

source_tweet_list = [i.split('.')[0] for i in os.listdir(original_tree_dir)]

# get all available tweet_set
tweet_set_path = pth.join('../auxiliary_data/available_tweet_set.txt')
if pth.exists(tweet_set_path):
    available_tweet_set = tools.txt2iterable(tweet_set_path)
else:
    for f in os.listdir(tweet_dir):
        f_path = pth.join(tweet_dir, f)
        df = pd.read_csv(f_path, encoding='utf-8', lineterminator='\n')
        try:
            t_list = df['id'].astype(str).tolist()
            t_list = [i.split('.')[0] for i in t_list]
            available_tweet_set.update(t_list)
        except:
            print(f)
            raise ValueError

print('len(available_tweet_set): {}'.format(len(available_tweet_set)))
tools.iterable2txt(
    available_tweet_set,
    pth.join(tweet_set_path)
)


# get all available user_set
user_set_path = pth.join('../auxiliary_data/available_user_set.txt')
if pth.exists(user_set_path):
    available_user_set = tools.txt2iterable(user_set_path)
else:
    for f in tqdm(os.listdir(user_dir)):
        f_path = pth.join(user_dir, f)
        df = pd.read_csv(f_path, encoding='utf-8', lineterminator='\n')
        t_list = df['id'].astype(str).tolist()
        t_list = [i.split('.')[0] for i in t_list]
        available_user_set.update(t_list)

print('len(available_user_set): {}'.format(len(available_user_set)))
tools.iterable2txt(
    available_user_set,
    pth.join(user_set_path)
)


def parse_record(line):
    line = str(line).rstrip()
    two_record = line.split('->')
    record0 = eval(two_record[0])
    record1 = eval(two_record[1])
    user1, tweet1 = record0[0], record0[1]
    user2, tweet2 = record1[0], record1[1]
    return user1, user2, tweet1, tweet2



def early_parse_record(line):
    line = str(line).rstrip()
    two_record = line.split('->')
    record0 = eval(two_record[0])
    record1 = eval(two_record[1])
    user1, tweet1, time1 = record0[0], record0[1], float(record0[2])
    user2, tweet2, time2 = record1[0], record1[1], float(record1[2])
    # if time1 > time2:
    #     print('a loop appear')
    return user1, user2, tweet1, tweet2, time1, time2



#
#
#
# def get_node_set(file):
#     source_id = file.split('.')[0]
#     node_set = set()
#     node_set.add(source_id)
#     with open(file, 'r') as f:
#         lines = f.readlines()
#         for line in lines[:-1]:
#             if '->' not in line:
#                 continue
#             user1, user2, tweet1, tweet2 = parse_record(line)
#             node_set.add(tweet2)
#     return node_set
#
#
# # is linked_list needed here?
#
# def check_ancestor(tweet, source_tweet, child2parent):
#     """
#     :param tweet: tweet to be check
#     :param child2parent: dictionary
#     :return: True/False  if connected to ancestor, True
#     """
#     count = 0
#     while tweet != source_tweet:
#         if count >= 50:
#             isloop = 1
#             break
#         if tweet in child2parent:
#             count += 1
#             tweet = child2parent[tweet]
#         else:
#             break
#     return tweet is source_tweet
#
# def build_graph(lines, source_id):
#     parent2child = defaultdict(list)
#     for line in lines:
#         if '->' not in line:
#             continue
#         line = line.rstrip()
#         _, _, parent_tweet, child_tweet = parse_record(line)
#         if parent_tweet != child_tweet:
#             parent2child[parent_tweet].append(child_tweet)
#     return parent2child
#
