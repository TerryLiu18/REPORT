"""help check record of tree file"""

import os
import os.path as pth
import pandas as pd
from pre_process import tools

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
        t_list = df['tweet_id'].astype(str).tolist()
        available_tweet_set.update(t_list)
    # print(available_tweet_set)
    tools.iterable2txt(
        available_tweet_set,
        pth.join(tweet_set_path)
    )


# get all available user_set
user_set_path = pth.join('../auxiliary_data/available_user_set.txt')
if pth.exists(user_set_path):
    available_user_set = tools.txt2iterable(user_set_path)
else:
    for f in os.listdir(user_dir):
        f_path = pth.join(user_dir, f)
        df = pd.read_csv(f_path, encoding='utf-8', lineterminator='\n')
        t_list = df['id'].astype(str).tolist()
        available_user_set.update(t_list)
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


def get_node_set(file):
    source_id = file.split('.')[0]
    node_set = set()
    node_set.add(source_id)
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if '->' not in line:
                continue
            user1, user2, tweet1, tweet2 = parse_record(line)
            node_set.add(tweet2)
    return node_set


# is linked_list needed here?

def check_ancestor(tweet, source_tweet, child2parent):
    """
    :param tweet: tweet to be check
    :param child2parent: dictionary
    :return: True/False  if connected to ancestor, True
    """
    count = 0
    while tweet != source_tweet:
        if count >= 50:
            isloop = 1
            break
        if tweet in child2parent:
            count += 1
            tweet = child2parent[tweet]
        else:
            break
    return tweet is source_tweet


def tree_line_checker(lines, source_id):
    """
    check if each edge is a available record
    :param file: file path
    :return: parent_available_dict  eg: ['1234':True, '3123':False]
    """
    parent_record_dict = {}
    child2parent = {}
    for line in lines:
        if '->' not in line:
            continue
        _, _, parent_tweet, child_tweet = parse_record(line)
        if parent_tweet != child_tweet:
            child2parent[child_tweet] = parent_tweet
    for line in lines:
        if '->' not in line:
            continue
        _, _, parent_tweet, child_tweet = parse_record(line)
        if parent_tweet == child_tweet:
            continue
        # print(parent_tweet, child_tweet)
        new_record_parent = parent_tweet

        count = 0
        isloop = 0
        while parent_tweet != source_id:
            if count >= 50:
                isloop = 1
                break
            if parent_tweet in child2parent:
                count += 1
                parent_tweet = child2parent[parent_tweet]
            else:
                break
        if parent_tweet == source_id:
            parent_record_dict[new_record_parent] = True
        else:
            parent_record_dict[new_record_parent] = False
        if isloop:
            parent_record_dict[new_record_parent] = True
    parent_record_dict[source_id] = True
    return parent_record_dict


if __name__ == '__main__':
    file_dir = pth.join('../datasets/twitter16/raw_data/tree_tree')
    for f in os.listdir(file_dir):
        source_id = f.split('.')[0]
        f = open(pth.join(file_dir, f), 'r')
        lines = f.readlines()
        f.close()
        pdict = tree_line_check(lines, source_id)
        print(pdict)