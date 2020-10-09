"""check if user is available in graph"""

import os
import os.path as pth
from tqdm import tqdm
import tree_checker

original_tweet_dir = tree_checker.tweet_dir
original_user_dir = tree_checker.user_dir
original_tree_dir = tree_checker.original_tree_dir

tweet_set = tree_checker.available_tweet_set
user_set = tree_checker.available_user_set
graph_tree_dir = pth.join('../raw_data/graph_tree')
tree_tree_dir = pth.join('../raw_data/tree_tree')

if not pth.exists(graph_tree_dir):
    os.mkdir(graph_tree_dir)

if not pth.exists(tree_tree_dir):
    os.mkdir(tree_tree_dir)

# tweet_set = tweet_set.add('ROOT')
# user_set = user_set.add('ROOT')

# get graph tree
for f in tqdm(os.listdir(original_tree_dir)):
    f_path = pth.join(original_tree_dir, f)
    new_f_path = pth.join(graph_tree_dir, f)
    with open(f_path, 'r') as fr, open(new_f_path, 'w') as fw:
        lines = fr.readlines()
        for line in lines[1:]:
            user1, user2, tweet1, tweet2 = tree_checker.parse_record(line)
            if str(user1) not in user_set or str(user2) not in user_set:
                continue
            fw.write(line)


# get tree tree
for f in tqdm(os.listdir(original_tree_dir)):
    f_path = pth.join(original_tree_dir, f)
    new_f_path = pth.join(tree_tree_dir, f)
    with open(f_path, 'r') as fr, open(new_f_path, 'w') as fw:
        lines = fr.readlines()
        for line in lines[1:]:
            user1, user2, tweet1, tweet2 = tree_checker.parse_record(line)
            if str(tweet1) not in tweet_set or str(tweet2) not in tweet_set:
                continue
            fw.write(line)