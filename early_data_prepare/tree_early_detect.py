"""check if user is available in graph"""

import os
import os.path as pth
from tqdm import tqdm
import tree_checker
from tree_checker import early_parse_record
from task import THRESHOLD_TIME

# THRESHOLD_TIME = 5   # 10, 20, 30, 60 ,90, 120

original_tree_dir = pth.join('../raw_data/tweet_tree')
early_tree_dir = pth.join('../raw_data/early_{}minutes_tweet_tree'.format(THRESHOLD_TIME))
early_graph_tree_dir = pth.join('../raw_data/early_{}minutes_graph_tree'.format(THRESHOLD_TIME))
early_tree_tree_dir = pth.join('../raw_data/early_{}minutes_tree_tree'.format(THRESHOLD_TIME))
tweet_dir = pth.join('../raw_data/tweet_profile')
user_dir = pth.join('../raw_data/user_profile')

if not pth.exists(early_tree_dir):
    os.mkdir(early_tree_dir)
if not pth.exists(early_graph_tree_dir):
    os.mkdir(early_graph_tree_dir)
if not pth.exists(early_tree_tree_dir):
    os.mkdir(early_tree_tree_dir)

delete_list = []

# delete trees whose first records are too late to appear
for f in os.listdir(original_tree_dir):
    f_path = pth.join(original_tree_dir, f)
    with open(f_path, 'r') as fr:
        lines = fr.readlines()
        user1, user2, tweet1, tweet2, time1, time2 = early_parse_record(lines[1])
        if time1 > THRESHOLD_TIME or time2 > THRESHOLD_TIME:
            delete_item = f.split('.')[0]
            delete_list.append(delete_item)

print('delete_list: {}'.format(delete_list))
print('length of delete_list: {}'.format(len(delete_list)))

original_tree_list = [i.split('.')[0] for i in os.listdir(original_tree_dir)]
new_tree_list = [i for i in original_tree_list if i not in delete_list]
print('len(new_tree_list): {}'.format(len(new_tree_list)))
print('new_tree_list: {}'.format(new_tree_list))

for source_id in tqdm(new_tree_list):
    f_path = pth.join(original_tree_dir, source_id+'.txt')
    new_f_path = pth.join(early_tree_dir, source_id+'.txt')
    with open(f_path, 'r') as fr, open(new_f_path, 'w') as fw:
        lines = fr.readlines()
        for line in lines:
            user1, user2, tweet1, tweet2, time1, time2 = tree_checker.early_parse_record(line)
            if time1 > THRESHOLD_TIME or time2 > THRESHOLD_TIME:
                break
            fw.write(line)
            flag = 1



# THRESHOLD_TIME = 10   # 10, 20, 30, 60 ,90, 120
# original_tree_dir = pth.join('../raw_data/tweet_tree')
# early_tree_dir = pth.join('../raw_data/early_{}minutes_tweet_tree'.format(THRESHOLD_TIME))
# early_graph_tree_dir = pth.join('../raw_data/early_{}minutes_graph_tree'.format(THRESHOLD_TIME))
# early_tree_tree_dir = pth.join('../raw_data/early_{}minutes_tree_tree'.format(THRESHOLD_TIME))
# if not pth.exists(early_tree_dir):
#     os.mkdir(early_tree_dir)
# if not pth.exists(early_graph_tree_dir):
#     os.mkdir(early_graph_tree_dir)
# if not pth.exists(early_tree_tree_dir):
#     os.mkdir(early_tree_tree_dir)
#
# THRESHOLD_TIME = 20   # 10, 20, 30, 60 ,90, 120
# early_tree_dir = pth.join('../raw_data/early_{}minutes_tweet_tree'.format(THRESHOLD_TIME))
# early_graph_tree_dir = pth.join('../raw_data/early_{}minutes_graph_tree'.format(THRESHOLD_TIME))
# early_tree_tree_dir = pth.join('../raw_data/early_{}minutes_tree_tree'.format(THRESHOLD_TIME))
# if not pth.exists(early_tree_dir):
#     os.mkdir(early_tree_dir)
# if not pth.exists(early_graph_tree_dir):
#     os.mkdir(early_graph_tree_dir)
# if not pth.exists(early_tree_tree_dir):
#     os.mkdir(early_tree_tree_dir)
#
# THRESHOLD_TIME = 30   # 10, 20, 30, 60 ,90, 120
# early_tree_dir = pth.join('../raw_data/early_{}minutes_tweet_tree'.format(THRESHOLD_TIME))
# early_graph_tree_dir = pth.join('../raw_data/early_{}minutes_graph_tree'.format(THRESHOLD_TIME))
# early_tree_tree_dir = pth.join('../raw_data/early_{}minutes_tree_tree'.format(THRESHOLD_TIME))
# if not pth.exists(early_tree_dir):
#     os.mkdir(early_tree_dir)
# if not pth.exists(early_graph_tree_dir):
#     os.mkdir(early_graph_tree_dir)
# if not pth.exists(early_tree_tree_dir):
#     os.mkdir(early_tree_tree_dir)
#
# THRESHOLD_TIME = 60   # 10, 20, 30, 60 ,90, 120
# early_tree_dir = pth.join('../raw_data/early_{}minutes_tweet_tree'.format(THRESHOLD_TIME))
# early_graph_tree_dir = pth.join('../raw_data/early_{}minutes_graph_tree'.format(THRESHOLD_TIME))
# early_tree_tree_dir = pth.join('../raw_data/early_{}minutes_tree_tree'.format(THRESHOLD_TIME))
# if not pth.exists(early_tree_dir):
#     os.mkdir(early_tree_dir)
# if not pth.exists(early_graph_tree_dir):
#     os.mkdir(early_graph_tree_dir)
# if not pth.exists(early_tree_tree_dir):
#     os.mkdir(early_tree_tree_dir)
#
# THRESHOLD_TIME = 90   # 10, 20, 30, 60 ,90, 120
# early_tree_dir = pth.join('../raw_data/early_{}minutes_tweet_tree'.format(THRESHOLD_TIME))
# early_graph_tree_dir = pth.join('../raw_data/early_{}minutes_graph_tree'.format(THRESHOLD_TIME))
# early_tree_tree_dir = pth.join('../raw_data/early_{}minutes_tree_tree'.format(THRESHOLD_TIME))
# if not pth.exists(early_tree_dir):
#     os.mkdir(early_tree_dir)
# if not pth.exists(early_graph_tree_dir):
#     os.mkdir(early_graph_tree_dir)
# if not pth.exists(early_tree_tree_dir):
#     os.mkdir(early_tree_tree_dir)
#
# THRESHOLD_TIME = 120   # 10, 20, 30, 60 ,90, 120
# early_tree_dir = pth.join('../raw_data/early_{}minutes_tweet_tree'.format(THRESHOLD_TIME))
# early_graph_tree_dir = pth.join('../raw_data/early_{}minutes_graph_tree'.format(THRESHOLD_TIME))
# early_tree_tree_dir = pth.join('../raw_data/early_{}minutes_tree_tree'.format(THRESHOLD_TIME))
# if not pth.exists(early_tree_dir):
#     os.mkdir(early_tree_dir)
# if not pth.exists(early_graph_tree_dir):
#     os.mkdir(early_graph_tree_dir)
# if not pth.exists(early_tree_tree_dir):
#     os.mkdir(early_tree_tree_dir)