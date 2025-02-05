import pandas as pd
import os.path as pth
import json
import csv
from task import SOURCE_TWEET_NUM, FILTER_NUM, DATASET_NAME, LOAD_DATA

filter_num = FILTER_NUM
print('filter_num: {}'.format(filter_num))

def read_dict_from_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print("{} successfully loaded!".format(path))
    return data

# label_path = pth.join('tweet_id2label.json')
label_path = pth.join('../datasets/{}/raw_data/label_dict.json'.format(DATASET_NAME))
label_dict = read_dict_from_json(label_path)
print(label_dict)

df = pd.read_csv('comments_text_encode1.csv', encoding='utf-8')
add_list = []
for i, line in df.iterrows():
    tree_id = line['tree_id']
    tweet_id = line['tweet_id']
    node_id = tree_id.split('_')[0]
    tree_num = tree_id.split('_')[1]
    if tree_num == '0':
        print(tweet_id)
        lb = label_dict[str(tweet_id)]
    else:
        lb = '9'
    add_list.append(lb)
print(add_list)

df['label'] = add_list
df.reset_index()
df.to_csv('comments_text_encode3.csv', encoding='utf-8', line_terminator='\n')
df = pd.read_csv('filtered_user_profile{}_encode1.csv'.format(str(filter_num)), encoding='utf-8')

length = df.shape[0]
df['node_id'] = list(range(SOURCE_TWEET_NUM, SOURCE_TWEET_NUM+length))
df = df[['node_id', 'user_id', 'created_time', 'statuses_count',
         'favourites_count', 'listed_count', 'followers_count',
         'friends_count', 'text', 'year', 'month', 'day', 'hour']]
df.reset_index()
df.to_csv('filtered_user_profile{}_encode2.csv'.format(str(filter_num)), encoding='utf-8', line_terminator='\n')