import os
import json
import os.path as pth
import pandas as pd
from tqdm import tqdm
from task import SOURCE_TWEET_NUM, FILTER_NUM
from tools import txt2iterable

filter_num = FILTER_NUM
print('filter_num: {}'.format(filter_num))

def read_dict_from_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print("{} successfully loaded!".format(path))
    return data


user_appearance_record = pth.join('../datasets/twitter15/raw_data/all_user_info.csv')
user_appear_df = pd.read_csv(user_appearance_record, lineterminator='\n')
user_appear_record = user_appear_df['user_id'].values.tolist()
user_appear_list = [str(i).split('.')[0] for i in user_appear_record]
user_appear_df = user_appear_df.set_index('user_id')
print(user_appear_df.head(10))

tweetid2userid_dict = read_dict_from_json('tweetid2userid_dict.json')
print(len(tweetid2userid_dict))

# label_path = pth.join('tweet_id2label.json')
label_path = pth.join('../datasets/twitter15/raw_data/label_dict.json')
label_dict = read_dict_from_json(label_path)
print(label_dict)

train_indices = txt2iterable(pth.join('train_indices.txt'))
test_indices = txt2iterable(pth.join('test_indices.txt'))
train_indices = [int(i) for i in train_indices]
test_indices = [int(i) for i in test_indices]

def user_tendency_checker(uid):
    """
    labels    labels_num  fake_score
    non-rumor     0           0
    false         1           10
    true          2           -1
    unverified    3           0
    """

    uid = str(uid).split('.')[0]
    if uid not in user_appear_list:
        return '---', '{-----banned-----}'
    fake_score_dict = {'0': 0, '1': 10, '2': -1, '3': 0}
    user_record = eval(user_appear_df.loc[int(uid)]['record'])
    total_appear_time = user_appear_df.loc[int(uid)]['appear in dataset']
    if int(total_appear_time) < FILTER_NUM:
        return '---', '{---only 1 connect---}'
    fake_score = 0
    for source_id, appear_time in user_record.items():
        label = str(label_dict[source_id])
        fake_score = fake_score + fake_score_dict[label]
    return fake_score, str(user_record)



df = pd.read_csv('../load_data15/comments_text.csv', encoding='utf-8')
add_list = []
user_list = []
fake_score_list = []
appear_record_list = []
type_list = []

for i, line in tqdm(df.iterrows()):
    tree_id = line['tree_id']
    tweet_id = line['tweet_id']
    node_id = tree_id.split('_')[0]
    tree_num = tree_id.split('_')[1]
    user_id = tweetid2userid_dict[str(tweet_id)]
    type = None
    if int(tree_num) in train_indices:
        type = 'train'
    elif int(tree_num) in test_indices:
        type = 'test'
    else:
        raise ValueError('tree num', tree_num)
    type_list.append(type)

    if tree_num == '0':
        lb = label_dict[str(tweet_id)]
        tree_label = lb
    else:
        lb = '_'+str(tree_label)
    add_list.append(lb)
    user_list.append(user_id)
    fake_score, appear_record = user_tendency_checker(user_id)
    fake_score_list.append(fake_score)
    appear_record_list.append(appear_record)


print(add_list)

df['label'] = add_list
df['user_id'] = user_list
df['fake_score'] = fake_score_list
df['appear_record'] = appear_record_list
df['type'] = type_list
df.reset_index()
order = ['tree_id', 'type', 'label', 'tweet_id', 'user_id', 'fake_score', 'appear_record', 'text']
df = df[order]
df.to_csv('comments_text_with_label2.csv', encoding='utf-8', line_terminator='\n')
# df = pd.read_csv('filtered_user_profile{}_encode1.csv'.format(str(filter_num)), encoding='utf-8')

# length = df.shape[0]
# df['node_id'] = list(range(SOURCE_TWEET_NUM, SOURCE_TWEET_NUM+length))
# df = df[['node_id', 'user_id', 'created_time', 'statuses_count',
#          'favourites_count', 'listed_count', 'followers_count',
#          'friends_count', 'text', 'year', 'month', 'day', 'hour']]
# df.reset_index()
# df.to_csv('filtered_user_profile{}_encode2.csv'.format(str(filter_num)), encoding='utf-8', line_terminator='\n')