import os.path as pth
import pandas as pd
from collections import defaultdict
from pre_process.tools import read_dict_from_json, txt2iterable, save_dict_to_json
from pre_process.utils import filtered_dataframe, tweet2mat

filter_num = 5

def add_edge_index(arr, a, b) -> "node_dict; a:b, b:a":
    """arr = {}"""
    a = int(a)
    b = int(b)
    arr[a].append(b)
    arr[b].append(a)
    return arr


tweet_user_dict = read_dict_from_json(pth.join('../load_data16/tweet_user_dict.json'))
user_tweet_dict = read_dict_from_json(pth.join('../load_data16/user_tweet_dict.json'))
delete_tweet_list = txt2iterable(pth.join('../datasets/twitter16/auxiliary_data/delete_tweet_list.txt'))
delete_user_list = txt2iterable(pth.join('../datasets/twitter16/auxiliary_data/delete_user_list.txt'))

for tweet in delete_tweet_list:
    if tweet in tweet_user_dict:
        del tweet_user_dict[tweet]
assert len(tweet_user_dict) == 790

for user in delete_user_list:
    if user in user_tweet_dict:
        del user_tweet_dict[user]


tweet_path = pth.join('../load_data16/processed_source_tweet.csv')
user_path = pth.join('../load_data16/filtered_user_profile_encode1.csv')
tweet_df = pd.read_csv(tweet_path)
user_df = pd.read_csv(user_path)

tweet_list = tweet_df['tweet_id'].astype(str).tolist()
user_list = user_df['user_id'].astype(str).tolist()
tweet_num = len(tweet_list)
user_num = len(user_list)
tweet2node_dict = dict(zip(tweet_list, [str(i) for i in list(range(tweet_num))]))
user2node_dict = dict(zip(user_list,  [str(i) for i in list(range(tweet_num, tweet_num+user_num))]))


tweet_num = len(tweet2node_dict)
max_tweet_index = tweet_num - 1
number_list = filtered_dataframe['appear in dataset'].tolist()
total_connection = sum(number_list)
user_num = len(number_list)
print("total_connection: {}".format(total_connection))
print("user_num: {}".format(user_num))

connection_counter = 0
record_num = 0
edge_idx = defaultdict(list)
record_counter = 0
user2mat = dict()


for index, row in filtered_dataframe.iterrows():
    record_num += 1
    user_mat_idx = max_tweet_index + record_num  # this is the user matrix_index
    user2mat[str(user_mat_idx)] = str(row['user_id'])
    # print(user_mat_idx)
    user_appearance = row['appear in dataset']
    # print("user_appearance = {}".format(user_appearance))
    connection_counter += user_appearance
    tweet_record = eval(row['record'])
    tweet_record_list = list(tweet_record.keys())
    # mat_id of user & tweet are both 'int'

    for tweet in tweet_record:
        try:
            # user_mat_idx = tweet_num + record_num
            tweet_mat_idx = int(tweet2mat(tweet))
            edge_idx = add_edge_index(edge_idx, user_mat_idx, tweet_mat_idx)
            record_counter += 1
        except Exception:
            print(tweet)
            raise ValueError("tweet_id not found")


if __name__ == '__main__':
    print(len(edge_idx))
    # save_path = '../load_data15/tw15_connections.json'
    save_path = '../load_data16/tw16_connections{}.json'.format(str(filter_num))
    save_dict_to_json(edge_idx, save_path)


