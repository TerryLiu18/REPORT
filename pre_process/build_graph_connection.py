import os.path as pth
import tools
from collections import OrderedDict, defaultdict
import pandas as pd
from utils import filtered_dataframe, tweet2node_dict, tweet2mat


def add_edge_index(arr, a, b) -> "node_dict; a:b, b:a":
    """arr = {}"""
    a = int(a)
    b = int(b)
    arr[a].append(b)
    arr[b].append(a)
    return arr


tweet_user_dict = tools.read_dict_from_json(pth.join('../load_data16/tweet_user_dict.json'))
user_tweet_dict = tools.read_dict_from_json(pth.join('../load_data16/user_tweet_dict.json'))

tweet_path = '../load_data16/processed_source_tweet.csv'
user_path = '../load_data16/filtered_user_profile_encode1.csv'

content1 = pd.read_csv(pth.join(tweet_path))
content2 = pd.read_csv(pth.join(user_path))
tweet_df = pd.DataFrame(content1)
user_df = pd.DataFrame(content2)
tweet_list = tweet_df['tweet_id'].astype(str).tolist()
user_list = user_df['user_id'].astype(str).tolist()
tweet_num = len(tweet_list)
user_num = len(user_list)
tweet2node_dict = dict(zip(tweet_list, [str(i) for i in list(range(tweet_num))]))
user2node_dict = dict(zip(user_list,  [str(i) for i in list(range(tweet_num, tweet_num+user_num))]))


tweet_num = len(tweet2node_dict)
max_tweet_index = tweet_num - 1  # 1309/681
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
            # print("user_mat_idx: {}".format(user_mat_idx))
            # print("tweet_mat_idx: {}".format(tweet_mat_idx))
            raise Exception("tweet_id not found")


if __name__ == '__main__':
    print(len(edge_idx))
    # save_path = '../load_data15/tw15_connections.json'
    save_path = '../load_data16/tw16_connections.json'
    tools.save_dict_to_json(edge_idx, save_path)


