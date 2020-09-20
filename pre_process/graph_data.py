import tools
from collections import defaultdict
import numpy as np
import os.path as pth
from utils import filtered_dataframe, tweet2node_dict, tweet2mat


def add_edge_index(arr, a, b): 
    """arr = {}"""
    arr.setdefault(a, []).append(b)
    arr.setdefault(b, []).append(a)
    return arr


def add_edge_index(arr, a, b):
    """arr = {}"""
    arr.append([a, b])
    arr.append([b, a])
    return arr


TWEET_NUM = len(tweet2node_dict)
MAX_TWEET_INDEX = TWEET_NUM - 1
NUMBER_LIST = filtered_dataframe['appear in dataset'].tolist()
total_connection = sum(NUMBER_LIST)
user_num = len(NUMBER_LIST)
print("total_connection: {}".format(total_connection))
print("user_num: {}".format(user_num))


edge_idx = dict()
user2mat = dict()

connection_counter = 0
record_counter = 0
record_num = 0

filtered_user_tweet_dict = dict()
filtered_tweet_user_dict = defaultdict(list)

for index, row in filtered_dataframe.iterrows():
    record_num += 1
    user_mat_idx = MAX_TWEET_INDEX + record_num  # this is the user matrix_index
    user2mat[str(user_mat_idx)] = str(row['user_id'])
    # print(user_mat_idx)
    user_appearance = row['appear in dataset']
    # print("user_appearance = {}".format(user_appearance))
    connection_counter += user_appearance
    tweet_record = eval(row['record'])
    tweet_record_list = list(tweet_record.keys())
    filtered_user_tweet_dict[int(user_mat_idx)] = [int(tweet) for tweet in tweet_record_list]
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

for user, tweet_list in filtered_user_tweet_dict.items():
    for tweet in tweet_list:
        filtered_tweet_user_dict.setdefault(tweet, []).append(user)

if record_num != user_num:
    print("record_num: {}".format(record_num))
    print("user_num: {}".format(user_num))
    raise Exception("record_num != user_num")


def find_t(l):
    if not l:
        raise Exception("no such connection []")
    if l[0] <= MAX_TWEET_INDEX < l[1]:
        return l[0]
    elif l[1] <= MAX_TWEET_INDEX < l[0]:
        return l[1]
    else:
        print("This is the wrong connection: {}".format(l))
        raise Exception("no such connection")



def check_edge_connection(edge_idx):
    """to make sure no isolation node"""
    tweet_set = set([find_t(connect) for connect in edge_idx])
    tweet_list = list(tweet_set)
    tweet_list.sort()

    if tweet_list == list(range(MAX_TWEET_INDEX+1)):
        print("No isolated connections! Build graph success")
        return True
    else:
        i = 0
        isolated_node = []
        for _ in tweet_list:
            if _ != i:
                isolated_node.append(_)
            i += 1
        raise Exception(isolated_node)


if __name__ == '__main__':
    print(len(edge_idx))
    if check_edge_connection(edge_idx):
        # f = open(pth.join(verified, 'edge_idx.txt'), 'w')
        # a = f.write(str(edge_idx))
        # f.close()
        # print(pth.join(verified, 'edge_idx.txt'))
        # edge_index = array2mat(edge_idx)
        save_path = '../load_data/tw15_connections.txt'
        save_path2 = '../load_data/tw15_connections.json'
        tools.iterable2txt(edge_idx, pth.join(save_path2))
        # with open(save_path, 'w') as f:
        #     f.write(np.array2string(edge_index, separator=' '))

#     #         np.add
#     #         .add_edge(int(user_mat_id), int(tweet_dict[tweet]))
#     #         connection += 1
#     #         print("Connection between {} and {} added, this is the {} connections"\
#     #                   .format(user_mat_id, tweet_dict[tweet], connection))
#     # print("{} connections has been added totally".format(connection))
# print(connection_counter)
