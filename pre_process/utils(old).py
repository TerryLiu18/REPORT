import os
import os.path as pth
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import tools
import argparse
from time import sleep

parser = argparse.ArgumentParser(description='choose dataset: "twitter15" or "covid-19"')
parser.add_argument('--dataset', type=str, help='choose dataset folder', default='twitter16')
parser.add_argument('--filter', type=int, help='filter users whose appearance < n in graph', default=4)
args = parser.parse_args()
dataset_name = args.dataset
filter_num = args.filter

# -------------------------------------- define path name ----------------------------------------------------
project_name = 'MyGAT'
loc = os.path.abspath(os.path.dirname(__file__)).split(project_name)[0]
root_path = pth.join(loc, project_name)
print("root_path: {}".format(root_path))

dataset_path = pth.join(root_path, 'datasets', dataset_name)
raw_data_dir = pth.join(dataset_path, 'raw_data')
auxiliary_data_dir = pth.join(dataset_path, 'auxiliary_data')
processed_data_dir = pth.join(dataset_path, 'processed_data')

if not pth.exists(raw_data_dir): os.makedirs(raw_data_dir)
if not pth.exists(auxiliary_data_dir): os.makedirs(auxiliary_data_dir)
if not pth.exists(processed_data_dir): os.makedirs(processed_data_dir)

user_profile_path = pth.join('../datasets/{}/raw_data/user_profile'.format(dataset_name))
tweet_tree_path = pth.join('../datasets/{}/raw_data/tweet_tree'.format(dataset_name))
tweet_content_path = pth.join('../datasets/{}/raw_data/tweet_profile'.format(dataset_name))

if not pth.exists(user_profile_path) and pth.exists(tweet_content_path) and pth.exists(tweet_tree_path):
    raise Exception("{}".format(user_profile_path))

# -------------------------------------- get auxiliary files ----------------------------------------------------

tweet2node_dict = dict()
count = 0
tweet2node_dict_path = pth.join('../load_data16/tweet2node_dict.json')

if not pth.exists(tweet2node_dict_path):
    for file in os.listdir(user_profile_path):
        tweet_id = file.split('.')[0]
        tweet2node_dict[tweet_id] = count
        count += 1
    # tools.iterable2txt(tweet2node_dict, tweet2node_dict_path)
    tools.save_dict_to_json(tweet2node_dict, tweet2node_dict_path)
    print("write: {}".format(tweet2node_dict_path))
else:
    print("{} already exists".format(tweet2node_dict_path))
    # tweet2node_dict = tools.txt2iterable(tweet2node_dict_path)
    tweet2node_dict = tools.read_dict_from_json(tweet2node_dict_path)
print("Transformation between tweet_id and node_index done")

# get tweet_user_dict, user_tweet_dict
csv_path = pth.join('../datasets/{}/raw_data/all_user_info.csv'.format(dataset_name))
df = pd.read_csv(csv_path, lineterminator='\n')
user_id_list = df['user_id'].astype(str).tolist()
user_appear_record = df["record\r"].tolist()
user_appear_record = [list(eval(i).keys()) for i in user_appear_record]
print('len(user_appear_record)', len(user_appear_record))
tweet_user_dict_path = pth.join(root_path, 'load_data16',  "tweet_user_dict.json")
user_tweet_dict_path = pth.join(root_path, 'load_data16', "user_tweet_dict.json")


if pth.isfile(tweet_user_dict_path) and pth.isfile(user_tweet_dict_path):
    print(tweet_user_dict_path)
    print(user_tweet_dict_path)
    user_tweet_dict = tools.read_dict_from_json(user_tweet_dict_path)
    tweet_user_dict = tools.read_dict_from_json(tweet_user_dict_path)
else:
    tweet_user_dict = defaultdict(list)
    user_tweet_dict = dict(zip(user_id_list, user_appear_record))
    for user, tweet_list in tqdm(user_tweet_dict.items()):
        for tweet in tweet_list:
            tweet_user_dict.setdefault(tweet, []).append(user)
    tools.save_dict_to_json(tweet_user_dict, tweet_user_dict_path)
    tools.save_dict_to_json(user_tweet_dict, user_tweet_dict_path)


# get maximum_appear_dict
tweet_maximum_appear_dict = dict()
maximum_appear_dict_path = pth.join('../datasets/twitter16/auxiliary_data/maximum_appear_dict.txt')
maximum_appear_dict_path2 = pth.join('../datasets/twitter16/auxiliary_data/maximum_appear_dict.json')
if pth.isfile(maximum_appear_dict_path):
    tweet_maximum_appear_dict = tools.txt2iterable(maximum_appear_dict_path)  # todo: maximum_appear_dict is wrong!
else:
    print("write maximum_appear_dict: {}".format(maximum_appear_dict_path))
    for tweet, user_list in tqdm(tweet_user_dict.items()):
        minimum_filter = 1
        # tweet_mat_idx = tweet2mat_idx[tweet]
        tweet_maximum_appear_dict[tweet] = minimum_filter
        for user in user_list:
            if user.lower() == 'root':
                continue
            length = len(user_tweet_dict[user])
            if length > minimum_filter:
                minimum_filter = len(user_tweet_dict[user])
        tweet_maximum_appear_dict[tweet] = minimum_filter
    tools.save_dict_to_json(tweet_maximum_appear_dict, maximum_appear_dict_path2)


def tweet2mat(tweet_id):
    """
    :param tweet_id:
    :return: matrix_id of tweet (0-1311)
    """
    if tweet_id in tweet2node_dict:
        return tweet2node_dict[str(tweet_id)]
    else:
        raise Exception("{} not found".format(tweet_id))


# ---------------------------------------------- DataFilter class ----------------------------------------------------

class DataFilter:
    """this is the verified input files sets"""

    def __init__(self):
        self.__user_info_path = pth.join(raw_data_dir, "all_user_info.csv")
        self.__users_profile_dir = pth.join(raw_data_dir, "user_profile")
        self.__tweet_tree_dir = pth.join(raw_data_dir, "tweet_tree")
        self.__tweet_profile_dir = pth.join(raw_data_dir, "tweet_profile")
        self.__filtered_user_info_path = pth.join(auxiliary_data_dir, 'filtered_user_profile.csv')
        self.__preprocess_dir = processed_data_dir
        self.__pretrained_twitter_dataset_dir = pth.join(root_path, 'pretrained_dataset')

    @property
    def user_info_path(self):
        return self.__user_info_path

    @property
    def user_profile_dir(self):
        return self.__users_profile_dir

    @property
    def tweet_tree_dir(self):
        return self.__tweet_tree_dir

    @property
    def tweet_profile_dir(self):
        return self.__tweet_profile_dir

    @property
    def filtered_user_info_path(self):
        return self.__filtered_user_info_path

    @property
    def preprocess_dir(self):
        return self.__preprocess_dir

    @property
    def pretrained_twitter_dataset_dir(self):
        return self.__pretrained_twitter_dataset_dir

    def filter_user(self, f_num):
        """
        filter users whose appearance < n
        write file to appear{n}_user_info
        :input: n
        :return: filtered_DataFrame
        """
        print("only keep users whose appearance >= n")
        file_name = "appear" + str(f_num) + "_user_info.csv"
        file_path = pth.join(auxiliary_data_dir, file_name)
        if pth.isfile(file_path):
            print("{} already exists".format(file_path))
            file = pd.read_csv(file_path)
            filtered_df = pd.DataFrame(file)
            filtered_df['user_id'] = filtered_df['user_id'].astype(str)
            return filtered_df
        else:
            print("create new filtered file: {}".format(file_path))
            file = pd.read_csv(self.user_info_path)
            content = pd.DataFrame(file)
            isolated_tweet = list()
            add_user_set = set()
            for tweet, appear_time in tweet_maximum_appear_dict.items():
                isolated_tweet.append(tweet)
                if appear_time < f_num:
                    for user in tweet_user_dict[tweet]:
                        if len(user_tweet_dict[user]) == appear_time:
                            add_user_set.add(user)

            filtered_df = pd.DataFrame(columns=["user_id", "appear in dataset", "record"])
            write_row = 0
            for i, row in tqdm(content.iterrows()):
                if row['appear in dataset'] < f_num and str(row['user_id']) not in add_user_set:
                    continue
                else: 
                    write_row += 1
                    filtered_df = filtered_df.append([row])
                    if write_row % 1000 == 0:
                        print("----------user number: {} ---------".format(write_row))
            # filtered_df['id'] = filtered_df['id'].apply(lambda x: str(x))
            filtered_df['user_id'] = filtered_df['user_id'].astype(str)
            filtered_df.to_csv(file_path)
            return filtered_df

    def __str__(self):
        print("filter users whose appearance < n, write file to appear{n}_user_info, return df")


input_raw_data = DataFilter()
filtered_dataframe = input_raw_data.filter_user(filter_num)
print('filtered_dataframe.size', filtered_dataframe.shape[0])


if __name__ == "__main__":
    cwd = os.getcwd()
    print("load utils success")