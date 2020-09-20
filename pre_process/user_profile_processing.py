"""
purposes:
1. prepare for user profile embedding
There are two output of this .py: 1. all user in a dir of csv; 2. filtered user to one csv
"""

import csv
import os
import os.path as pth
import argparse
import pandas as pd
from shutil import copyfile
from pre_process.sentence_clean import SentenceClean
from utils import user_profile_path, processed_data_dir, filtered_dataframe
from tools import _convert_time
from tqdm import tqdm
from time import sleep

parser = argparse.ArgumentParser(description='input filter number')
parser.add_argument('--filter', type=int, help='filter number', default=4)
args = parser.parse_args()
filter_num = args.filter

filtered_user_list = filtered_dataframe['user_id'].tolist()
user_profile_dir = pth.join('../datasets/twitter16/raw_data/user_profile')   # todo: right path
filtered_user_profile_path = pth.join('../load_data16/filtered_user_profile.csv')

if not pth.isdir('../datasets/twitter16/processed_data/processed_user_profile'):
    os.mkdir(pth.join('../datasets/twitter16/processed_data/processed_user_profile'))


def get_text(sentence) -> "str ('' if None)":
    if sentence is not None:
        sentence = SentenceClean(sentence, method='regex').sentence_cleaner()
        return sentence
    else:
        return ''


filtered_df = pd.DataFrame(columns=[["user_id", "created_time", "geo_enabled","location", "statuses_count",
                                     "favourites_count", "listed_count", "followers_count", "friends_count", "text"]])
file_counter = 0
record_count = 0
output_list = []
with open(pth.join(filtered_user_profile_path), 'w', newline='', encoding='utf-8') as p1:
    # this is all filtered_user_profile(.csv)
    filtered_write = csv.writer(p1)
    filtered_write.writerow(
        ["user_id", "created_time", "geo_enabled", "location", "statuses_count", "favourites_count",
         "listed_count", "followers_count", "friends_count", "text"]
    )
    for file in tqdm(os.listdir(user_profile_dir)):
        file_counter += 1
        tweet_id = file.split('.')[0]
        file_path = pth.join(user_profile_dir, file)
        df = pd.read_csv(file_path, encoding='utf-8', lineterminator='\n')  # get a csv
        # print('df.shape[0]', df.shape[0])
        # sleep(10000)
        df = df.where(df.notnull(), None)
        df = df.rename(columns={"id": "user_id", "description": "text"})  # rename column names
        # df["created _at"] = df["created _at"].apply(lambda x: _convert_time(x))  # 2020-8-16-18
        # df["geo_enabled"] = df["geo_enabled"].apply(lambda x: int(x))  # True: 1, False:0
        # df["location"] = df["location"].apply(lambda x: None if x is None else 1)  # None:0 , exists: 1
        # df["description"] = df["description"].apply(lambda x: get_text(x))  # None:0 , exists: 1
        # processed_df = df
        # processed_save_path = pth.join(utils.processed_data_dir, 'processed_user_profile', file)
        # processed_df.to_csv(processed_save_path, line_terminator='\n', encoding='utf-8')
        #
        # filter_df = df
        # filter_df = filter_df[filter_df['user_id'].astype(str) in filtered_user_list]

        processed_save_path = pth.join(processed_data_dir, 'processed_user_profile', file)
        with open(processed_save_path, 'w', newline='', encoding='utf-8') as pw:
            processed_write = csv.writer(pw)
            processed_write.writerow(
                ["user_id", "created_time", "geo_enabled", "location", "statuses_count", "favourites_count",
                 "listed_count", "followers_count", "friends_count", "text"]
            )

            for index, row in df.iterrows():
                # print(row['id'])

                user_id = row['user_id']
                created_time = _convert_time(row['created _at'])
                geo_enabled = 1 if row['geo_enabled'] is not None or False else 0
                # geo_enabled = int(row['geo_enabled'])
                location = 0 if row['location'] is None else 1
                statuses_count = 0 if row['statuses_count'] is None else int(row['statuses_count'])
                favourites_count = 0 if row['favourites_count'] is None else int(row['favourites_count'])
                listed_count = 0 if row['listed_count'] is None else int(row['listed_count'])
                followers_count = 0 if row['followers_count'] is None else int(row['followers_count'])
                friends_count = 0 if row['friends_count'] is None else int(row['friends_count'])
                # favourites_count = row['favourites_count']
                # listed_count = row['listed_count']
                # followers_count = row['followers_count']
                # if row['friends_count'].isnull() or row['friends_count'] is None:
                #     print('row', row)
                #     raise ValueError('Error')
                # friends_count = row['friends_count']
                text = get_text(row['text'])
                # todo: add is_verified

                processed_write.writerow(
                    [user_id, created_time, geo_enabled, location, statuses_count, favourites_count,
                     listed_count, followers_count, friends_count, text]
                )

                # get filtered user word, process the file
                if str(user_id) not in filtered_user_list or str(user_id) in output_list:
                    continue
                else:
                    record_count += 1
                    output_list.append(str(user_id))
                    filtered_write.writerow(
                        [user_id, created_time, geo_enabled, location, statuses_count, favourites_count,
                         listed_count, followers_count, friends_count, text]
                    )

    print("record_count: {}".format(record_count))
    print("len(output_list) = {}".format(len(output_list)))

    if record_count == len(output_list):
        f = open('filtered_user_id_record15.txt', 'w')
        f.write(str(output_list))
        f.close
    else:
        raise Exception("record_count:{} != len(output): {}, Error!".format(record_count, len(output_list)))

print("--user processing finished!--")


