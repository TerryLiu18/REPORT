"""
return source_tweet_info.csv
columns[matrix_idx, source_tweet_id, source_user_id, link, text, label]
"""

import pandas as pd
import os
import os.path as pth
import re
import csv
import json
from ast import literal_eval
from time import sleep
# from check_verified import verified_tweet_content_dict
from tqdm import tqdm

cwd = os.getcwd()
tree_file_path = pth.join(cwd, 'tweet_tree')
file_list = os.listdir(pth.join(cwd, 'tweet_tree'))
output = pth.join(cwd, 'source_tweet_info.csv')


def get_label(lb):
    labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']
    if lb in labelset_nonR:
        return 0
    elif lb in labelset_f:
        return 1
    elif lb in labelset_t:
        return 2
    elif lb in labelset_u:
        return 3
    else:
        raise Exception('label not found: {}'.format(lb))


def read_dict_from_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # print("{} successfully loaded!".format(path))
    return data


def save_dict_to_json(item, path, overwrite=True):
    if os.path.exists(path) and overwrite is False:
        print("{} already exists".format(path))
    else:
        try:
            item = json.dumps(item, indent=4)
            with open(path, "w", encoding='utf-8') as f:
                f.write(item)
        except Exception as e:
            print("write error==>", e)


def get_url(input_text):
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', input_text)
    if urls:
        return urls[0]
    else:
        return None


tweet2label = read_dict_from_json('label_dict.json')
tweet2matid = read_dict_from_json('tweet2matrix_idx.json')
tweet2text = dict()

with open('source_tweets.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.rstrip()
        info = line.split('\t')
        tweet_id, text = info[0], info[1]
        tweet2text[tweet_id] = text


with open(output, 'w', encoding='utf-8', newline='') as csv_write: #write to source_tweet_info.csv
    csv_w = csv.writer(csv_write)
# csv.writerow(['user_id','tweet_id','link', 'record_number', 'label'])
    csv_w.writerow(["matrix_idx", "source_tweet_id", "source_user_id", "text", "link",
                    "label", "record_number", "user_number", "unique_tweet_number"])
    count = 0
    for file in file_list:
        count += 1
        source_id = file.split('.')[0]
        # print(tweet_id)
        with open(pth.join(tree_file_path, file), 'r', encoding='utf-8') as fr:
            # file_path = pth.join(pth.join(cwd, 'verified_tree_twitter15'))
            content_lines = fr.read()
            record_number = content_lines.count("->")
            content_lines = content_lines.rstrip("over")
            content_lines = content_lines.replace("\n","->")
            content_lines = content_lines.rstrip("-->")
            # print("this is content_lines: {}".format(content_lines))
            content_list = content_lines.split("->")
            # print("this is content_list: {}".format(content_list))
            record_list = []
            for record in content_list:
                record = record.strip()
                record_list.append(literal_eval(record))
            # print(len(record_list))

            #source record
            # source_record = record_list[1]  # todo user record_list[1] in twitter15
            source_record = record_list[0]
            tweet_id = source_record[1]
            user_id = source_record[0]
            if tweet_id != source_id:
                print(source_id)
                # raise Exception("{} go wrong!".format(source_id))
            matrix_idx = tweet2matid[source_id]
            tweet_label = tweet2label[source_id]
            text = tweet2text[source_id]
            link = r"https://twitter.com/" + str(user_id) + "/status/" + str(tweet_id) + '\t'
            # text = verified_tweet_content_dict[tweet_id].replace("\n", "\t")

            # count user number
            thread_user_list = []
            thread_tweet_list = []
            for record in record_list[1:]:
                uni_user_id = record[0]
                uni_tweet_id = record[1]
                thread_user_list.append(uni_user_id)
                thread_tweet_list.append(uni_tweet_id)
                user_number = len(set(thread_user_list))
                unique_tweet_number = len(set(thread_tweet_list))
            csv_w.writerow([matrix_idx, source_id, user_id, text, link, tweet_label,
                            record_number, user_number, unique_tweet_number])

print("-------finish {} files------".format(count))