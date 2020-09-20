import os
import csv
import os.path as pth
import pandas as pd
from tqdm import tqdm
from time import sleep
import tree_checker
import tools

cwd = os.getcwd()
input_file_dir = pth.join(cwd, 'graph_tree')
all_users_info = pth.join(cwd, "all_users_info.csv")
appearall_users_info = pth.join(cwd, "all_user_info.csv")


if pth.exists(input_file_dir):
    print("Path is TRUE")

thread_user_dictlist = {}
thread_user_dictset = {}
all_users_list = []
all_users_set = set()

print(len(os.listdir(input_file_dir)))
files = os.listdir(input_file_dir)

user_no = 0
for file in tqdm(os.listdir(input_file_dir)):
    tweet_id = file.split('.')[0]
    thread_user_dictlist[tweet_id] = []
    thread_user_dictset[tweet_id] = set()
    with open(pth.join(input_file_dir, file), 'r') as fr:
        lines = fr.readlines()
        row_number = len(lines)
        for line in lines:
            two_list = line.split('->')
            tweet_user1 = eval(two_list[0])[0]
            thread_user_dictlist[tweet_id].append(tweet_user1)
            tweet_user2 = eval(two_list[1])[0]
            thread_user_dictlist[tweet_id].append(tweet_user2)

        thread_user_dictset[tweet_id] = set(thread_user_dictlist[tweet_id])
        all_users_list.extend(thread_user_dictlist[tweet_id])
        # print("a file has completed")
all_users_set = set(all_users_list)
print("all files have counted, {} users in total".format(len(all_users_set)))


all_users_info = []
num = 0

if not pth.exists(appearall_users_info):
    with open(appearall_users_info, 'w', newline='') as fall:
        fallr = csv.writer(fall, delimiter=',', lineterminator='\n')
        fallr.writerow(['user_id', 'appear in dataset', 'record'])

        for user in all_users_set:
            num += 1
            user_info = dict()
            user_info['user_id'] = user
            user_info['appear in dataset'] = 0
            user_info['record'] = {}
            # all_users_info[num] = user_info

            for tweet in thread_user_dictlist:
                # print(tweet)
                if user in thread_user_dictset[tweet]:
                    user_info['record'][tweet] = thread_user_dictlist[tweet].count(user)
                    user_info['appear in dataset'] += 1

            if num % 10000 == 0:
                print("This is the No.{}: {}".format(num, user_info))
            all_users_info.append(user_info)
            list_info = [user_info['user_id'], user_info['appear in dataset'], user_info['record']]
            fallr.writerow(list_info)

print("---Operation finish!---")

source_tweet_list = tree_checker.source_tweet_list
assert len(thread_user_dictset) == 802

df = pd.read_csv(appearall_users_info, lineterminator='\n')
df['user_id'] = df['user_id'].astype(str)
df = df.set_index("user_id", drop=True)

# print(df.loc['246930965', 'appear in dataset'])

delete_tweet_list = []

for tweet in thread_user_dictset:
    connect_flag = 0
    for user in thread_user_dictset[tweet]:
        appear_time = df.loc[user, 'appear in dataset']
        if appear_time > 1:
            connect_flag = 1
            continue
    if not connect_flag:
        delete_tweet_list.append(tweet)

tools.iterable2txt(delete_tweet_list, pth.join('../auxiliary_data/delete_tweet_list.txt'))

drop_cnt = 0
for tweet in delete_tweet_list:
    for user in thread_user_dictset[tweet]:
        df = df.drop(user)
        drop_cnt += 1
        print('drop: {}'.format(user))
print("drop_cnt", drop_cnt)
df.to_csv(appearall_users_info, line_terminator='\n')
