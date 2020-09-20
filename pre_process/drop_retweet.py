"""
this file aims to eliminate re-tweets in twitter15 trees
file output: 1. processed_data/proccessed_tweet_tree
             2. tweet2tree_id (dictionary)
file form:
[tweet_id, user_id, text(string)] -> [tweet_id, user_id, text(string)]
"""

import os
import os.path as pth
import tools
import pandas as pd
from tqdm import tqdm
from utils import raw_data_dir, processed_data_dir, tweet2node_dict
from pre_process import tree_checker
from time import sleep

# input_tree_structure_dir = pth.join(raw_data_dir, "tweet_tree")
input_tree_content_dir = pth.join(processed_data_dir, "processed_tweet_tree_profile")

output_dir = pth.join(processed_data_dir, 'processed_tweet_tree')
output_tree_structure_dir = pth.join(processed_data_dir, 'processed_tweet_tree')

tweet2tree_id_path = pth.join(processed_data_dir, 'tweet2tree_id_dict.txt')
tweet2tree_id_dict = dict()   # this is the dict between tree_id

# if not pth.isdir('../datasets/twitter16/processed_data/processed_tweet_tree_profile'):
#     os.mkdir(pth.join('../datasets/twitter16/processed_data/processed_tweet_tree_profile'))

"""
user_id, tweet_id, time -> ...
['ROOT', 'ROOT', '0.0']->['139255910', '356268980211687424', '0.0']
['139255910', '356268980211687424', '0.0']->['441994716', '356268980211687424', '6.7']
['441994716', '356268980211687424', '6.7']->['222615066', '356268980211687424', '6.7']
['139255910', '356268980211687424', '0.0']->['44748394', '356268980211687424', '6.7']

,tweet_id,user_id,text,retweet_count,favorite_count
0,273185394944794625,822368798,i was one of those hurt ! i just came home from hospital .,0,0
1,273470833022865408,631634039,,0,0
2,273220001492762624,538417111,"oh , if only .",0,0
3,273224999287025664,129145854,is this story true or faking news ? ?,0,0
4,273186640015544320,376427402,"now that is real darwinism , rt 42 mil dead in bloodiest black friday weekend on record <url>",0,0
"""


def _get_tweet_record(tweet_id, df, recorded_list):
    if tweet_id not in recorded_list:
        raise Exception("tweet_id not in recorded_list")
    idx = df[df['tweet_id'].astype(str) == str(tweet_id)].index.tolist()
    # df_new = df.set_index('tweet_id', drop=True, append=False, inplace=False, verify_integrity=False)

    if len(idx) != 1:
        raise Exception("Error! index:{}, tweet_id:{}, recorded_list: {}"
                        .format(idx, tweet_id, recorded_list))
    idx = idx[0]
    u_id, tex = df.loc[idx, 'tweet_id'], df.loc[idx, 'text']
    return str(u_id), str(tex)


counter = 0
abnormal_count = 0
abnormal_list = []
input_tree_structure_dir = pth.join(raw_data_dir, "tree_tree")
f_list = os.listdir(input_tree_structure_dir)

for file in tqdm(f_list):
    file_tweet_id = file.split('.')[0]
    content = pd.read_csv(pth.join(input_tree_content_dir, file_tweet_id+'.csv'), lineterminator='\n', encoding='utf-8')
    df = pd.DataFrame(content)
    recorded_comments_list = df['tweet_id'].astype(str).tolist()  # these are the tweets whose texts are unique and can be found
    node_id = tweet2node_dict[file_tweet_id]  # node_id: from 0 to 1311
                                              # content {node_id:[node_id1, node_id2]...}
    counter += 1
    record_dict = dict()
    write_tweet_list = list()
    # this is used to record whether a tweet is taken (each comments shall appear in his tree once and once only)

    with open(pth.join(input_tree_structure_dir, file), 'r', encoding='utf-8') as fr:
        with open(pth.join(output_tree_structure_dir, file), 'w', encoding='utf-8') as fw:
            fw.write('[user_id, tweet_id, text, retweet, favorite]\n')
            lines = fr.readlines()
            firstline = lines[0]
            record_num = 0
            two_list = firstline.split('->')
            # first_record = eval(two_list[1])  # TODO: no 'ROOT' in twitter16, thus, please chosse twolist[0]
            first_record = eval(two_list[0])
            source_tweet_id = first_record[1]
            source_user_id = first_record[0]

            if str(source_tweet_id) != str(file_tweet_id) or str(source_tweet_id) not in recorded_comments_list:
                print(source_tweet_id)
                # print(file_tweet_id)
                # print(recorded_comments_list)
                # print("Error!")
                # sleep(10000)
                # raise Exception("first record not source: {}".format(source_tweet_id))
            else:
                tree_id = str(node_id) + '_' + str(record_num)
                if source_tweet_id not in tweet2tree_id_dict.keys():
                    tweet2tree_id_dict[source_tweet_id] = tree_id
                else:
                    abnormal_count += 1
                    abnormal_list.append(source_tweet_id)
                    if not isinstance(tweet2tree_id_dict[source_tweet_id], list):
                        record_list = [tweet2tree_id_dict[source_tweet_id], tree_id]
                        tweet2tree_id_dict[source_tweet_id] = record_list
                    else:
                        tweet2tree_id_dict[source_tweet_id].append(tree_id)

                write_tweet_list.append(source_tweet_id)
                # u_id, text, ret, favorite_cnt = ...
                # root_rec = ["ROOT", "ROOT", "text", "retweet", "favorite"]
                # first_rec = [u_id, source_tweet_id, text, ret, favorite_cnt]

                u_id, text = _get_tweet_record(source_tweet_id, df, recorded_comments_list)
                root_rec = ["ROOT", "ROOT", "text"]
                first_rec = [u_id, source_tweet_id, text]
                fw.write(str(root_rec) + '-->' + str(first_rec) + '\n')
                record_num += 1

            # TODO: in twitter15, the last line is 'over' and the first line contain 'ROOT',
            #  remember to adapt to the file

            for line in lines:
                if line is None:
                    continue
                two_record = line.split('->')
                record0 = eval(two_record[0])
                record1 = eval(two_record[1])
                tweet0 = record0[1]
                tweet1 = record1[1]

                if tweet1 in write_tweet_list:
                    continue
                if tweet0 not in recorded_comments_list or tweet1 not in recorded_comments_list:
                    continue
                # each tweet in a line shall be available in the recorded_tweet_list

                # assert tweet0 in write_tweet_dict.keys(), "tweet0: {}, file: {} ,counter: {}".format(tweet0, file, counter)
                # if tweet0 not in write_tweet_list:
                    # tree_id0 = write_tweet_dict[tweet0]
                # the first record in a line must have appeared before (otherwise the tree has a broken branch)
                # if tweet0 not in write_tweet_dict.keys():
                #     print("add new record: {}".format(tweet0))
                #     tree_id0 = str(node_id) + "_" + str(record_id)
                #     write_tweet_dict[tweet0] = tree_id0
                #     tweet_id = record0[1]
                #     print(tweet_id)
                #     user_id = record0[0]
                #     index = df[df['id'].astype(str) == str(tweet_id)].index.tolist()
                #     assert len(index) == 1
                #     index = index[0]
                #     print("this is index: {}".format(index))
                #     print(type(index))
                #     text = df.loc[index, 'text']
                #     retweet_count = df.loc[index, 'retweet_count']
                #     favorite_count = df.loc[index, 'favorite_count']
                #     new_write.writerow([tree_id0, tweet_id, user_id, text,
                #                         parent, retweet_count, favorite_count])
                #     record_id += 1
                if tweet0 in write_tweet_list:
                    user_id0, text0 = _get_tweet_record(tweet0, df, recorded_comments_list)
                    write_rec0 = [user_id0, tweet0, text0]
                    # user_id0, text0, retweet_count0, favorite_count0 = \
                    # _get_tweet_record(tweet0, df, recorded_comments_list)
                    # write_rec0 = [user_id0, tweet0, text0, retweet_count0, favorite_count0]
                if tweet1 not in write_tweet_list:
                    # print("add new record: {}".format(tweet0))
                    # parent = tree_id0
                    tree_id = str(node_id) + '_' + str(record_num)
                    tweet2tree_id_dict[tweet1] = tree_id
                    write_tweet_list.append(tweet1)
                    # write_tweet_dict[tweet1] = tree_id
                    # print(index)
                    # user_id1, text1, retweet_count1, favorite_count1 =\
                    user_id1, text1 = _get_tweet_record(tweet1, df, recorded_comments_list)
                    # _get_tweet_record(tweet1, df, recorded_comments_list)
                    # write_rec1 = [user_id1, tweet1, text1, retweet_count1, favorite_count1]
                    write_rec1 = [user_id1, tweet1, text1]
                    new_line = str(write_rec0) + '-->' + str(write_rec1) + '\n'
                    fw.write(new_line)
                    record_num += 1

print(len(tweet2tree_id_dict))
print("abnormal_count: {}".format(abnormal_count))
print("abnormal_list:{}".format(abnormal_list))
tools.iterable2txt(tweet2tree_id_dict, tweet2tree_id_path)
