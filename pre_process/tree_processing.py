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

if not pth.isdir('../datasets/twitter16/processed_data/processed_tweet_tree_profile'):
    os.mkdir(pth.join('../datasets/twitter16/processed_data/processed_tweet_tree_profile'))

if not pth.isdir('../datasets/twitter16/processed_data/processed_tree_tree'):
    os.mkdir(pth.join('../datasets/twitter16/processed_data/processed_tree_tree'))

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
    # u_id, tex, retweet_cnt, favorite_cnt = \
    #     df.loc[idx, 'user_id'], df.loc[idx, 'text'], df.loc[idx, 'retweet_count'], df.loc[idx, 'favorite_count']
    # return str(u_id), str(tex), retweet_cnt, favorite_cnt
    u_id, tex = df.loc[idx, 'tweet_id'], df.loc[idx, 'text']
    return str(u_id), str(tex)


counter = 0
abnormal_count = 0
abnormal_list = []
input_tree_structure_dir = pth.join(raw_data_dir, "tree_tree")
f_list = os.listdir(input_tree_structure_dir)

for file in tqdm(f_list):
    source_id = file.split('.')[0]
    f_path = pth.join(input_tree_structure_dir, file)
    fout_path = pth.join(output_tree_structure_dir, file)
    content = pd.read_csv(pth.join(input_tree_content_dir, source_id+'.csv'), lineterminator='\n', encoding='utf-8')
    df = pd.DataFrame(content)
    recorded_comments_list = df['tweet_id'].astype(str).tolist()  # these are the tweets whose texts are unique and can be found
    node_id = tweet2node_dict[source_id]  # node_id: from 0 to 1311; content {node_id:[node_id1, node_id2]...}
    first_tree_id = str(node_id) + '_0'
    tweet2tree_id_dict[source_id] = first_tree_id

    record_num = 0
    record_dict = dict()
    write_tweet_set = set()
    write_tweet_set.add(source_id)

    with open(f_path, 'r', encoding='utf-8') as fr, open(fout_path, 'w', encoding='utf-8') as fw:
        # fw.write('[user_id, tweet_id, text]\n')   # todo: I have delete all titles!
        lines = fr.readlines()

        # get parent_record_dict
        parent_record_dict = tree_checker.tree_line_checker(lines, source_id)

        firstline = lines[0].rstrip()
        source_user_id, _, source_tweet_id, _ = tree_checker.parse_record(firstline)
        # first_record = eval(two_list[1])  # TODO: no 'ROOT' in twitter16, thus, please chosse twolist[0]

        if str(source_tweet_id) != str(source_id):
            print('source_id', source_id)

    # TODO: in twitter15, the last line is 'over' and the first line contain 'ROOT',
    #  remember to adapt to the file

        for line in lines:
            if line is None:
                continue
            user0, user1, tweet0, tweet1 = tree_checker.parse_record(line)
            if tweet0 == tweet1:
                continue
            if tweet0 not in recorded_comments_list or tweet1 not in recorded_comments_list:
                continue
            if tweet0 not in parent_record_dict.keys():
                continue
            else:
                if parent_record_dict[tweet0] is True:
                    if tweet1 not in write_tweet_set:
                        record_num += 1
                        tree_id = str(node_id) + '_' + str(record_num)
                        tweet2tree_id_dict[tweet1] = tree_id
                        write_tweet_set.add(tweet1)

                    user_id0, text0 = _get_tweet_record(tweet0, df, recorded_comments_list)
                    write_rec0 = [user_id0, tweet0, text0]
                    user_id1, text1 = _get_tweet_record(tweet1, df, recorded_comments_list)
                    write_rec1 = [user_id1, tweet1, text1]
                    new_line = str(write_rec0) + '->' + str(write_rec1) + '\n'
                    fw.write(new_line)

            # # if tweet1 in write_tweet_list:
            # #     continue
            #
            # if tweet0 in write_tweet_list:
            #     user_id0, text0 = _get_tweet_record(tweet0, df, recorded_comments_list)
            #     write_rec0 = [user_id0, tweet0, text0]
            # if tweet1 not in write_tweet_list:
            #     tree_id = str(node_id) + '_' + str(record_num)
            #     tweet2tree_id_dict[tweet1] = tree_id
            #     write_tweet_list.append(tweet1)
            #     user_id1, text1 = _get_tweet_record(tweet1, df, recorded_comments_list)
            #     write_rec1 = [user_id1, tweet1, text1]
            #     new_line = str(write_rec0) + '->' + str(write_rec1) + '\n'
            #     fw.write(new_line)
            #     record_num += 1

print(len(tweet2tree_id_dict))
print("abnormal_count: {}".format(abnormal_count))
print("abnormal_list:{}".format(abnormal_list))
tools.iterable2txt(tweet2tree_id_dict, tweet2tree_id_path)
