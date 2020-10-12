import os
import json
import os.path as pth
import pandas as pd
from tools import txt2iterable, save_dict_to_json

def get_bad_user(thresh_score, df):
    bad_user_dict = {}
    df = df[df['fake_score'].astype(int) >= thresh_score]
    for i, row in df.iterrows():
        user = str(row['user_id']).split('.')[0]
        tree_id = row['tree_id']
        if user in bad_user_dict.keys():
            bad_user_dict[user].append(tree_id)
        else:
            bad_user_dict[user] = [tree_id]
    return bad_user_dict


def save_bad_user(bad_user_dict, thresh_score):
    save_path = pth.join('bad_user_score{}.json'.format(thresh_score))
    save_dict_to_json(bad_user_dict, save_path)
    print('save >{} score user to {}'.format(thresh_score, save_path))


df = pd.read_csv('filtered_comments_text_with_label2.csv', lineterminator='\n')
score = [16, 17, 18, 20, 25, 30, 40]
for s in score:
    bad_user_dict = get_bad_user(s, df)
    save_bad_user(bad_user_dict, s)