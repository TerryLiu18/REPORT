'''
Fun: construct dictionary for words from tweets or user description
Huajie @ 2020/8/15
'''

import os
from os.path import dirname, abspath
import json
from datetime import datetime
import pandas as pd
import numpy as np


# modified by tianrui
def _get_word_list(file_name):
    """available to both users and tweets"""
    df = pd.read_csv(file_name, encoding='utf-8')
    df = pd.DataFrame(df, columns=['text'])
    df['text'] = df['text'].replace(['None'], '')
    df = df.where(df.notnull(), '')
    df_text = df['text'].str.split()
    text_list = df_text.tolist()
    words_list = []
    sent_len = []
    for text in text_list:
        sent_len.append(len(text))
        words_list.extend(text)
    ## max sent_len
    return words_list, sent_len


def _padding_sen(seq_indx, max_len):
    diff = max_len - len(seq_indx)
    return seq_indx + [3] * diff


def _data_normalize(df):
    """normalize value of each column"""
    list_cols = ['statuses_count', 'favourites_count', 'listed_count', 'followers_count',
                 'friends_count', 'year', 'month', 'day', 'hour']
    df[list_cols] = df[list_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    return df


def _convert_createdTime(df_created):
    time_list = df_created.tolist()
    df_new = pd.DataFrame(time_list, columns=['year', 'month', 'day', 'hour']).astype(int)
    # print(df_new.head())
    return df_new


def _word2index(text, word2index, max_len):
    seq_indx = [word2index.get(x, 0) for x in text]
    if len(seq_indx) < max_len:
        seq_indx = _padding_sen(seq_indx, max_len)
    else:
        seq_indx = seq_indx[:max_len]
    sent_input = [1] + seq_indx + [2]

    return sent_input


def _user_seq2index(file_name, out_file, word2index, max_len):
    # print(word2index)
    print("text max len: ", max_len)
    df = pd.read_csv(file_name)

    # convert created time
    df.drop(["geo_enabled", "location"], axis=1, inplace=True)
    df_created = df['created_time'].str.split('-')
    df_new = _convert_createdTime(df_created)

    # concate two data frames
    df = pd.concat([df, df_new], axis=1)

    # normalize values of columns
    df = _data_normalize(df)
    # print(df.head())

    # encode text to index
    df_text = pd.DataFrame(df, columns=['text'])
    df_text['text'] = df_text['text'].replace(['None'], '')
    df_text = df_text.where(df_text.notnull(), '')
    # print(df_text.head())

    df_text['text'] = df_text['text'].str.split()
    df['text'] = df_text['text'].apply(lambda text: _word2index(text, word2index, max_len))
    # print(df_text.head())

    df.to_csv(out_file, sep=',', encoding='utf-8', index=False)


def _tweets_seq2index(file_name, out_file, word2index, max_len):
    # print(word2index)
    print("text max len: ", max_len)
    df = pd.read_csv(file_name)
    # encode text to index
    df_text = pd.DataFrame(df, columns=['text'])
    df_text['text'] = df_text['text'].replace(['None'], '')
    df_text = df_text.where(df_text.notnull(), '')
    # split string to list
    df_text['text'] = df_text['text'].str.split()
    df['text'] = df_text['text'].apply(lambda text: _word2index(text, word2index, max_len))

    df.to_csv(out_file, sep=',', encoding='utf-8', index=False)


def _words_count_filter(words_list, frequency=2):
    """count the words in the tweets"""
    word2num = dict()
    for w in words_list:
        if w in word2num:
            word2num[w] += 1
        else:
            word2num[w] = 1

    # choose freq words to construct dict
    words_set = []
    for w, num in word2num.items():
        if num >= frequency:
            words_set.append(w)
    return words_set


def _map_word2index(words_list, out_file):
    word2index = dict()
    word2index['<UNK>'] = 0
    word2index['<SOS>'] = 1
    word2index['<END>'] = 2
    word2index['<PAD>'] = 3
    for i, word in enumerate(words_list):
        word2index[word] = i + 4
    # write to json file
    # print("save word2index to >>: ", out_file)
    with open(out_file, "w") as fout:
        json.dump(word2index, fout, indent=4)
    return word2index


def main():
    filter_num = 5
    dir_path = dirname(dirname(abspath(__file__)))
    print(dir_path)
    file_path = os.path.join('load_data16/filtered_user_profile{}.csv'.format(str(filter_num)))
    user_file_path = os.path.join(dir_path, file_path)
    # print("file path >> ", file_path)
    print("file path >> ", user_file_path)
    print('map user description to dict and index')
    words_list, sent_len = _get_word_list(user_file_path)
    words_set = _words_count_filter(words_list, frequency=2)

    out_file = 'load_data16/filtered_user_profile_words_mapping{}.json'.format(str(filter_num))
    out_file = os.path.join(dir_path, out_file)
    print("users unique words count: ", len(words_set))
    word2index = _map_word2index(words_set, out_file)

    # map sentence to index
    max_len = np.percentile(sent_len, 99)
    out_file = os.path.join(
        dir_path, 'load_data16/filtered_user_profile{}_encode1.csv'.format(str(filter_num))
    )
    _user_seq2index(user_file_path, out_file, word2index, int(max_len))

    # read tweets content
    print('-' * 89)
    print('***map tweets words to index using dictionary***')
    tweet_file = os.path.join('../load_data16/comments_text.csv')
    tweet_words_list, sent_len = _get_word_list(tweet_file)

    # choose the freq words
    words_set = _words_count_filter(tweet_words_list, frequency=3)
    print("Tweet unique words count: ", len(words_set))

    # map words to index using dict
    out_file = 'load_data16/tweets_words_mapping.json'
    out_file = os.path.join(dir_path, out_file)
    word2index = _map_word2index(words_set, out_file)
    # print(sorted(sent_len))
    # replace text with index
    max_len = np.percentile(sent_len, 99)
    out_file = os.path.join(
        dir_path, 'load_data16/comments_text_encode1.csv'
    )
    _tweets_seq2index(tweet_file, out_file, word2index, int(max_len))


if __name__ == '__main__':
    main()
    print("well done!!")
