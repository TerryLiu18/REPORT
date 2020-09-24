import pandas as pd
import os.path as pth
import util
import numpy as np
import random


def load_data_process(dataset_name, user_filter=4):
    """prepare for dataloader"""

    if dataset_name == 'twitter15':
        load_data = 'load_data15'
        SOURCE_TWEET_NUM = 1310
    elif dataset_name == 'twitter16':
        load_data = 'load_data16'
        SOURCE_TWEET_NUM = 790
    else:
        raise ValueError('please input twitter15 or twitter16')
    
    # prepare df for acceleration
    tweet_path = pth.join('../{}/comments_text_encode3.csv'.format(load_data))
    tree_dict_path = pth.join('../{}/tree_dictionary.json'.format(load_data))
    user_file_path = pth.join('../{}/filtered_user_profile{}_encode2.csv'.format(load_data, str(user_filter)))
    graph_connection_path = pth.join('../{}/graph_connections{}.json'.format(load_data, str(user_filter)))


    tree_edge_dict = util.read_dict_from_json(tree_dict_path)  # dictionary

    source_tree_id_list = [str(node) + '_0' for node in list(range(SOURCE_TWEET_NUM))]
    tweet_df = pd.read_csv(tweet_path, index_col=0)
    tweet_df['text'] = tweet_df['text'].apply(lambda text: eval(text))

    source_tweet_df = tweet_df[tweet_df['tree_id'].isin(source_tree_id_list)]
    source_tweet_df.insert(0, 'node_id', range(SOURCE_TWEET_NUM))  # range(1310))
    source_tweet_df = source_tweet_df.reset_index()
    source_tweet_df = source_tweet_df.drop('index', axis=1)

    tweet_df = tweet_df.set_index('tree_id')   # can not switch sequence here
    df_len = source_tweet_df.shape[0]
    assert df_len == SOURCE_TWEET_NUM

    u_df = pd.read_csv(user_file_path)[['node_id', 'statuses_count', 'favourites_count',
        'listed_count', 'followers_count', 'friends_count', 'text', 'year',
        'month', 'day', 'hour']].set_index('node_id')
    u_df['text'] = u_df['text'].apply(lambda text: eval(text))

    # adjdict: {'1':[1311,1312,1313], '1311':[1], '1312':[1], '1313':[1]}
    adjdict = util.read_dict_from_json(graph_connection_path)
    
    return tweet_df, u_df, source_tweet_df, tree_edge_dict, SOURCE_TWEET_NUM, adjdict