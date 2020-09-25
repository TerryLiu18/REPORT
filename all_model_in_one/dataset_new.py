import time
import random
import torch
import util
import numpy as np

import pandas as pd
import os.path as pth

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import OrderedDict
from torch.utils.data.sampler import SubsetRandomSampler
from loaddata_process import load_data_process

# tweet_df, u_df, source_tweet_df, tree_edge_dict, SOURCE_TWEET_NUM, adjdict = \
#     _load_data_preprocess('twitter16', user_filter=4)

u_df, adjdict, source_tweet_df = None, None, None

def tree2num(tree_id) -> int:
    """
    :param tree_id:  '3_5'
    :return: '5
    """
    return int(tree_id.split('_')[1])


class TwitterDataset(Dataset):
    """
    getitem: 1. source_tweet info: (feature, label)
             2. tweet_tree info: (connections, feature, label)
    """
    def __init__(self, tweet_df, source_tweet_df, tree_edge_index, df_length):
        self.tweet_df = tweet_df
        self.source_tweet_df = source_tweet_df
        self.tree_edge_index = tree_edge_index
        self.len = df_length

    def __len__(self):
        return self.len

    def get_tweet_graph(self, index):
        """
        get data for GraphGCN
        :param index:
        :return:
        """
        graph_tweet = self.source_tweet_df.loc[index, ['node_id', 'text', 'label']]
        return graph_tweet

    def get_tweet_tree(self, index):
        """
        get data for treeGAT
        :param index:
        :return:
        """

        SOURCE_TWEET_NUM = self.len
        e_index = self.tree_edge_index[str(index)]
        tree_edge_dict = e_index

        max_idx = 0
        if not e_index[str(index)+'_0']:
            max_idx = 0   # tree only has one node
        else:
            for c_list in e_index.values():   # c_list: ['0_1', '0_2']
                last_child = tree2num(c_list[-1])
                if last_child > max_idx:
                    max_idx = last_child

        # tree_tweet_df = self.tweet_df.set_index('tree_id')
        start_index = str(index) + '_0'
        if index < SOURCE_TWEET_NUM-1:   # index < 1309
            end_index = str(index+1) + '_0'
            tree_feature = self.tweet_df.loc[start_index: end_index, ['text']].iloc[:-1]
            # loc slice is both left and right closed!!!
        else:  # index = 1309
            tree_feature = self.tweet_df.loc[start_index:, ['text']]
        return (tree_edge_dict, tree_feature, max_idx)

    def __getitem__(self, index):
        graph_info = self.get_tweet_graph(index)
        tree_info = self.get_tweet_tree(index)
        return graph_info, tree_info  # tweet(dict), (tree_edge_index, tree_feature)


def tree_reset_index(batch_size, sigma_max_idx, m):
    """
    :param batch_size: number of root tree in this batch
    :param sigma_max_idx: int  (calculated in for loop)
    :param m: int
    :return: new_index
    """

    """
    x in form of 'n_m'
    tree0:  10_0(0):[10_1(3), 10_2(4), 10_3(5)] 10_1:[10_4(6), 10_5(7)]  10_2:[10_6(8)]
    tree1:  22_0(1):[22_1(9), 22_2(10), 22_3(11), 22_4(12), 22_5(13), 22_6(14)]
    tree2:  33_0(2):[33_1(15), 33_2(16), 33_3(17)] 33_2:[33_4(18)] 33_3:[2_5(19)]

    ***** transfer function: ree_id ('n_m') -> new_node_id(k) ***************

        new_node_id (child) = len(batch) - 1 + \Sigma_{i=0}^{n-1}{max('m') in tree n} + m
        (in the example above, 33_4 = (3-1) + (6 + 6) + 4 = 18)
                               22_2 = (3-1) + 6 + 2 = 10)
    """

    new_index = batch_size - 1 + sigma_max_idx + m
    return new_index


#  new_node_id = len(batch) - 1 + \\sum_{i=0}^{n-1}{max('m') in tree i} + i


def merge_tree(tree_feature_list):
    """
    merge trees with tree_id to one tree with new_node_id

    :param tree_edge_index_list:  a list of connection dict
    :param tree_feature_list:   a list of feature dict
    :return: merged_tree_edge_index, merged_tree_feature
    """
    root_features = [t.iloc[[0]] for t in tree_feature_list]
    non_root_features = [t.iloc[1:] for t in tree_feature_list]
    merged_tree_feature = pd.concat(root_features + non_root_features, axis=0, ignore_index=False)

    return merged_tree_feature


def twitter_collate(batch):
    """
    task1: graph edge_index reform and re-number;
    task2: tree edge_index%feature merge
    :param batch: graph_info(tweet), (tree_edge_index, tree_feature)
    :param embed_size:
    :return: loss_tweets_cnt, labels, graph_edge_index, graph_loss_feature, user_feature, graph_no_loss_feature, \
           merged_tree_edge_index, merged_tree_feature

    graph edge_index form:
    0-batch_size-1: loss tweet from __getitem__
    batch_size->batch_size+new_user_idx-1: user
    batch_size+new_user_idx->batch_size+new_user_idx+found_tweet: no loss tweets
    """
    global adjdict
    global u_df
    global source_tweet_df

    # start1 = time.time()
    loss_tweet_map = OrderedDict()
    user_map = OrderedDict()
    no_loss_tweet_map = OrderedDict()
    loss_tweet_idx = 0
    user_idx = 0
    no_loss_tweet_idx = 0

    labels = []
    graph_edge_index = []
    tree_feature_list = []
    bias = len(batch)
    merged_tree_edge_index = []
    new_index = 0
    sigma_max_idx = 0
    indices = []

    for graph_info, (tree_edge_index, tree_feature, max_idx) in batch:
        # new_node_id(child) = len(batch) - 1 + \Sigma_{i = 0} ^ {n - 1}{max('m') in tree n} + m

        batch_size = len(batch)

        # add merged_tree_edge_index for every tree_edge_index
        for f, c_list in tree_edge_index.items():
            m = int(f.split('_')[1])
            if m == 0:     # root node, use new_index
                father = new_index
                new_index += 1
            else:
                father = tree_reset_index(batch_size, sigma_max_idx, m)  # non root_node, use tree_reset_index
            for c in c_list:
                m = int(c.split('_')[1])
                child = tree_reset_index(batch_size, sigma_max_idx, m)
                merged_tree_edge_index.append([father, child])
                # root2children.setdefault(new_index-1, []).append(father)
                # root2children.setdefault(new_index-1, []).append(child)
        sigma_max_idx += max_idx     # update sigma after doing operation on one forest
        indices += [new_index-1] * max_idx
        # tree_edge_index_list.append(tree_edge_index)
        tree_feature_list.append(tree_feature)
        t = graph_info['node_id']
        loss_tweet_map[str(t)] = loss_tweet_idx
        loss_tweet_idx += 1
        labels.append(graph_info['label'])
        u_list = adjdict[str(t)]

        for u in u_list:
            if str(u) not in user_map:
                user_map[str(u)] = bias + user_idx
                user_idx += 1
            graph_edge_index.append([loss_tweet_map[str(t)], user_map[str(u)]])
            graph_edge_index.append([user_map[str(u)], loss_tweet_map[str(t)]])
    ## end for
    # print("sigma_max_idx", sigma_max_idx+batch_size)
    indices = list(range(0, batch_size)) + indices
    bias += len(user_map)

    for u in user_map:
        t_list = adjdict[str(u)]
        for t in t_list:
            if str(t) in loss_tweet_map:
                graph_edge_index.append([user_map[str(u)], loss_tweet_map[str(t)]])
                graph_edge_index.append([loss_tweet_map[str(t)], user_map[str(u)]])
            elif str(t) in no_loss_tweet_map:
                graph_edge_index.append([user_map[str(u)], no_loss_tweet_map[str(t)]])
                graph_edge_index.append([no_loss_tweet_map[str(t)], user_map[str(u)]])
            else:
                no_loss_tweet_map[str(t)] = bias + no_loss_tweet_idx
                no_loss_tweet_idx += 1
                graph_edge_index.append([user_map[str(u)], no_loss_tweet_map[str(t)]])
                graph_edge_index.append([no_loss_tweet_map[str(t)], user_map[str(u)]])

    merged_tree_feature = merge_tree(tree_feature_list)
    merged_tree_feature = merged_tree_feature['text'].tolist()

    loss_t_list = [int(t) for t in list(loss_tweet_map.keys())]
    ## sort no_loss nodes based on values
    sort_no_loss = sorted(no_loss_tweet_map.items(), key=lambda item: item[1])
    no_loss_t_list = [int(t) for t, _ in sort_no_loss]
    u_list2 = [int(u) for u in list(user_map.keys())]
    graph_nodes = loss_t_list + no_loss_t_list
    graph_node_features = source_tweet_df.loc[graph_nodes]['text'].tolist()
    
    ## edge index
    graph_edge_index = np.transpose(graph_edge_index)
    merged_tree_edge_index = np.transpose(merged_tree_edge_index)
    # print("edge_index shape: ", graph_edge_index.shape)
    
    ## user features
    user_feature = u_df.loc[u_list2]
    user_text = user_feature['text'].tolist()
    # print(user_feature['text'].head())
    user_feats = user_feature[['statuses_count', 'favourites_count','listed_count', \
                                'followers_count', 'friends_count','year','month', \
                                'day', 'hour']].values.tolist()
    
    return torch.LongTensor(graph_node_features), torch.LongTensor(graph_edge_index), \
            torch.LongTensor(user_text), torch.tensor(user_feats, dtype=torch.float32), \
            torch.LongTensor(merged_tree_edge_index), torch.LongTensor(merged_tree_feature), \
            torch.LongTensor(labels), torch.LongTensor(indices)


def get_dataloader(tweet_df, source_tweet_df_para, tree_edge_dict, SOURCE_TWEET_NUM, u_df_para, adjdict_para, batch_size=64, seed=0):
    global u_df
    global adjdict
    global source_tweet_df

    u_df = u_df_para
    adjdict = adjdict_para
    source_tweet_df = source_tweet_df_para
    tweetdata = TwitterDataset(tweet_df=tweet_df, source_tweet_df=source_tweet_df_para,
                 tree_edge_index=tree_edge_dict, df_length=SOURCE_TWEET_NUM)
    indices = list(range(SOURCE_TWEET_NUM))
    random.seed(seed)
    random.shuffle(indices)
    split = int(SOURCE_TWEET_NUM * 0.8)
    train_indices, test_indices = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    train_data = DataLoader(tweetdata,batch_size=batch_size,sampler=train_sampler,collate_fn=twitter_collate,num_workers=5)
    test_data = DataLoader(tweetdata,batch_size=batch_size,sampler=test_sampler,collate_fn=twitter_collate,num_workers=5)
    return train_data, test_data


if __name__ == "__main__":
    start = time.time()
    tweet_df, u_df, source_tweet_df, tree_edge_dict, SOURCE_TWEET_NUM, adjdict = \
    load_data_process('twitter16', user_filter=4)
    train_data, test_data = get_dataloader(tweet_df, source_tweet_df, tree_edge_dict, SOURCE_TWEET_NUM)
    for x in train_data:
        # loss_tweets_cnt, labels, graph_edge_index, graph_loss_feature, user_feature, graph_no_loss_feature, \
        # merged_tree_edge_index, merged_tree_feature
        print('-----------------------0.graph_node_feature------------------')
        print(x[0])
        print('-----------------------1.graph_edge_index------------------')
        print(x[1])
        print('-----------------------2.user_text------------------')
        print(x[2])
        print('-----------------------3.user_feats------------------')
        print(x[3])
        print('-----------------------4.merged_tree_edge_index-----------------')
        print(x[4])
        print('-----------------------5.merged_tree_feature------------------')
        print(x[5])
        print('-----------------------6.labels------------------')
        print(x[6])
        print('-----------------------7.indices------------------')
        print(x[7])