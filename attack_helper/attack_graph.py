import os
import os.path as pth
import csv
import pandas as pd
from attack.tools import save_dict_to_json, read_dict_from_json
from attack.task import FILTER_NUM, DATASET_NAME, LOAD_DATA

original_graph_name = "graph_connection{}.json".format(FILTER_NUM)
graph_trace_dir = pth.join('tree_dict_trace')
graph0 = pth.join(graph_trace_dir, original_graph_name)

bad_user_path = pth.join('bad_user_score20.json')
bad_users_dict = read_dict_from_json(bad_user_path)
bad_users = list(bad_users_dict.keys())
bad_users = [i.split('.')[0] for i in bad_users]


class yrange:
    def __init__(self, n):
        self.i = 0
        self.n = n

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.n:
            i = self.i
            self.i += 1
            return i
        else:
            raise StopIteration()

def read_nth_tree(n):
    nth_graph_name = "graph_connection{}_{}.json".format(FILTER_NUM, n)
    new_graph = read_dict_from_json(nth_graph_name)
    return new_graph

def graph_add_edge(graph, user, tweet):
    user = str(user).split('.')[0]
    tweet = str(tweet).split('.')[0]
    graph[str(user)].append(int(tweet))
    graph[str(tweet)].append(int(user))
    return graph


def test_attack(new_graph, old_graph):
    # model***
    current_output_list = None
    expect_output_list = None
    assert len(current_output_list) == len(expect_output_list)
    L_loss_list = [abs(current_output_list[i] - expect_output_list[i]) for i in range(len(current_output_list))]
    L_loss = sum(L_loss_list)
    return L_loss


class AttackGraph:
    """change tree"""
    def __init__(self, n, bad_user_list, target_tweet_list, loss_value, cost, k=1):
        self.graph = read_nth_tree(n)            # self.graph is a dictionary
        self.bad_user_set = bad_user_list
        self.target_tweet = target_tweet_list
        self.loss_value = loss_value
        self.cost = cost
        self.K = k

    def alter_graph(self):
        """
        get L_loss_list:
        {(user1,tweet1): loss1, (user2,tweet2): loss2 ...}
        """
        L_loss_dict = dict()   # edge2loss
        for bad_user in self.bad_user_set:
            for fake_tweet in self.target_tweet:
                new_graph = graph_add_edge(self.graph, bad_user, fake_tweet)
                target_loss = test_attack(new_graph, self.graph)
                L_loss_dict[(str(bad_user), str(fake_tweet))] = target_loss
        return L_loss_dict

    def update_graph(self, L_loss_dict):
        """return the best K attack  (which make the output varies the most)"""
        L_loss_dict = sorted(L_loss_dict.items(),key = lambda x:x[1],reverse = True)
        selected_attack = []
        for i in range(self.K):
            selected_attack.append(L_loss_dict[i])
        return selected_attack




def add_edge(tree_dict_path, N, bad_user_set, target_tweet_set, K=1):
    """
    input: tree_dict(N-1),
    output: tree_dict(N) * K (adopt beam search?), N+1
    """
    tree_dict =
