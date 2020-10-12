import os
import os.path as pth
import csv
import pandas as pd
from attack.tools import save_dict_to_json, read_dict_from_json
from attack.task import FILTER_NUM, DATASET_NAME, LOAD_DATA, SOURCE_TWEET_NUM


original_graph_name = "graph_connection{}.json".format(FILTER_NUM)
graph_trace_dir = pth.join('tree_dict_trace')
graph0 = pth.join(graph_trace_dir, original_graph_name)

bad_user_path = pth.join('../attack15/bad_user_score20.json')
bad_users_dict = read_dict_from_json(bad_user_path)
bad_users = list(bad_users_dict.keys())
bad_users = [i.split('.')[0] for i in bad_users]
all_user_path = pth.join('../load_data15_1473/filtered_user_profile2_encode2.csv')
df = pd.read_csv(all_user_path, lineterminator='\n')
node_id_list = df['node_id'].tolist()
user_id_list = df['user_id'].tolist()
node_id_list = [int(i) for i in node_id_list]
user_id_list = [str(i) for i in user_id_list]
user2node_dict = dict(zip(user_id_list, node_id_list))

# take record of add trace
graph_trace_list = []    # take record of the best K add for each step (global list)



def graph_add_edge(graph, user, tweet):
    assert int(user) >= SOURCE_TWEET_NUM
    if int(user) in graph[str(tweet)] and \
            int(tweet) in graph[str(user)]:
        print('exists!')
        return graph
    else:
        # print(graph[str(user)])
        graph[str(user)].append(int(tweet))
        graph[str(tweet)].append(int(user))
        # print(graph[str(user)])
        return graph

graph_path = pth.join('../load_data15/graph_connections2.json')
graph = read_dict_from_json(graph_path)
new_graph = graph_add_edge(graph, 10000, 1000)
# print(new_graph)



# def choose_best_attack(new_graph, old_graph):
#     # model***
#     # current_output_list = None
#     # expect_output_list = None
#     # assert len(current_output_list) == len(expect_output_list)
#
#     #todo: we should create a function in main_attack.py
#     # def test_target_output(graph_connection, old_graph_connection):
#     #     """output: score"""
#     #     all_output = torch.zeros(SOURCE_TWEET_NUM, 4)
#     #     for data in test_loader:
#     #         output = model(user_feats, graph_node_features, graph_connection, merged_tree_feature, merged_tree_edge_index, indx)
#     #         output = F.softmax(output, dim=1)  # shape: batchsize*4
#     #         all_output[xxx] = output  #similar to BiGCN concat?
#     #         # here similarly we get a old_output, old_all_output for comparison
#     #     target_output = all_output[chosen_node]
#     #     # suppose there are F target tweets, then the target_output will be of shape: N*4
#     #     # a example:
#     #  """
#     # [[-1.4431, -1.2558, -1.4073, -1.4517],
#     #  [-1.5590, -1.1705, -1.4162, -1.4406],
#     #  [-1.0486, -1.3741, -1.6422, -1.5948],
#     #  [-0.8670, -1.4667, -1.7875, -1.7052],
#     #  [-1.3005, -1.2864, -1.5100, -1.4678],
#     #  [-0.7508, -1.5499, -1.9028, -1.7922],
#     #  [-1.0351, -1.3763, -1.6541, -1.6044]]
#     #  """
#     #      fake_num = xxx  # should be start with N, and then decrease
#     #      attack_score = 0
#     #      attack_minor_score = 0
#     #      _, y_pred = all_output.max(dim=1)
#     #      fake_output = all_output[:,1]
#     #      for y_p in y_pred:
#     #         if y_p != 1:
#     #            fake_num = fake_num + 1
#     #      attack_score = fake_num - old_fake_num
#     #      calculate the decrease of each fake_output  (for example, from 0.5->0.35, get 0.15, from 0.5->0.55, get 0)
#     #      return attack_score, fake_output_decrease
#     #
#
#
#     # old_output = test_attack_model(old_graph)
#     # new_output = test_attack_model(new_graph)
#
#
#     L_loss_list = [abs(current_output_list[i] - expect_output_list[i]) for i in range(len(current_output_list))]
#     L_loss = sum(L_loss_list)
#     return L_loss
#


def alter_graph(graph, bad_user_set, tweet_set, graph_trace_list):
    """
    alter graph using Beam Search Algorithm
    {(user1,tweet1): loss1, (user2,tweet2): loss2 ...}
    """

    graph_trace_cluster = graph_trace_list[-1]  # cluster will be [(a,b), (c,d) ...]  (K pairs)
    for edge in graph_trace_cluster:
        user = edge[0]
        tweet = edge[1]
        attack_score_dict = []
        fake_output_decrease_dict = []
        for bad_user in bad_user_set:
            for tweet in tweet_set:
                if int(bad_user) not in graph[str(tweet)] and\
                    int(tweet) not in graph[str(bad_user)]:
                    new_graph = graph_add_edge(graph, bad_user, tweet)
                    attack_score, fake_output_decrease = test_target_output(new_graph, graph)

                    # take record added edge->attack_score
                    attack_score_dict[(str(bad_user), str(tweet))] = attack_score
                    fake_output_decrease_dict[(str(bad_user), str(tweet))] = fake_output_decrease
    return attack_score_dict, fake_output_decrease_dict



def update_graph(self, attack_score_dict, fake_output_decrease_dict):
    """return the best K attack  (which make the output varies the most)"""
    selected_attack = []
    record = []
    attack_score_dict = sorted(attack_score_dict.items(), key=lambda x: x[1], reverse=True)
    fake_output_decrease_dict = sorted(fake_output_decrease_dict.items(), key=lambda x: x[1], reverse=True)
    # if max(attack_score
    for i in range(self.K):    # we user K=1 first
        selected_attack.append(fake_output_decrease_dict[i])
    record = list(fake_output_decrease_dict.keys())
    graph_trace_list.append(record)
    return selected_attack


