import os
import os.path as pth
from tools import txt2iterable, read_dict_from_json

graph_connection_path = '../load_data15_1473/graph_connections2.json'
adjdict = read_dict_from_json(pth.join(graph_connection_path))

n2l_path = pth.join('../attack15/node_id2label_dict.json')
node2label_dict = read_dict_from_json(n2l_path)

def get_fake_score(user_node):
    # user_index = user_map[str(user_node)]
    user_appear_tweet_list = adjdict[str(user_node)]

    fake_score = 0
    fake_score_dict = {'0': 0, '1': 10, '2': -1, '3': 0}
    for tweet in user_appear_tweet_list:
        label = node2label_dict[str(tweet)]
        fake_score += fake_score_dict[str(label)]
    return fake_score

