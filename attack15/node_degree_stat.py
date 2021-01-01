import os
import csv
import os.path as pth
import pandas as pd
from attack import tools
from attack.tools import save_dict_to_json, read_dict_from_json, txt2iterable, iterable2txt
from attack.task import DATASET_NAME, LOAD_DATA, SOURCE_TWEET_NUM

tree_dict_path = pth.join('../load_data15/tree_dictionary.json')
user_profile_path = pth.join('../load_data15/filtered_user_profile3_encode2.csv')
graph_connect_path = pth.join('../load_data15/graph_connections3.json')

node2label_path = pth.join('source_tweet_info.csv')
df = pd.read_csv(node2label_path, lineterminator='\n')
mat_id_list = df['matrix_idx'].tolist()
label_list = df['label'].tolist()
# print(mat_id_list)
# print(label_list)
node2label = dict(zip(mat_id_list, label_list))
node2label_path = pth.join('node_id2label_dict.json')
save_dict_to_json(node2label, node2label_path)
print(node2label)
print('-'*89)

tree_dict = read_dict_from_json(tree_dict_path)
user_profile_df = pd.read_csv(user_profile_path, lineterminator='\n')
graph_connect_dict = read_dict_from_json(graph_connect_path)

node_degree_dict = {}
# print(graph_connect_dict)
for node, connect_nodes in graph_connect_dict.items():
    if int(node) < SOURCE_TWEET_NUM:  # is tweet node
        degree = len(connect_nodes)
        node_degree_dict[node] = degree

node_degree_dict = sorted(node_degree_dict.items(), key=lambda item: int(item[0]), reverse=True)
print(node_degree_dict)
node_degree_dict_path = pth.join('node_degree_dict.json')
save_dict_to_json(node_degree_dict, node_degree_dict_path)

node2label2degree = pth.join('node2label2degree.csv')
with open(node2label2degree, 'w', newline='') as fw:
    csv_write = csv.writer(fw)
    csv_write.writerow(['node_id', 'label', 'degree'])
    for node, degree in node_degree_dict:
        label = node2label[int(node)]
        csv_write.writerow([node, label, degree])

