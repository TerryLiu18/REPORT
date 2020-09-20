import os
import json
from pre_process.tools import read_dict_from_json

# def save_dict_to_json(item, path, overwrite=True):
#     if os.path.exists(path) and overwrite is False:
#         print("{} already exists".format(path))
#     else:
#         try:
#             item = json.dumps(item, indent=4)
#             with open(path, "w", encoding='utf-8') as f:
#                 f.write(item)
#                 print("success write dict to json: {}".format(path))
#         except Exception as e:
#             print("write error==>", e)



graph16 = read_dict_from_json('../load_data16/tw16_connections.json')


cnt = 0
iso_list = []
for i in range(len(graph16)):
    if str(i) not in graph16.keys():
        cnt += 1
        iso_list.append(i)
        # print('i:', i)
print(cnt)
print(iso_list)