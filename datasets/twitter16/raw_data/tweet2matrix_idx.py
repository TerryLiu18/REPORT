import os
import json
import os.path as pth

def read_dict_from_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # print("{} successfully loaded!".format(path))
    return data

def save_dict_to_json(item, path, overwrite=True):
    if os.path.exists(path) and overwrite is False:
        print("{} already exists".format(path))
    else:
        try:
            item = json.dumps(item, indent=4)
            with open(path, "w", encoding='utf-8') as f:
                f.write(item)
        except Exception as e:
            print("write error==>", e)

tree_dir = pth.join(os.getcwd(), 'tweet_tree')
tweet2matrix_idx = dict()
cnt = 0

for file in os.listdir(tree_dir):
    tweet_id = file.split('.')[0]
    tweet2matrix_idx[tweet_id] = cnt
    cnt += 1
    
save_dict_to_json(tweet2matrix_idx, 'tweet2matrix_idx.json')


