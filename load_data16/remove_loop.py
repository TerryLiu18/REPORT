from pre_process.tools import read_dict_from_json, save_dict_to_json

def get_tail(num):
    """
    :param num:'0_3' 
    :return: 3
    """
    return int(num.split('_')[1])


tree_dict = read_dict_from_json('tree_dictionary_backup.json')
new_tree_dict = dict()
# print(tree_dict)
for num in tree_dict.keys():
    new_tree_dict[num] = dict()
    node_stack = set()
    node_stack.add(0)
    one_tree_record = tree_dict[num]
    for node_id, child_list in one_tree_record.items():
        if get_tail(node_id) in node_stack:
            new_tree_dict[num][node_id] = []
            for child in child_list:
                if get_tail(child) not in node_stack:
                    node_stack.add(get_tail(child))
                    new_tree_dict[num][node_id].append(child)
            if new_tree_dict[num][node_id] == [] and get_tail(node_id) != 0:
                del new_tree_dict[num][node_id]
    # break

save_dict_to_json(new_tree_dict, 'tree_dictionary.json')