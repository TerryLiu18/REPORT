from pre_process.tools import read_dict_from_json, save_dict_to_json


def get_tail(num):
    """
    :param num:'0_3' 
    :return: 3
    """
    return int(num.split('_')[1])


def int2rec(node_id, num):
    """
    :param node_id: 0
    :param num: 3
    :return: '0_3'
    """
    return str(node_id)+'_'+str(num)


def sort_record(rec_list):
    """
    :param rec_list:  ['0_3', '0_1', '0_2', '0_0']
    :return: ['0_0','0_1', '0_2', '0_3', ]
    """
    if not rec_list:  # empty list
        return rec_list
    node_id = rec_list[0].split('_')[0]
    rec_list = [get_tail(i) for i in rec_list]
    max_idx = max(rec_list)
    sorted_record_list = [int2rec(node_id, i) for i in range(max_idx+1)]
    return sorted_record_list



# tree_dict = read_dict_from_json('backup/tree_dictionary_backup.json')
# new_tree_dict = dict()
# # print(tree_dict)
# for num in tree_dict.keys():
#     new_tree_dict[num] = dict()
#     node_stack = set()
#     node_stack.add(0)
#     one_tree_record = tree_dict[num]
#     for node_id, child_list in one_tree_record.items():
#         if get_tail(node_id) in node_stack:
#             new_tree_dict[num][node_id] = []
#             for child in child_list:
#                 if get_tail(child) not in node_stack:
#                     node_stack.add(get_tail(child))
#                     new_tree_dict[num][node_id].append(child)
#             if new_tree_dict[num][node_id] == [] and get_tail(node_id) != 0:
#                 del new_tree_dict[num][node_id]
    # break

# save_dict_to_json(new_tree_dict, 'tree_dictionary_no_sort.json')


no_sort_tree_dict = read_dict_from_json('tree_dictionary_no_sort.json')
for num in no_sort_tree_dict.keys():
    one_tree_record = no_sort_tree_dict[num]
    for node_id, child_list in one_tree_record.items():
        if len(child_list) != 0 :
            child_list.sort(key=lambda i: int(i.split('_')[1]))

print(no_sort_tree_dict)
save_dict_to_json(no_sort_tree_dict, 'tree_dictionary.json')



