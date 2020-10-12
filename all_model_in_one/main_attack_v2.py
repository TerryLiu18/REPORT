'''
tianrui liu @2020/09/11
Fun: tune the model
'''
import os
import time
import json
import argparse
import random
import torch
import torch.nn.functional as F
import numpy as np
import os.path as pth
import pandas as pd

from time import sleep
from tqdm import tqdm
from datetime import datetime
from torch import nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import copy
import util
from util import str2bool
from GloveEmbed import _get_embedding
from early_stopping_attack_v2 import EarlyStopping
from dataset_attack import get_dataloader, data_split
from evaluate import evaluation4class

SOURCE_TWEET_NUM = 1472

## add args
if True:   # parameter
    parser = argparse.ArgumentParser(description='GAT for fake news detection')
    parser.add_argument('--model', default='ensemble', type=str, help='ensemble/graph2tree/tree2graph/tree/graph')
    parser.add_argument('--train', default=True, type=str2bool, help='train or traverse')
    parser.add_argument('--patience', default=10, type=int, help='how long to wait after last time validation loss improved')
    parser.add_argument('--freeze', default=False, type=str2bool, help='embedding freeze or not')
    parser.add_argument('--load_ckpt', default=False, type=str2bool, help='load checkpoint')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--gpu', default=1, type=int, help='gpu id')
    parser.add_argument('--epoches', default=90, type=int, help='maximum training epoches')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--dropout', default=0.3, type=float, help='drop out rate')
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='weight decay')
    parser.add_argument('--embed_dim', default=100, type=int, help='pretrain embed size')
    parser.add_argument('--tweet_embed_size', default=100, type=int, help='tweet embed size')
    parser.add_argument('--tree_hidden_size1', default=100, type=int, help='hidden size for TreeGCN')
    parser.add_argument('--tree_hidden_size2', default=100, type=int, help='hidden size for TreeGCN')

    parser.add_argument('--graph_hidden_size1', default=100, type=int, help='hidden size for GraphGCN')
    parser.add_argument('--graph_hidden_size2', default=100, type=int, help='hidden size for GraphGCN')

    parser.add_argument('--linear_hidden_size1', default=64, type=int, help='hidden size for fuly connected layer')

    parser.add_argument('--direction', default='td', type=str, help='tree direction: topdown(td)/bottomup(bu)')
    parser.add_argument('--user_feat_size', default=9, type=int, help = 'user features')
    # parser.add_argument('--text_input_size', default=200, type=int, help = 'tweets and description input size')
    parser.add_argument('--user_out_size', default=100, type=int, help = 'user description embed size')
    # parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_name', default='best', type=str, help='load previous checkpoint. insert checkpoint filename')
    parser.add_argument('--attack', default='True', type=str2bool, help='whether testing attack')
    args = parser.parse_args()
    args.user_out_size = int(args.tweet_embed_size/2)
# args.graph_hidden_size2 = args.tweet_embed_size


# import model
if args.model == 'ensemble':
    print('---using ensemble---')
    from model_ensemble import Net
elif args.model == 'graph':
    print('---using graph---')
    from model_graph import Net
elif args.model == 'tree':
    print('---using tree---')
    from model_tree import Net
elif args.model == 'graph2tree':
    print('---using graph2tree---')
    from model_graph2tree import Net
    args.graph_hidden_size2 = args.tweet_embed_size
elif args.model == 'tree2graph':
    print('---using tree2graph---')
    from model_tree2graph import Net
    args.tree_hidden_size2 = args.tweet_embed_size
else:
    raise ValueError('parameter not found! input should among "ensemble/graph2tree/tree2graph/tree/graph"')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print("device: {}{} ".format(device, str(args.gpu)))

def _load_word2index(word_file):
    with open(word_file) as jsonfile:
        word_map = json.load(jsonfile)

    vocab_size = len(word_map)
    return word_map, vocab_size

def _load_checkpoint():
    ckp_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    if os.path.isfile(ckp_path):
        checkpoint = torch.load(ckp_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optim_state'])
        global_step = checkpoint['global_step']
        curr_epoch = checkpoint['curr_epoch']
    return model, optimizer, global_step, curr_epoch


def adjust_learning_rate(optim, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 15))
    for param_group in optim.param_groups:
        param_group['lr'] = lr

def _compute_accy_count(y_pred, y_labels):
    return 1.0*y_pred.eq(y_labels).sum().item()
    
def _compute_accuracy(y_pred, y_labels):
    return 1.0*y_pred.eq(y_labels).sum().item()/y_labels.size(0)
    
def _train_model(train_indices, test_indices, model):
    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    os.makedirs('logs/', exist_ok=True)
    log_file = 'logs/' + time + 'args.log'
    fw_log = open(log_file, "w")
    json.dump(args.__dict__, fw_log, indent=2)
    fw_log.write('\n')

    # take record of parameters
    parameter_record = pth.join('./parameter_record.md')   
    md = open(parameter_record, 'a') 

    model.train()
    global_step = 0
    max_iter = int(np.ceil(len(train_indices) / args.batch_size)) * args.epoches
    writer = SummaryWriter()
    pbar = tqdm(total=max_iter)
    pbar.update(global_step)

    early_stopping = EarlyStopping(args.patience)
    for epoch in range(curr_epoch, args.epoches):
        # adjust_learning_rate(args.lr, optimizer, epoch)
        train_loader, _ = get_dataloader(args.batch_size, train_indices, test_indices, adjdict)
        adjust_learning_rate(optimizer, epoch)
        train_accy = []
        for data in train_loader:
            graph_node_features, graph_edge_index, user_feats = Variable(data[0]).to(device), Variable(data[1]).to(device), Variable(data[3]).to(device)
            merged_tree_edge_index, merged_tree_feature, labels = Variable(data[4]).to(device), Variable(data[5]).to(device), Variable(data[6]).to(device)
            indx = data[7].to(device)

            # print('data', data)
            global_step += 1
            pbar.update(1)
            optimizer.zero_grad()
            output = model(user_feats, graph_node_features, graph_edge_index, merged_tree_feature, merged_tree_edge_index, indx)
            # print(output)
            # print("label: ",labels)
            loss = loss_fun(output, labels)
            loss.backward()
            optimizer.step()
            ## compute accuracy
            _, y_pred = output.max(dim=1)
            accy = _compute_accuracy(y_pred, labels)
            train_accy.append(accy)
            if global_step % 10 == 0:
                print("epoch: {} global step: {} loss: {:.5f} accuracy: {:.5f}"\
                        .format(epoch, global_step, loss.item(), accy))
                fw_log.write("epoch: {} global step: {} loss: {:.5f} train accuracy: {:.5f}\n"\
                        .format(epoch, global_step, loss.item(), accy))
            writer.add_scalar('Loss/train', loss, global_step)
            if global_step % 10 == 0:
                writer.flush()
        ## save checkpoint for best accuracy epoch
        # if (epoch+1) % 1 == 0:
        #     _save_checkpoint(model, global_step, epoch)
        # if (epoch+1) % 2 == 0:
            # test_accy = _testing_model(test_loader,model)
        Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = _testing_model(model,test_loader)
        early_stopping(Acc_all, F1, F2, F3, F4, global_step, epoch, args.ckpt_dir, args.ckpt_name, model, optimizer)
        fw_log.write('epoch: {} testing accuracy: {:4f}\n'.format(epoch, Acc_all))
        fw_log.flush()
        if early_stopping.early_stop:
            # _save_checkpoint(model, early_stopping.global_step, early_stopping.epoch) #### false
            print('early stop!')
            break
        model.train()
    fw_log.write("BEST Accuracy: {:.4f}".format(early_stopping.best_accs))
    print("BEST Accuracy: {:.4f}".format(early_stopping.best_accs))
    md_write = '|{}| gpu: {} | {} | {} | {} | seed: {} | direction: {} | acc: {:.4f} | F1: {:.4f} | F2: {:.4f} | F3: {:.4f} | F4: {:.4f} | \n'.format(
        str(time), str(args.gpu), str(args.model), str(args.batch_size), str(args.lr), 
         str(args.seed), str(args.direction), early_stopping.best_accs, early_stopping.F1, early_stopping.F2, early_stopping.F3, early_stopping.F4)
    md.write(md_write)
    md.close()
    fw_log.close()
    writer.close()
    
def _testing_model(model,test_loader):
    model.eval()
    all_pred = []
    all_y = []
    for data in test_loader:
        graph_node_features, graph_edge_index, user_feats = Variable(data[0]).to(device), Variable(data[1]).to(device), Variable(data[3]).to(device)
        merged_tree_edge_index, merged_tree_feature, labels = Variable(data[4]).to(device), Variable(data[5]).to(device), Variable(data[6]).to(device)
        indx = data[7].to(device)
        output = model(user_feats, graph_node_features, graph_edge_index, merged_tree_feature, merged_tree_edge_index, indx)
        output = F.log_softmax(output, dim=1)
        # print('output', output)
        _, y_pred = output.max(dim=1)
        all_pred += y_pred
        all_y += labels
    Acc_all, Acc1, Prec1, Recll1, F1,Acc2, Prec2, Recll2, F2,Acc3, Prec3, Recll3, F3,Acc4, Prec4, Recll4, F4 = evaluation4class(all_pred, all_y)
    print('testing accuracy >>: ', Acc_all)
    return Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4

def graph_dict_add_edge(graph, user, tweet):
    assert int(user) >= SOURCE_TWEET_NUM
    if int(user) in graph[str(tweet)] and \
            int(tweet) in graph[str(user)]:
        print('{}-{} edge exists!'.format(user, tweet))
        return graph
    else:
        graph_new = copy.deepcopy(graph)
        graph_new[str(user)].append(int(tweet))
        graph_new[str(tweet)].append(int(user))
        return graph_new

# def graph_index_add(graph, bad_user_idx, target_tweet_idx):
#     user_reindex = user_map[str(bad_user_idx)]
#     tweet_reindex = loss_tweet_map[str(target_tweet_idx)]
#     print(graph)
#     graph = torch.cat((graph, torch.tensor([[user_reindex, tweet_reindex], [tweet_reindex, user_reindex]]).to(device)), 1)
#     print(graph)

def alter_graph(graph, bad_user_set, current_graph_index):
    """
    alter graph using Beam Search Algorithm
    {(user1,tweet1): loss1, (user2,tweet2): loss2 ...}
    """
    # graph_trace_cluster = graph_trace_list[-1]  # cluster will be [(a,b), (c,d) ...]  (K pairs)
    # for edge in graph_trace_cluster:
    correct_label_last, label_score_last, _ = calc_target_output(current_graph_index, None, None)
    # tweet_set = list(range(SOURCE_TWEET_NUM))

    best_label_attack = 0
    best_value_attack = 0
    label_attack = 0
    value_attack = 0
    label_chosen_edge = None
    value_chosen_edge = None
    best_graph_adj = None
    best_graph_index_map = None

    print(correct_label_last)
    print(label_score_last)

    for bad_user in tqdm(bad_user_set):
        for tweet in tqdm(test_indices):   # all test tweet indices
            if int(bad_user) not in graph[str(tweet)] and int(tweet) not in graph[str(bad_user)]:
                new_graph = graph_dict_add_edge(graph, bad_user, tweet)
                correct_label2, label_score2, graph_index_new = calc_target_output(bad_user, tweet)
                label_attack = correct_label_last - correct_label2
                value_attack = label_score_last - label_score2

                if best_label_attack < label_attack:
                    best_label_attack = label_attack
                    label_chosen_edge = (bad_user, tweet)
                    best_graph_adj = new_graph
                    best_graph_index_map = graph_index_new
                elif best_value_attack < value_attack:
                    best_value_attack = value_attack
                    value_chosen_edge = (bad_user, tweet)
                    best_graph_adj = new_graph
                    best_graph_index_map = graph_index_new
        print(correct_label2)
        print(label_score2)
    if best_label_attack > 0:
        chosen_edge = label_chosen_edge
        return chosen_edge, best_label_attack, None, best_graph_index_map, best_graph_adj
    else:
        chosen_edge = value_chosen_edge
        return chosen_edge, None, best_value_attack, best_graph_index_map, best_graph_adj


@torch.no_grad()
def calc_target_output(current_graph_index, bad_user_idx, target_tweet_idx):
    model.eval()

    all_pred = []
    all_labels = []
    all_y_pred = []
    if (not bad_user_idx) and (not target_tweet_idx):
        graph_new = current_graph_index
    else:
        graph_new = copy.deepcopy(current_graph_index)
        user_reindex = user_map[str(bad_user_idx)]
        tweet_reindex = loss_tweet_map[str(target_tweet_idx)]
        graph_new = torch.cat((graph_new, torch.tensor([[user_reindex, tweet_reindex], [tweet_reindex, user_reindex]]).to(device)), 1)

    output = model(user_feats, graph_node_features, graph_new, merged_tree_feature, merged_tree_edge_index, indx)
    output = F.softmax(output, dim=1)
    all_pred = torch.Tensor.cpu(output).detach().numpy()
    all_labels = labels
    _, y_pred = output.max(dim=1)
    all_y_pred = y_pred
    
    rumor_score = 0
    correct = 0
    target_node_list = []
    for i in range(len(all_labels)):
        if all_labels[i] == 1 and all_y_pred[i] == 1:
            correct += 1
        if all_labels[i] == 1:
            rumor_score += all_pred[i,1]
    return correct, rumor_score, graph_new


if __name__ == '__main__':
    ###
    print("-"*89)
    print('start model training now!!')
    print("-"*89)
    torch.cuda.set_device(args.gpu)
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if (not args.attack) and args.train:
        # load vocab of tweets
        TWEETS_WORD_FILE = pth.join('../load_data15_1473/tweets_words_mapping.json')
        tweets_word_map, _ = _load_word2index(TWEETS_WORD_FILE)
        glove_file = '../glove/glove.twitter.27B.{}d.txt'.format(args.embed_dim)
        embed_dim = args.embed_dim
        graph_connection_path = '../load_data15_1473/graph_connections2.json'
        adjdict = util.read_dict_from_json(pth.join(graph_connection_path))
        print("--load pretrain embedding now--")
        tweet_embedding_matrix = _get_embedding(glove_file, tweets_word_map, embed_dim)
        model = Net(args, tweet_embedding_matrix) # load model

    if args.load_ckpt and args.attack:
        TWEETS_WORD_FILE = pth.join('../load_data15_1473/tweets_words_mapping.json')
        tweets_word_map, _ = _load_word2index(TWEETS_WORD_FILE)
        embed_dim = args.embed_dim
        tweet_embedding_matrix = torch.FloatTensor(np.zeros((len(tweets_word_map), embed_dim)))
        model = Net(args, tweet_embedding_matrix)
        graph_connection_path = '../load_data15_1473/graph_connections2.json'
        graph_dict = util.read_dict_from_json(pth.join(graph_connection_path))
        adjdict = graph_dict
        
        bad_user_path = pth.join('../attack15/bad_user_score40.json')
        bad_users_dict = util.read_dict_from_json(bad_user_path)
        bad_users = list(bad_users_dict.keys())
        bad_users = [i.split('.')[0] for i in bad_users]
        all_user_path = pth.join('../load_data15_1473/filtered_user_profile2_encode2.csv')
        df = pd.read_csv(all_user_path, lineterminator='\n')
        node_id_list = df['node_id'].tolist()
        user_id_list = df['user_id'].tolist()
        node_id_list = [int(i) for i in node_id_list]
        user_id_list = [str(i) for i in user_id_list]
        user2node_dict = dict(zip(user_id_list, node_id_list))
        bad_user_set = [user2node_dict[i] for i in bad_users]
        # take record of add trace
        graph_trace_list = []    # take record of the best K add for each step (global list)
    
    model.to(device)
    train_indices, test_indices = data_split()
    train_loader, test_loader = get_dataloader(args.batch_size, train_indices, test_indices, adjdict)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    loss_fun = nn.CrossEntropyLoss()
    global_step = 0
    curr_epoch = 0
    # get all info from test dataloader
    for data in test_loader:
        graph_node_features, graph_edge_index, user_feats = Variable(data[0]).to(device), Variable(data[1]).to(device), Variable(data[3]).to(device)
        merged_tree_edge_index, merged_tree_feature, labels = Variable(data[4]).to(device), Variable(data[5]).to(device), Variable(data[6]).to(device)
        indx = data[7].to(device)
        loss_tweet_map, user_map, no_loss_tweet_map = data[8], data[9], data[10]

    # print(bad_user_set)  
    bad_user_set = [usr_id for usr_id in bad_user_set if str(usr_id) in user_map.keys()]
    # print(bad_user_set)  

    if args.load_ckpt and args.attack:
        model, optimizer, global_step, curr_epoch = _load_checkpoint()
        # train_loader, test_loader = get_dataloader(args.batch_size, train_indices, test_indices, adjdict)
        print("***loading checkpoint successfully***")
        print("[checkpoint current epoch: {} and step: {}]".format(curr_epoch, global_step))
        # _testing_model(model, test_loader)
        
        # label_correct, pred_score = calc_target_output(graph_dict, bad_user_set)
        # label_score, pred_score = calc_attack(old_graph_dict, new_graph_dict)
        # correct_label_last, label_score_last = calc_target_output(None, None)

        current_graph_index = copy.deepcopy(graph_edge_index)
        edge_list = []
        # for loop for insert edge
        for k in range(10):
            edge, label, value, best_graph_index_map, best_graph_dict = alter_graph(graph_dict, bad_user_set, current_graph_index)
            edge_list.append(edge)
            # switch map and adj matrix
            current_graph_index = best_graph_index_map
            graph_dict = best_graph_dict

            print(edge)
            print(label)
            print(value)
        
    if (not args.attack) and args.train:
        _train_model(train_indices, test_indices, model)
    # else:
    #     print("*****testing model now*****")
    #     accy = _testing_model()       
    