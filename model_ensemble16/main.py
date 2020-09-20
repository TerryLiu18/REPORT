'''
tianrui liu @2020/09/11
Fun: tune the model
'''
import os
import json
import argparse
import random
import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from datetime import datetime
from torch import nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from util import str2bool
from dataset16 import get_dataloader
from models_ensemble import Net
from GloveEmbed import _get_embedding
from early_stopping import EarlyStopping

## add args
parser = argparse.ArgumentParser(description='GAT for fake news detection')
parser.add_argument('--train', default=True, type=str2bool, help='train or traverse')
parser.add_argument('--patience', default=4, type=int, help='how long to wait after last time validation loss improved')
parser.add_argument('--freeze', default=False, type=str2bool, help='embedding freeze or not')
parser.add_argument('--load_ckpt', default=False, type=str2bool, help='load checkpoint')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--epoches', default=90, type=int, help='maximum training epoches')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--dropout', default=0.3, type=float, help='drop out rate')
parser.add_argument('--weight_decay', default=1e-6, type=float, help='weight decay')
parser.add_argument('--embed_dim', default=100, type=int, help='pretrain embed size')
parser.add_argument('--tweet_embed_size', default=100, type=int, help='tweet embed size')
parser.add_argument('--tree_hidden_size1', default=100, type=int, help='hidden size for TreeGCN')
parser.add_argument('--tree_hidden_size2', default=100, type=int, help='hidden size for TreeGCN')

parser.add_argument('--graph_hidden_size1', default=100, type=int, help='hidden size for GraphGCN')
parser.add_argument('--graph_hidden_size2', default=100, type=int, help='hidden size for GraphGCN')

parser.add_argument('--direction', default='td', type=str, help='tree direction: topdown(td)/bottomup(bu)')
parser.add_argument('--user_feat_size', default=9, type=int, help = 'user features')
# parser.add_argument('--text_input_size', default=200, type=int, help = 'tweets and description input size')
parser.add_argument('--user_out_size', default=100, type=int, help = 'user description embed size')
# parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
parser.add_argument('--ckpt_name', default='last', type=str, help='load previous checkpoint. insert checkpoint filename')
args = parser.parse_args()
args.user_out_size = int(args.tweet_embed_size/2)
# args.graph_hidden_size2 = args.tweet_embed_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: {}{} ".format(device, str(args.gpu)))

def _load_word2index(word_file):
    with open(word_file) as jsonfile:
        word_map = json.load(jsonfile)

    vocab_size = len(word_map)
    return word_map, vocab_size

def _save_checkpoint(model, global_step, epoch):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    save_dir = os.path.join(args.ckpt_dir, args.ckpt_name)
    checkpoint_dict = {'model_state': model.state_dict(), 'optim_state': optimizer.state_dict(), 'global_step':global_step, 'curr_epoch': epoch}
    torch.save(checkpoint_dict, save_dir)

def _load_checkpoint(global_step, curr_epoch):
    ckp_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    if os.path.isfile(ckp_path):
        checkpoint = torch.load(ckp_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optim_state'])
        global_step = checkpoint['global_step']
        curr_epoch = checkpoint['curr_epoch']
    return model, optimizer, global_step, curr_epoch


# def _adjust_learning_rate(learning_rate, optim, epoch):
#     """Sets the learning rate to the initial LR decayed by 1/10 every args.lr epochs"""
#     lr = learning_rate
#     if 10 < epoch <= 20:
#         lr = 0.0001
#     elif 20 < epoch < 40:
#         lr = 0.0008
#     else:
#         lr = 0.0002
#     for param_group in optim.param_groups:
#         param_group['lr'] = lr

def adjust_learning_rate(optim, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optim.param_groups:
        param_group['lr'] = lr

def _compute_accy_count(y_pred, y_labels):
    return 1.0*y_pred.eq(y_labels).sum().item()
    
def _compute_accuracy(y_pred, y_labels):
    return 1.0*y_pred.eq(y_labels).sum().item()/y_labels.size(0)
    
# def _train_model(global_step, train_loader, model):
def _train_model():
    # time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # os.makedirs('logs/', exist_ok=True)
    # log_file = 'logs/' + time + 'args.log'
    # fw_log = open(log_file, "w")
    # json.dump(args.__dict__, fw_log, indent=2)
    # fw_log.write('\n')

    model.train()
    global_step = 0
    max_iter = len(train_loader) * args.epoches
    writer = SummaryWriter()
    pbar = tqdm(total=max_iter)
    pbar.update(global_step)

    early_stopping = EarlyStopping(args.patience)
    for epoch in range(curr_epoch, args.epoches):
        # adjust_learning_rate(args.lr, optimizer, epoch)
        adjust_learning_rate(optimizer, epoch)
        train_accy = []
        for data in train_loader:
            graph_node_features, graph_edge_index, user_text, user_feats = Variable(data[0]).to(device), \
                        Variable(data[1]).to(device), Variable(data[2]).to(device), Variable(data[3]).to(device)
            merged_tree_edge_index, merged_tree_feature, labels = Variable(data[4]).to(device), Variable(data[5]).to(device), Variable(data[6]).to(device)
            indx = data[7].to(device)
            global_step += 1
            pbar.update(1)
            optimizer.zero_grad()
            output = model(user_text, user_feats, graph_node_features, graph_edge_index, merged_tree_feature, merged_tree_edge_index, indx)
            print(output)
            print("label: ",labels)
            loss = loss_fun(output, labels)
            loss.backward()
            optimizer.step()
            ## compute accuracy
            _, y_pred = output.max(dim=1)
            accy = _compute_accuracy(y_pred, labels)
            train_accy.append(accy)
            if global_step % 10 == 0:
                print("epoch: {} global step: {} loss: {:.5f} accuracy: {:.5f}"\
                        .format(epoch+1, global_step, loss.item(), accy))
            #     fw_log.write("epoch: {} global step: {} loss: {:.5f} train accuracy: {:.5f}\n"\
            #             .format(epoch+1, global_step, loss.item(), accy))
            # writer.add_scalar('Loss/train', loss, global_step)
            # if global_step % 10 == 0:
            #     writer.flush()
        ## save checkpoint for best accuracy epoch
        # if (epoch+1) % 1 == 0:
        #     _save_checkpoint(model, global_step, epoch)
        if (epoch+1) % 2 == 0:
            # test_accy = _testing_model(test_loader,model)
            test_accy = _testing_model()
            early_stopping(test_accy)
            if early_stopping.early_stop:
                _save_checkpoint(model, global_step, epoch)
                print('early stop!')
                break
            # fw_log.write('epoch: {} testing accuracy: {:4f}\n'.format(epoch+1, test_accy))
            # fw_log.flush()
            model.train()
    print("BEST Accuracy: {:.4f}".format(early_stopping.best_accs))
    # fw_log.close()
    writer.close()
    
# def _testing_model(test_loader, model):
def _testing_model():
    model.eval()
    sum_count = 0
    test_count = 0
    for data in test_loader:
        graph_node_features, graph_edge_index, user_text, user_feats = Variable(data[0]).to(device), \
                        Variable(data[1]).to(device), Variable(data[2]).to(device), Variable(data[3]).to(device)
        merged_tree_edge_index, merged_tree_feature, labels = Variable(data[4]).to(device), Variable(data[5]).to(device), Variable(data[6]).to(device)
        indx = data[7].to(device)
        output = model(user_text, user_feats, graph_node_features, graph_edge_index, merged_tree_feature, merged_tree_edge_index, indx)
        output = F.log_softmax(output, dim=1)
        # _, y_pred = torch.max(output.data, 1)
        _, y_pred = output.max(dim=1)
        count = _compute_accy_count(y_pred, labels)
        sum_count += count
        test_count += len(labels)
    test_accy = 1.0 * sum_count / test_count
    print('testing accuracy >>: ', test_accy)
    return test_accy


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
    ## load vocab of user descriptions
    
    USER_PROFILE_WORD = '../load_data16/filtered_user_profile_words_mapping.json'
    user_word_map, _ = _load_word2index(USER_PROFILE_WORD)
    
    ## load vocab of tweets

    TWEETS_WORD_FILE = '../load_data16/tweets_words_mapping.json'
    tweets_word_map, _ = _load_word2index(TWEETS_WORD_FILE)
    # glove_file = '../glove/glove.twitter.27B.{}d.txt'.format(args.embed_dim)
    glove_file = os.path.join('../pretrained_dataset/glove.twitter.27B/glove.twitter.27B.{}d.txt'.format(args.embed_dim))
    embed_dim = args.embed_dim

    print("--load pretrain embedding now--")
    tweet_embedding_matrix = _get_embedding(glove_file, tweets_word_map, embed_dim)
    user_embedding_matrix = 0  # currently not use
    # user_embedding_matrix = _get_embedding(glove_file, user_word_map, embed_dim)
    print("***end of load pretrain embedding***")
    ## data loader
    train_loader, test_loader = get_dataloader(args.batch_size, args.seed)
    ## load model
    model = Net(args, tweet_embedding_matrix, user_embedding_matrix, direction=args.direction)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    loss_fun = nn.CrossEntropyLoss()
    global_step = 0
    curr_epoch = 0
    if args.load_ckpt:
        model, optimizer, global_step, curr_epoch = _load_checkpoint(global_step,curr_epoch)
        print("***loading checkpoint successfully***")
        print("[checkpoint current epoch: {} and step: {}]".format(curr_epoch, global_step))
    print("start training model now")  # train model
    if args.train:
        _train_model()
    else:
        print("*****testing model now*****")
        accy = _testing_model()       
