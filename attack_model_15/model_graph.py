'''
tianrui liu 2020/09/11
fun: polish the model
'''

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean
from torch.autograd import Variable
from torch.nn.init import xavier_uniform
import torch.nn.functional as F
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class UserEncoder(nn.Module):
    """encode user features include description and other counts
    text_input_size: text embedding size
    user_feat_size: dim of input list of [counts]
    """
    def __init__(self, freeze, user_feat_size=9, user_out_size=50):
        super(UserEncoder, self).__init__()
        self.user_out_size = user_out_size
        self.num_layer = 2
        self.freeze = freeze
        
        ## other features
        self.fc = nn.Sequential(
                    nn.Linear(user_feat_size, 100),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(100, self.user_out_size*2),
                    # nn.Dropout(p=0.2)
                    )
        self.fc.apply(init_weights)

    def forward(self, user_feats):
        user_feats = self.fc(user_feats)  # batch * dim
        return user_feats


class GraphGCN(nn.Module):
    '''fun: Graph for user and source tweets'''
    def __init__(self, args, tweet_embedding_matrix, graph_num_hidden1=64, graph_num_hidden2=100):
        super(GraphGCN, self).__init__()
        self.user_feat_size = args.user_feat_size
        self.text_input_size = args.embed_dim
        self.user_out_size = args.user_out_size
        self.tweet_out_size = args.tweet_embed_size
        self.dropout = args.dropout
        self.freeze = args.freeze
        # user embedding
        self.userEmbed = UserEncoder(self.freeze, self.user_feat_size, self.user_out_size)
        self.num_layer = 2

        # tweet embedding
        self.tweets_embedding = nn.Embedding.from_pretrained(tweet_embedding_matrix, freeze=self.freeze)
        self.GRU = nn.GRU(self.text_input_size, self.tweet_out_size, self.num_layer, dropout=self.dropout)

        self.conv1 = GCNConv(self.tweet_out_size, graph_num_hidden1)
        self.conv2 = GCNConv(graph_num_hidden1, graph_num_hidden2)
        

    def forward(self, user_feats, graph_node_features, graph_edge_index, indices):
        batch_size = max(indices) + 1
        user_embedding = self.userEmbed(user_feats) # batch * hidden
        tweet_embed = self.tweets_embedding(graph_node_features) # tweet node embedding
        tweet_embed = tweet_embed.permute(1,0,2) ##seq * batch * hidden
        state_size = [self.num_layer, tweet_embed.size(1), self.tweet_out_size]
        h0 = Variable(torch.zeros(state_size)).to(device)
        out_tweet, hn = self.GRU(tweet_embed, h0)
        hn = hn[-1,:,:]  # last layer
        
        # concat graph embedding and user embedding
        x_input = torch.cat([hn[:batch_size], user_embedding, hn[batch_size:]], 0)  #root -->user-->no loss tweet
        x = self.conv1(x_input, graph_edge_index)
        x = F.elu(x)
        x = self.conv2(x, graph_edge_index)
        x = F.elu(x)
        return x[:batch_size, :]  # choose root (loss_tweet)


class Net(nn.Module):
    """docstring for ClassName"""
    def __init__(self, args, tweet_embedding_matrix):
        super(Net, self).__init__()
        self.direction = args.direction

        self.graph_hidden_size1 = args.graph_hidden_size1
        self.graph_hidden_size2 = args.graph_hidden_size2
        # self.graph_hidden_size3 = args.graph_hidden_size3

        self.num_layer = 2
        self.text_input_size = args.embed_dim
        self.args = args
        ## model
        self.graphGCN = GraphGCN(self.args, tweet_embedding_matrix, self.graph_hidden_size1, self.graph_hidden_size2)
        self.fc = nn.Linear(self.graph_hidden_size2, 4)   

    def forward(self, user_feats, graph_node_features, graph_edge_index, merged_tree_feature, merged_tree_edge_index, indices):
        graph_output = self.graphGCN(user_feats, graph_node_features, graph_edge_index, indices)
        out_y = self.fc(graph_output)
        return out_y
