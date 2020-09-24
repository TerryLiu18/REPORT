
"""
tianrui liu 2020/09/11
fun: tuning models
"""

import torch
import torch.nn as nn
# import torch.nn.functional as F
# from userModel import UserEncoder
from time import sleep
from torch_geometric.nn import GATConv, GCNConv
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
        # user_embed = torch.cat((hn,user_feats), 1)  # 2 * embed = 100 dim
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
        ## user embedding
        self.userEmbed = UserEncoder(self.freeze, self.user_feat_size, self.user_out_size)
        self.num_layer = 2
        self.batch_size = args.batch_size

        ## tweet embedding
        self.tweets_embedding = nn.Embedding.from_pretrained(tweet_embedding_matrix, freeze=self.freeze)
        self.GRU = nn.GRU(self.text_input_size, self.tweet_out_size, self.num_layer, dropout=0.2)

        self.conv1 = GCNConv(self.tweet_out_size, graph_num_hidden1)
        self.conv2 = GCNConv(graph_num_hidden1, graph_num_hidden2)
        
    def forward(self, user_feats, graph_node_features, graph_edge_index, tree_embed, indices):
        batch_size = max(indices) + 1
        user_embedding = self.userEmbed(user_feats) ## batch * hidden
        tweet_embed = graph_node_features
        ## tweet node embedding
        tweet_embed = self.tweets_embedding(graph_node_features)
        tweet_embed = tweet_embed.permute(1,0,2) ##seq * batch * hidden
        state_size = [self.num_layer, tweet_embed.size(1), self.tweet_out_size]
        h0 = Variable(torch.randn(state_size)).to(device)
        out_tweet, hn = self.GRU(tweet_embed, h0)
        hn = hn[-1,:,:] ## last layer
        
        ## concat graph embedding and user embedding
        x_input = torch.cat([hn[:batch_size], user_embedding, hn[batch_size:]], 0)  #root -->user-->no loss tweet
        for i in range(batch_size):
            x_input[i, :] = tree_embed[i, :]
        x = self.conv1(x_input, graph_edge_index)
        x = F.elu(x)
        x = self.conv2(x, graph_edge_index)
        x = F.elu(x)
        return x[:batch_size, :]


class TreeGCN(nn.Module):
    """GAT for tree"""
    def __init__(self, args, tweet_embedding_matrix, direction, text_input_size=100, hidden_size1=100, hidden_size2=100):
        super(TreeGCN, self).__init__()
        # print('batch size', args.batch_size)
        self.direction = direction
        self.batch_size = args.batch_size #batch_size
        self.tweet_out_size =  args.tweet_embed_size
        self.num_layer = 2
        self.dropout = args.dropout
        self.direction = direction

        self.tweets_embedding = nn.Embedding.from_pretrained(tweet_embedding_matrix, freeze=False)
        self.GRU = nn.GRU(text_input_size, self.tweet_out_size, self.num_layer) ##embedding size, hidden size
        self.conv1 = GCNConv(self.tweet_out_size, hidden_size1)
        self.conv2 = GCNConv(self.tweet_out_size+hidden_size1, hidden_size2)

    def forward(self, merged_tree_feature, merged_tree_edge_index, indices):
        embed_tree_feature = self.tweets_embedding(merged_tree_feature)
        embed_tree_feature = embed_tree_feature.permute(1,0,2)  #seq * batch * hidden
        state_size = [self.num_layer, embed_tree_feature.size(1), self.tweet_out_size]
        h0 = Variable(torch.randn(state_size)).to(device)
        out_nodes, hn = self.GRU(embed_tree_feature, h0)
        x = hn[-1,:,:] ## last layer

        # reverse the edge_index matrix
        if self.direction == 'bu':
            merged_tree_edge_index = merged_tree_edge_index[[1, 0]] 

        # for i in range(max(indices) + 1):  #batch size
        #     x_input[i,:] = user_root_embed[i,:]
        x1 = copy.copy(x.float())
        x = self.conv1(x, merged_tree_edge_index)
        x2 = copy.copy(x)
        root_extend = torch.zeros(len(indices), x1.size(1)).to(device)   
        batch_size = max(indices) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(indices, num_batch))
            root_extend[index] = x1[num_batch]
        x = torch.cat((x,root_extend), 1)

        x = F.elu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, merged_tree_edge_index)
        x = F.elu(x)

        x = scatter_mean(x, indices, dim=0) ## nodes * embedding
        return x


class Net(nn.Module):
    """docstring for ClassName"""
    def __init__(self, args, tweet_embedding_matrix):
        super(Net, self).__init__()
        self.tree_hidden_size1 = args.tree_hidden_size1
        self.tree_hidden_size2 = args.tree_hidden_size2

        self.graph_hidden_size1 = args.graph_hidden_size1
        self.graph_hidden_size2 = args.graph_hidden_size2

        self.direction = args.direction
        self.num_layer = 2
        self.text_input_size = args.embed_dim
        self.batch_size = args.batch_size
        self.args = args
        ## model
        self.graphGCN = GraphGCN(self.args, tweet_embedding_matrix, self.graph_hidden_size1, self.graph_hidden_size2)
        self.TreeGCN = TreeGCN(self.args, tweet_embedding_matrix, self.direction, self.text_input_size, self.tree_hidden_size1, self.tree_hidden_size2)
        self.fc = nn.Linear(self.graph_hidden_size2, 4)   


    def forward(self, user_feats, graph_node_features, graph_edge_index, merged_tree_feature, merged_tree_edge_index, indices):
        tree_embed = self.TreeGCN(merged_tree_feature, merged_tree_edge_index, indices)
        graph_root_embed = self.graphGCN(user_feats, graph_node_features, graph_edge_index, tree_embed, indices)
        out_y  = self.fc(graph_root_embed)
        return out_y
