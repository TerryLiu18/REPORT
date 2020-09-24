'''
Huajie Shao 2020/08/19
fun: build net for models
'''

import torch
import torch.nn as nn
# import torch.nn.functional as F
# from userModel import UserEncoder
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

class TreeGCN(nn.Module):
    """GAT for tree"""
    def __init__(self, args, tweet_embedding_matrix, text_input_size=100, hidden_size1=100, hidden_size2=100,hidden_size3=100):
        super(TreeGCN, self).__init__()
        # print('batch size', args.batch_size)
        self.batch_size = args.batch_size #batch_size
        self.tweet_out_size =  args.tweet_embed_size
        self.num_layer = 1
        self.dropout = args.dropout

        self.tweets_embedding = nn.Embedding.from_pretrained(tweet_embedding_matrix, freeze=False)
        self.GRU = nn.GRU(text_input_size, self.tweet_out_size, self.num_layer) ##embedding size, hidden size
        self.drop_layer = nn.Dropout(self.dropout)
        # self.conv1 = GATConv(self.tweet_out_size, hidden_size1, heads=8, dropout=self.dropout)
        self.conv1 = GCNConv(self.tweet_out_size, hidden_size1)
        ## second layer
        # self.conv2 = GATConv(8*hidden_size1, hidden_size2, heads=1, concat=False, dropout=self.dropout)
        self.conv2 = GCNConv(self.tweet_out_size+hidden_size1, hidden_size2)

    def forward(self, merged_tree_feature, merged_tree_edge_index, indices):
        embed_tree_feature = self.tweets_embedding(merged_tree_feature)
        embed_tree_feature = embed_tree_feature.permute(1,0,2)  #seq * batch * hidden
        state_size = [self.num_layer, embed_tree_feature.size(1), self.tweet_out_size]
        h0 = Variable(torch.randn(state_size)).to(device)
        out_nodes, hn = self.GRU(embed_tree_feature, h0)
        x_input = hn[-1,:,:] ## last layer
        ## replace the embedding of root source
        # if x_input.size(0) < self.batch_size:
        #     self.batch_size = x_input.size(0)
        #     print("lower dim for input size")
        x = x_input
        x1 = copy.copy(x.float())
        x = self.conv1(x, merged_tree_edge_index)
        x2 = copy.copy(x)
        rootindex = indices
        root_extend = torch.zeros(len(indices), x1.size(1)).to(device)   
        batch_size = max(indices) + 1
        for num_batch in range(batch_size):
            index = (torch.eq(indices, num_batch))
            root_extend[index] = x1[num_batch]
        x = torch.cat((x,root_extend), 1)
        
        x = F.relu(x)
        x = self.drop_layer(x)
        x = self.conv2(x, merged_tree_edge_index)
        x = F.relu(x)
        root_extend = torch.zeros(len(indices), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = torch.eq(indices, num_batch)
            root_extend[index] = x2[num_batch]
        x = torch.cat((x,root_extend), 1)
        # print(x.shape)
        # print(indices.shape)
        # x= scatter_mean(x, indices, dim=0)
        return x
        

class Net(nn.Module):
    """docstring for ClassName"""
    def __init__(self, args, tweet_embedding_matrix, user_embedding_matrix):
        super(Net, self).__init__()
        self.hidden_size1 = args.hidden_size1
        self.hidden_size2 = args.hidden_size2
        self.hidden_size3 = args.hidden_size3

        self.graph_hidden_size1 = args.graph_hidden_size1
        self.graph_hidden_size2 = args.graph_hidden_size2
        self.graph_hidden_size3 = args.graph_hidden_size3

        self.num_layer = 2
        self.text_input_size = args.embed_dim
        self.batch_size = args.batch_size
        self.args = args
        ## model
        self.TreeGCN = TreeGCN(self.args, tweet_embedding_matrix, self.text_input_size, self.hidden_size1, self.hidden_size2, self.hidden_size3)
        self.fc = nn.Linear(self.hidden_size2 + self.hidden_size1, 4)
        

    def forward(self, user_text, user_feats, graph_node_features, graph_edge_index, merged_tree_feature, merged_tree_edge_index, indices):
        ##model(user_text, user_feats, graph_node_features, graph_edge_index, merged_tree_feature, merged_tree_edge_index,indx)
        child_embed = self.TreeGCN(merged_tree_feature, merged_tree_edge_index, indices)
        mean_embed = scatter_mean(child_embed, indices, dim=0) ## nodes * embedding
        out_y  = self.fc(mean_embed)
        
        return out_y
