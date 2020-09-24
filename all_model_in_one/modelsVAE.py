'''
Huajie Shao 2020/08/19
fun: build net for models
'''

import torch
import torch.nn as nn
# import torch.nn.functional as F
# from userModel import UserEncoder
from torch_geometric.nn import GATConv
from torch_scatter import scatter_mean
from torch.autograd import Variable
from userVAE import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class graphGAT(nn.Module):
    '''fun: Graph for user and source tweets'''
    def __init__(self, args, tweet_embedding_matrix, user_embedding_matrix, graph_num_hidden1=64, graph_num_hidden2=100):
        super(graphGAT, self).__init__()
        self.user_feat_size = args.user_feat_size
        self.text_input_size = args.embed_dim
        self.user_out_size = args.user_out_size
        self.tweet_out_size = args.tweet_embed_size
        self.dropout = args.dropout
        self.z_dim = args.z_dim
        ## user embedding using VAE
        self.userEmbed = VAE(self.user_feat_size, self.z_dim)
        self.num_layer = 2
        self.batch_size = args.batch_size

        ## tweet embedding
        self.tweets_embedding = nn.Embedding.from_pretrained(tweet_embedding_matrix, freeze=False)
        self.GRU = nn.GRU(self.text_input_size, self.tweet_out_size, self.num_layer)

        self.conv1 = GATConv(self.tweet_out_size, graph_num_hidden1, heads=8, dropout=self.dropout)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * graph_num_hidden1, graph_num_hidden2, heads=1, concat=False, dropout=self.dropout)
        
    def forward(self, user_text, user_feats, graph_node_features, graph_edge_index):
        user_embedding, kl_loss, rec_loss = self.userEmbed(user_feats) ## batch * hidden
        ## tweet node embedding
        tweet_embed = self.tweets_embedding(graph_node_features)
        tweet_embed = tweet_embed.permute(1,0,2) ##seq * batch * hidden
        state_size = [self.num_layer, tweet_embed.size(1), self.tweet_out_size]
        h0 = Variable(torch.randn(state_size)).to(device)
        out_tweet, hn = self.GRU(tweet_embed, h0)
        hn = hn[-1,:,:] ## last layer
        
        ## concat graph embedding and user embedding
        x_input = torch.cat([hn[:self.batch_size], user_embedding, hn[self.batch_size:]], 0)  #root -->user-->no loss tweet
        x = self.conv1(x_input, graph_edge_index)
        x = self.conv2(x, graph_edge_index)
        return x, kl_loss, rec_loss


class TreeGAT(nn.Module):
    """GAT for tree"""
    def __init__(self, args, tweet_embedding_matrix, text_input_size=100, hidden_size1=100, hidden_size2=100):
        super(TreeGAT, self).__init__()
        # print('batch size', args.batch_size)
        self.batch_size = args.batch_size #batch_size
        self.tweet_out_size =  args.tweet_embed_size
        self.num_layer = 2
        self.dropout = args.dropout

        self.tweets_embedding = nn.Embedding.from_pretrained(tweet_embedding_matrix, freeze=False)
        self.GRU = nn.GRU(text_input_size, self.tweet_out_size, self.num_layer) ##embedding size, hidden size
        self.conv1 = GATConv(self.tweet_out_size, hidden_size1, heads=8, dropout=self.dropout)
        ## second layer
        self.conv2 = GATConv(8 * hidden_size1, hidden_size2, heads=1, concat=False, dropout=self.dropout)

        
    def forward(self, merged_tree_feature, merged_tree_edge_index,user_root_embed):
        embed_tree_feature = self.tweets_embedding(merged_tree_feature)
        embed_tree_feature = embed_tree_feature.permute(1,0,2)  #seq * batch * hidden
        state_size = [self.num_layer, embed_tree_feature.size(1), self.tweet_out_size]
        h0 = Variable(torch.randn(state_size)).to(device)
        out_nodes, hn = self.GRU(embed_tree_feature, h0)
        x_input = hn[-1,:,:] ## last layer
        
        ## replace the embedding of root source
        for i in range(self.batch_size):
            x_input[i,:] = user_root_embed[i,:]
        
        x = self.conv1(x_input, merged_tree_edge_index)
        x = self.conv2(x, merged_tree_edge_index)
        return x
        

class Net(nn.Module):
    """docstring for ClassName"""
    def __init__(self, args, tweet_embedding_matrix, user_embedding_matrix):
        super(Net, self).__init__()
        self.hidden_size1 = args.hidden_size1
        self.hidden_size2 = args.hidden_size2
        self.graph_hidden_size1 = args.graph_hidden_size1
        self.graph_hidden_size2 = args.graph_hidden_size2
        self.num_layer = 2
        self.text_input_size = args.embed_dim
        self.batch_size = args.batch_size
        self.args = args
        ## model
        self.graphGAT = graphGAT(self.args, tweet_embedding_matrix, user_embedding_matrix, self.graph_hidden_size1, self.graph_hidden_size2)
        self.TreeGAT = TreeGAT(self.args, tweet_embedding_matrix, self.text_input_size, self.hidden_size1, self.hidden_size2)
        self.fc = nn.Linear(self.hidden_size2, 3)
        
    def forward(self, user_text, user_feats, graph_node_features, graph_edge_index, merged_tree_feature, merged_tree_edge_index, indices):
        user_root_embed, kl_loss, rec_loss = self.graphGAT(user_text, user_feats, graph_node_features, graph_edge_index)
        child_embed = self.TreeGAT(merged_tree_feature, merged_tree_edge_index, user_root_embed)
        mean_embed = scatter_mean(child_embed, indices, dim=0) ## nodes * embedding
        out_y  = self.fc(mean_embed)
        
        return out_y, kl_loss, rec_loss
        