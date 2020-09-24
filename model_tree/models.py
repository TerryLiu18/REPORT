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
from torch.nn.init import xavier_uniform
import torch.nn.functional as F

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
    def __init__(self, user_embedding_matrix, freeze, user_feat_size=9, user_text_input_size=100, user_out_size=50):
        super(UserEncoder, self).__init__()
        self.user_out_size = user_out_size
        self.num_layer = 2
        self.freeze = freeze
        # self.user_disc_embedding = nn.Embedding(vocab_size, embed_dim)
        self.user_disc_embedding = nn.Embedding.from_pretrained(user_embedding_matrix, freeze=self.freeze)
        ## gru for text encoding
        self.GRU = nn.GRU(user_text_input_size, self.user_out_size, self.num_layer,dropout=0.2)
        
        ## other features
        self.fc = nn.Sequential(
                    nn.Linear(user_feat_size, 100),
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(100, self.user_out_size*2),
                    # nn.Dropout(p=0.2)
                    )
        self.fc.apply(init_weights)

    def forward(self, user_text, user_feats):
        # x = self.user_disc_embedding(user_text) ## (batch_size, seq_length, embed_dim)
        # state_size = [self.num_layer, user_text.size(0), self.user_out_size]  # batch * out_size
        # h0 = Variable(torch.randn(state_size)).to(device)  #num_layers, batch, hidden_size
        # ## text and features embedding
        # x = x.permute(1,0,2)
        # out_text, hn = self.GRU(x, h0) # input of shape (seq_len, batch, input_size)
        # hn = hn[-1,:,:] ## last layer
        user_feats = self.fc(user_feats)  # batch * dim
        # user_embed = torch.cat((hn,user_feats), 1)  # 2 * embed = 100 dim
        
        return user_feats


class graphGAT(nn.Module):
    '''fun: Graph for user and source tweets'''
    def __init__(self, args, tweet_embedding_matrix, user_embedding_matrix, graph_num_hidden1=64, graph_num_hidden2=100):
        super(graphGAT, self).__init__()
        self.user_feat_size = args.user_feat_size
        self.text_input_size = args.embed_dim
        self.user_out_size = args.user_out_size
        self.tweet_out_size = args.tweet_embed_size
        self.dropout = args.dropout
        self.freeze = args.freeze
        ## user embedding
        self.userEmbed = UserEncoder(user_embedding_matrix, self.freeze, self.user_feat_size, self.text_input_size, self.user_out_size)
        self.num_layer = 2
        self.batch_size = args.batch_size

        ## tweet embedding
        self.tweets_embedding = nn.Embedding.from_pretrained(tweet_embedding_matrix, freeze=self.freeze)
        self.GRU = nn.GRU(self.text_input_size, self.tweet_out_size, self.num_layer, dropout=0.2)

        self.conv1 = GATConv(self.tweet_out_size, graph_num_hidden1, heads=8, dropout=self.dropout)
        self.conv2 = GATConv(8 * graph_num_hidden1, graph_num_hidden2, heads=1, concat=False,dropout=self.dropout)
        
    def forward(self, user_text, user_feats, graph_node_features, graph_edge_index):
        user_embedding = self.userEmbed(user_text, user_feats) ## batch * hidden
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
        x = F.relu(x)
        x = self.conv2(x, graph_edge_index)
        x = F.relu(x)
        return x


class TreeGAT(nn.Module):
    """GAT for tree"""
    def __init__(self, args, tweet_embedding_matrix, text_input_size=100, hidden_size1=100, hidden_size2=100,hidden_size3=100):
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
        self.conv2 = GATConv(8*hidden_size1, hidden_size2, heads=1, concat=False, dropout=self.dropout)

    def forward(self, merged_tree_feature, merged_tree_edge_index,user_root_embed):
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
        for i in range(self.batch_size):  #batch size
            x_input[i,:] = user_root_embed[i,:]
        
        x = self.conv1(x_input, merged_tree_edge_index)
        x = F.relu(x)
        x = self.conv2(x, merged_tree_edge_index)
        x = F.relu(x)
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
        self.graphGAT = graphGAT(self.args, tweet_embedding_matrix, user_embedding_matrix, self.graph_hidden_size1, self.graph_hidden_size2)
        self.TreeGAT = TreeGAT(self.args, tweet_embedding_matrix, self.text_input_size, self.hidden_size1, self.hidden_size2,self.hidden_size3)
        self.fc = nn.Linear(self.hidden_size2, 4)
        

    def forward(self, user_text, user_feats, graph_node_features, graph_edge_index, merged_tree_feature, merged_tree_edge_index, indices):
        ##model(user_text, user_feats, graph_node_features, graph_edge_index, merged_tree_feature, merged_tree_edge_index,indx)
        user_root_embed = self.graphGAT(user_text, user_feats, graph_node_features, graph_edge_index)
        child_embed = self.TreeGAT(merged_tree_feature, merged_tree_edge_index, user_root_embed)
        mean_embed = scatter_mean(child_embed, indices, dim=0) ## nodes * embedding
        out_y  = self.fc(mean_embed)
        
        return out_y