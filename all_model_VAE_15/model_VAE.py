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
# from userVAE import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def _kl_loss(mu, logvar):
    klds = -0.5*(1 + logvar - mu.pow(2) - torch.exp(logvar))
    total_kld = klds.sum(1).mean(0, True)
    return total_kld


def _decoder_matrix(Z):
    A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
    return A_pred

def _reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

def _loss_fun(graph_edge_index, A_pred):
    ## recon accuracy
    norm = A_pred.shape[0] * A_pred.shape[0] / float((A_pred.shape[0] * A_pred.shape[0] - A_pred.sum()) * 2)
    ## covert from sparse to dense
    # label_size = graph_edge_index.shape[1]  ##size for label = 1
    labels = torch.ones(graph_edge_index.shape[1], dtype=torch.float).to(device)  #gpu
    dense_A = torch.sparse.FloatTensor(graph_edge_index, labels, torch.Size([A_pred.shape[0],A_pred.shape[0]])).to_dense()
    diag = torch.eye(A_pred.shape[0]).to(device)
    A_ground = dense_A + diag
    ## get weight
    pos_weight = float(A_ground.shape[0] * A_ground.shape[0] - A_ground.sum()) / A_ground.sum()
    weight_mask = A_ground.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = pos_weight
    ## recon loss
    rec_loss = norm * F.binary_cross_entropy(A_pred.view(-1), A_ground.view(-1), weight = weight_tensor.to(device))
    # print(A_pred)
    # print("ground truth: ",A_ground)
    return rec_loss

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
        self.z_dim = args.z_dim
        #### user embedding
        self.userEmbed = UserEncoder(self.freeze, self.user_feat_size, self.user_out_size)
        self.fc = nn.Linear(self.z_dim, 100)
        self.num_layer = 2

        # tweet embedding
        self.tweets_embedding = nn.Embedding.from_pretrained(tweet_embedding_matrix, freeze=self.freeze)
        self.GRU = nn.GRU(self.text_input_size, self.tweet_out_size, self.num_layer, dropout=self.dropout)

        self.conv1 = GCNConv(self.tweet_out_size, graph_num_hidden1)
        self.conv_mean = GCNConv(graph_num_hidden1, graph_num_hidden2)
        self.conv_logvar = GCNConv(graph_num_hidden1, graph_num_hidden2)


    def forward(self, user_feats, graph_node_features, graph_edge_index, indices):
        batch_size = max(indices) + 1
        user_embedding = self.userEmbed(user_feats)

        ## tweet results
        tweet_embed = self.tweets_embedding(graph_node_features) # tweet node embedding
        tweet_embed = tweet_embed.permute(1,0,2) ##seq * batch * hidden
        state_size = [self.num_layer, tweet_embed.size(1), self.tweet_out_size]
        h0 = Variable(torch.randn(state_size)).to(device)
        out_tweet, hn = self.GRU(tweet_embed, h0)
        hn = hn[-1,:,:]  # last layer
        
        # concat graph embedding and user embedding
        x_input = torch.cat([hn[:batch_size], user_embedding, hn[batch_size:]], 0)  #root -->user-->no loss tweet
        x = self.conv1(x_input, graph_edge_index)
        x = F.elu(x)
        ##second layer
        mu = self.conv_mean(x, graph_edge_index)
        logvar = self.conv_logvar(x, graph_edge_index)
        mu = F.relu(mu)
        logvar = F.relu(logvar)
        ## get latent Z by reparameter
        Z = _reparametrize(mu, logvar)
        A_pred = _decoder_matrix(Z)
        ## recon accuracy
        rec_loss1 = _loss_fun(graph_edge_index, A_pred)
        kl_loss1 = _kl_loss(mu, logvar)

        return Z[:batch_size, :], kl_loss1, rec_loss1


class TreeGCN(nn.Module):
    """GAT for tree"""
    def __init__(self, args, direction, tweet_embedding_matrix, text_input_size=100, hidden_size1=100, hidden_size2=100):
        super(TreeGCN, self).__init__()
        # print('batch size', args.batch_size)
        self.batch_size = args.batch_size #batch_size
        self.tweet_out_size =  args.tweet_embed_size
        self.num_layer = 2
        self.dropout = args.dropout
        self.direction = direction

        self.tweets_embedding = nn.Embedding.from_pretrained(tweet_embedding_matrix, freeze=False)
        self.GRU = nn.GRU(text_input_size, self.tweet_out_size, self.num_layer) ##embedding size, hidden size
        self.conv1 = GCNConv(self.tweet_out_size, hidden_size1)
        self.conv2 = GCNConv(self.tweet_out_size + hidden_size1, hidden_size2)

    def forward(self, merged_tree_feature, merged_tree_edge_index, indices):
        batch_size = max(indices) + 1
       # print('batch_size', batch_size)
       # print(merged_tree_feature.shape)
        embed_tree_feature = self.tweets_embedding(merged_tree_feature)
       # print('embed_tree_feature1', embed_tree_feature.shape)

        embed_tree_feature = embed_tree_feature.permute(1,0,2)  #seq * batch * hidden
       # print('embed_tree_feature2', embed_tree_feature.shape)

        state_size = [self.num_layer, embed_tree_feature.size(1), self.tweet_out_size]
        h0 = Variable(torch.randn(state_size)).to(device)
        out_nodes, hn = self.GRU(embed_tree_feature, h0)
       # print('hn', hn.shape)
        x_input = hn[-1,:,:] ## last layer
        # # replace the embedding of root source

        if self.direction == 'bu':
            index = torch.LongTensor([1, 0])
            merged_tree_edge_index[index] = merged_tree_edge_index
        x1 = copy.copy(x_input.float())
        x = self.conv1(x_input, merged_tree_edge_index)
        x2 = copy.copy(x)

        root_extend = torch.zeros(len(indices), x1.size(1)).to(device)   
        for num_batch in range(batch_size):
            index = (torch.eq(indices, num_batch))
            root_extend[index] = x1[num_batch]
        
        x = torch.cat((x,root_extend), 1)
        
        x = F.elu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, merged_tree_edge_index)
        x = F.elu(x)
        root_extend = torch.zeros(len(indices), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = torch.eq(indices, num_batch)
            root_extend[index] = x2[num_batch]
        x = torch.cat((x,root_extend), 1)

        # x -> [batch_size, embedding_size(hidden_size2)] node * embedding
        if self.direction == 'td':
            x = scatter_mean(x, indices, dim=0)  # do average on each tree
        elif self.direction == 'bu':
            x = scatter_mean(x, indices, dim=0)  # do average on each tree
            # x = x[0: batch_size, ]   # select the root embedding
        else:
            print("direction as td or bu")
        return x   # x.size()  [batch_size, self.tweet_out_size + hidden_size2]
        

class Net(nn.Module):
    """docstring for ClassName"""
    def __init__(self, args, tweet_embedding_matrix):
        super(Net, self).__init__()
        self.direction = args.direction
        self.tree_hidden_size1 = args.tree_hidden_size1
        self.tree_hidden_size2 = args.tree_hidden_size2
        # self.tree_hidden_size3 = args.tree_hidden_size3

        self.graph_hidden_size1 = args.graph_hidden_size1
        self.graph_hidden_size2 = args.graph_hidden_size2
        self.linear_size = args.linear_hidden_size1

        self.num_layer = 2
        self.text_input_size = args.embed_dim
        self.args = args
        ## model
        self.GraphGCN = GraphGCN(self.args, tweet_embedding_matrix, self.graph_hidden_size1, self.graph_hidden_size2)
        self.TreeGCN = TreeGCN(self.args, self.direction, tweet_embedding_matrix, self.text_input_size, self.tree_hidden_size1, self.tree_hidden_size2)
        # self.fc1 = nn.Linear(self.tree_hidden_size1 + self.tree_hidden_size2 + self.graph_hidden_size2, 4)
        self.fc1 = nn.Linear(self.graph_hidden_size2, 4)

    def forward(self, user_feats, graph_node_features, graph_edge_index, merged_tree_feature, merged_tree_edge_index, indices):
        graph_output,kl_loss, rec_loss = self.GraphGCN(user_feats, graph_node_features, graph_edge_index, indices)
        # tree_output = self.TreeGCN(merged_tree_feature, merged_tree_edge_index, indices)
        tweet_feature = graph_output
        # tweet_feature = torch.cat((graph_output, tree_output), 1)
        out_y = self.fc1(tweet_feature)
        
        return out_y, kl_loss, rec_loss
        
