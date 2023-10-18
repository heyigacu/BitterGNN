import numpy as np

import dgl
from dgl.nn.functional import edge_softmax
import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgllife.model.readout import WeightedSumAndMax

__all__ = ['HGNNLayer','MyGNN', 'MyPredictor']

class HGNNLayer(nn.Module):
    def __init__(self,
        n_node_feats = 1,
        n_edge_feats = 1,
        n_heads = 2,
        n_node2node_hidden_feats = 50,
        n_edge2node_hidden_feats = 50,
        n_node_out_feats = 50,
        n_node2edge_hidden_feats = 50,
        n_edge2edge_hidden_feats = 50,
        n_edge_out_feats = 50,
        node_gat = True,
        edge_gat = True,
        weave = False,
        mpnn = True,
        attn_activation= nn.LeakyReLU(negative_slope=0.2),
        activation=F.relu,
        attn_dropout = 0.0,
        feat_dropout = 0.0,
        xavier_normal = False,
        ):
        """
        args: 
            n_node_feats <int>: defalut 1, number of node embedding feature
            n_edge_feats <int>: defalut 1, number of edge embedding feature
            n_heads <int>: defalut 1, number of attention heads
            n_node2node_hidden_feats <int>: defalut 50, number of hidden neurons of node to node 
            n_edge2node_hidden_feats <int>: defalut 50, number of hidden neurons of edge to node 
            n_node_out_feats <int>: defalut 50, number of out neurons of node 
            n_node2edge_hidden_feats <int>: defalut 50, number of hidden neurons of node to edge 
            n_edge2edge_hidden_feats <int>: defalut 50, number of hidden neurons of edge to edge 
            n_edge_out_feats <int>: defalut 50, number of out neurons of edge 
            node_gat <bool>: defalut true, if add attention between a node with neighbour node
            edge_gat <bool>: defalut true, if add attention between a node with neighbour edge
            weave <bool>: defalut true, if use the weave learning, here is edge2node and edge2edge
            mpnn <bool>: defalut true, if use the mpnn module
            attn_activation <instance>: defalut nn.LeakyReLU(negative_slope=0.2), instance of activation function for attention
            activation <instance>: defalut F.relu, instance of activation function for attention
            attn_dropout <float>: defalut 0.0, dropout of attention
            feat_dropout <float>: defalut 0.0, dropout between 2 layers 
            xavier_normal <bool>: defalut False, if xavier normalization
        note:
            node_gat, edge_gat, weave, mpnn can combinate to 16 models
        """
        super(HGNNLayer, self).__init__()
        
        self.n_node2node_hidden_feats = n_node2node_hidden_feats
        self.n_edge2node_hidden_feats = n_edge2node_hidden_feats
        self.n_node_out_feats = n_node_out_feats
        self.n_node2edge_hidden_feats = n_node2edge_hidden_feats
        self.n_edge2edge_hidden_feats = n_edge2edge_hidden_feats
        self.n_edge_out_feats = n_edge_out_feats

        self.residual = True
        self.share_weights = True
        self.xavier_normal = xavier_normal
        self.n_heads = n_heads
        self.node_gat = node_gat
        self.edge_gat = edge_gat
        self.weave = weave
        self.mpnn = mpnn
        self.attn_activation = attn_activation
        self.activation = activation
        self.feat_dropout = nn.Dropout(feat_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)

        # Layers for updating node representations
        self.node_to_node_src = nn.Linear(n_node_feats, n_node2node_hidden_feats * n_heads)
        if self.share_weights:
            self.node_to_node_dst = self.node_to_node_src
        else:
            self.node_to_node_dst = nn.Linear(n_node_feats, n_node2node_hidden_feats * n_heads)
        self.attn_node_to_node = nn.Parameter(torch.FloatTensor(size=(1, n_heads, n_node2node_hidden_feats)))

        self.edge_to_node = nn.Linear(n_edge_feats, n_edge2node_hidden_feats * n_heads)
        self.attn_edge_to_node = nn.Parameter(torch.FloatTensor(size=(1, n_heads, n_edge2node_hidden_feats)))
        self.update_node = nn.Linear(n_node2node_hidden_feats+n_edge2node_hidden_feats, n_node_out_feats)
        self.gru = nn.GRU(n_node_out_feats, n_node2node_hidden_feats)
   
        # Layers for updating edge representations
        self.node_to_edge = nn.Linear(n_node_feats, n_node2edge_hidden_feats)
        self.edge_to_edge = nn.Linear(n_edge_feats, n_edge2edge_hidden_feats)
        self.update_edge = nn.Linear(n_node2edge_hidden_feats+n_edge2edge_hidden_feats, n_edge_out_feats)
        
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize model parameters."""

        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.attn_node_to_node, gain=gain)
        nn.init.xavier_normal_(self.attn_edge_to_node, gain=gain)
        
        if self.share_weights:
            fc_list = [self.node_to_node_src, 
                        self.edge_to_node,
                        self.update_node,
                        self.node_to_edge,
                        self.edge_to_edge,
                        self.update_edge,]
        else: 
            fc_list = [self.node_to_node_src, 
                        self.node_to_node_dst,
                        self.edge_to_node,
                        self.update_node,
                        self.node_to_edge,
                        self.edge_to_edge,
                        self.update_edge,]                   
        for fc in fc_list:
            if self.xavier_normal:
                nn.init.xavier_normal_(fc.weight, gain=gain)
                nn.init.constant_(fc.bias, 0)
            else:
                fc.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        """
        args:
            g <DGL bigraph>:
            node_feats <tensor>: size(n_nodes, n_node_feats)
            edge_feats <tensor>: size(n_edges, n_edge_feats)
        """

        g = g.local_var()
        g_self = dgl.add_self_loop(g)
        edge_feats = self.feat_dropout(edge_feats)
        node_feats = self.feat_dropout(node_feats)
        # node initial learning
        h_src = h_dst = node_feats
        h_e = edge_feats
        feat_src = self.node_to_node_src(h_src).view(-1, self.n_heads, self.n_node2node_hidden_feats) # Wh_i: (n_nodes, n_heads, n_node2node_hidden_feats)
        if self.share_weights:
            feat_dst = feat_src # Wh_i =  Wh_j = W1h_i
        else:
            feat_dst = self.node_to_node_dst(h_dst).view(-1, self.n_heads, self.n_node2node_hidden_feats) # Wh_j = W2h_j
        g.srcdata.update({"h_n2n_l": feat_src}) # Wh_i: (num_src_edge, num_heads, out_dim)
        g.dstdata.update({"h_n2n_r": feat_dst}) # Wh_j: (num_src_edge, num_heads, out_dim)
        # edge initial learning
        g.edata['e'] = self.edge_to_node(edge_feats).view(-1, self.n_heads, self.n_edge2node_hidden_feats) #(n_edges, n_heads, n_edge2node_hidden_feats)
        feat_e = g.edata['e'] #(n_edges, n_heads, n_edge2node_hidden_feats)

        #####################
        # update node
        #####################
        # node to node
        """
        implement gat in original paper
        # self.attn_node_to_node = nn.Parameter(torch.FloatTensor(size=(1, n_heads, 2, n_node2node_hidden_feats)))
        # g.apply_edges(fn.copy_u('el' ,'el'))
        # g.apply_edges(fn.copy_u('er', 'er'))
        # g.edata['e'] = torch.stack((g.edata['el'],g.edata['er']),dim=2) # [wh_i||wh_j] for i for j: (n_edges, n_heads, out_dim, 2)
        # e = self.leaky_relu(self.attn_node_to_node.transpose(2,3) * g.edata.pop("e")) # e = leakyrelu(a^T * [wh_i||wh_j]) for i for j: (n_edges, n_heads, out_dim, 2)
        # g.edata["a"] = self.attn_drop(edge_softmax(g, e))  # a = softmax(e): (num_edge, num_heads, out_dim, 2)
        In fact, a_{ij} should shape as (num_edge, num_heads) so need sum, mean or max last 2 dimension, so we use sum to implement it 
        # g.update_all(fn.u_mul_e("el", "a", "m"), fn.sum("m", "ft")) #h_i^' = sigma(awh_j)
        # rst = g.dstdata["ft"]
        """
        if self.node_gat:
            g.apply_edges(fn.u_add_v("h_n2n_l", "h_n2n_r", "e_n2n")) # [wh_i||wh_j]: (n_edges, n_heads, n_node2node_hidden_feats)
            e_n2n = g.edata.pop("e_n2n") # (n_edges, n_heads, n_node2node_hidden_feats)
            # self.attn_node_to_node: (1, n_heads, n_node2node_hidden_feats) 
            e_n2n = self.attn_activation((e_n2n * self.attn_node_to_node).sum(dim=-1).unsqueeze(dim=2)) # leakyrelu(a^T[wh_i||wh_j]): (n_edges, n_heads, 1)
            g.edata["a_n2n"] = self.attn_dropout(edge_softmax(g, e_n2n)) # softmax(e(h_i,h_j)): (n_edges, n_heads, 1)
            # np.savetxt('node_attention.txt', g.edata["a_n2n"].mean(1).detach().cpu().numpy())
            g.update_all(fn.u_mul_e("h_n2n_l", "a_n2n", "m"), fn.sum("m", "h_n2n")) # a*wh_i
            a_n2n = g.edata.pop("a_n2n") # delete g.edata["a_n2n"]
            if self.residual:
                node_node_feats = self.activation(g.ndata.pop("h_n2n")+feat_src) # h'_i = sigma(a*wh_i):(n_nodes, n_heads, n_node2node_hidden_feats)
            else:
                node_node_feats = self.activation(g.ndata.pop("h_n2n")) # h'_i = sigma(a*wh_i):(n_nodes, n_heads, n_node2node_hidden_feats)
        else:
            node_node_feats = self.activation(feat_src)
    
        # edge to node
        if self.edge_gat:
            g.apply_edges(fn.copy_u('h_n2n_l','h_e'))
            e_e2n = g.edata.pop('h_e') + feat_e #[wh_i||we_j]
            es = self.attn_activation((self.attn_edge_to_node*e_e2n).sum(dim=-1).unsqueeze(dim=2)) #leakyrelu(a^T[wh_i||we_j])
            g.edata['a_e2n'] = self.attn_dropout(edge_softmax(g, es)) # (n_edges, n_heads, 1)
            a_e2n = g.edata.pop('a_e2n') # delete g.edata["a_e2n"]
            # np.savetxt('edge_attention.txt', a_e2n.mean(1).detach().cpu().numpy())
            g.edata['e2n'] = a_e2n * feat_e #(n_edges, n_heads, n_edge2node_hidden_feats)
            g.update_all(fn.copy_e('e2n', 'm'), fn.sum('m', 'e2n'))
            if self.residual:
                edge_node_feats = self.activation(g.ndata.pop('e2n')+feat_src)
            else:
                edge_node_feats = self.activation(g.ndata.pop('e2n')) # (n_nodes, n_heads, n_edge2node_hidden_feats)         
        else:
            g.update_all(fn.copy_e('e', 'm'), fn.sum('m', 'e2n'))
            edge_node_feats = self.activation(g.ndata.pop('e2n'))

        # update node
        new_node_feats = self.activation(self.update_node(torch.cat([node_node_feats, edge_node_feats], dim=2))) # (n_nodes, n_heads, n_node_out_feats)
        if self.mpnn: 
            new_node_feats = new_node_feats.unsqueeze(0).mean(2) # (1, n_nodes, n_node_out_feats)
            old_node_feats = feat_src.unsqueeze(0).mean(2) # (1, n_nodes, n_node2node_hidden_feats)
            new_node_feats, hidden_feats = self.gru(new_node_feats, old_node_feats)
            new_node_feats = new_node_feats.squeeze(0) # (n_nodes, n_node_out_feats)
        else:
            new_node_feats = new_node_feats.mean(1) # (n_nodes, n_node_out_feats)
            
        #####################
        # update edge
        #####################
        if self.weave:
            g.ndata['h_n2e'] = self.activation(self.node_to_edge(h_src)) # (n_nodes, n_node2edge_hidden_feats)
            g.apply_edges(fn.u_add_v('h_n2e', 'h_n2e', 'he'))
            node_edge_feats = g.edata.pop('he') # (n_nodes, n_node2edge_hidden_feats)
            edge_edge_feats = self.activation(self.edge_to_edge(h_e)) # (n_edges, n_edge2edge_hidden_feats)
            new_edge_feats = self.activation(self.update_edge(torch.cat([node_edge_feats, edge_edge_feats], dim=1))) # (n_edges, n_edge_out_feats)       
        else:
            new_edge_feats = self.activation(feat_e).mean(1)  # (n_edges, n_node_out_feats)
        return new_node_feats, new_edge_feats

class HGNN(nn.Module):
    def __init__(self,
                 n_node_feats=1,
                 n_edge_feats=1,
                 num_layers=2,
                 n_heads=5,
                node_gat = True,
                edge_gat = True,
                weave = True,
                mpnn = True,
                 n_hidden_feats=50,
                 activation=F.relu,
                 attn_activation = nn.LeakyReLU(negative_slope=0.2),
                attn_dropout = 0.0,
                feat_dropout = 0.0,
                xavier_normal=False,
                 ):
        super(HGNN, self).__init__()

        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gnn_layers.append(HGNNLayer(n_node_feats = n_node_feats,
                                                n_edge_feats = n_edge_feats,
                                                n_heads = n_heads,
                                                n_node2node_hidden_feats = n_hidden_feats,
                                                n_edge2node_hidden_feats = n_hidden_feats,
                                                n_node_out_feats = n_hidden_feats,
                                                n_node2edge_hidden_feats = n_hidden_feats,
                                                n_edge2edge_hidden_feats = n_hidden_feats,
                                                n_edge_out_feats = n_hidden_feats,
                                                node_gat = node_gat,
                                                edge_gat = edge_gat,
                                                weave=weave,
                                                mpnn = mpnn,
                                                attn_activation= attn_activation,
                                                activation=activation,
                                                attn_dropout = attn_dropout,
                                                feat_dropout = feat_dropout,
                                                xavier_normal = xavier_normal,
                                                ))
            else:
                self.gnn_layers.append(HGNNLayer(n_node_feats = n_hidden_feats,
                                                n_edge_feats = n_hidden_feats,
                                                n_heads = n_heads,
                                                n_node2node_hidden_feats = n_hidden_feats,
                                                n_edge2node_hidden_feats = n_hidden_feats,
                                                n_node_out_feats = n_hidden_feats,
                                                n_node2edge_hidden_feats = n_hidden_feats,
                                                n_edge2edge_hidden_feats = n_hidden_feats,
                                                n_edge_out_feats = n_hidden_feats,
                                                attn_activation= nn.LeakyReLU(negative_slope=0.2),
                                                activation=F.relu,
                                                node_gat = node_gat,
                                                edge_gat = edge_gat,
                                                weave=weave,
                                                mpnn = mpnn,
                                                attn_dropout = attn_dropout,
                                                feat_dropout = feat_dropout,
                                                xavier_normal = xavier_normal,
                                                ))
    def reset_parameters(self):
        """Reinitialize model parameters."""
        for layer in self.gnn_layers:
            layer.reset_parameters()

    def forward(self, g, node_feats, edge_feats, save_info=False):
        """Updates node representations (and edge representations)."""
        for i in range(len(self.gnn_layers)):
            node_feats, edge_feats = self.gnn_layers[i](g, node_feats, edge_feats)
            if save_info:
                np.savetxt('edge{}.txt'.format(i+1), edge_feats.detach().cpu().numpy())
                np.savetxt('node{}.txt'.format(i+1), node_feats.detach().cpu().numpy())
        return node_feats, edge_feats
    

class HGNNPredictor(nn.Module):
    def __init__(self,
                n_node_feats=1,
                n_edge_feats=1,
                num_layers=2,
                n_heads=5,
                node_gat = True,
                edge_gat = True,
                weave = True,
                mpnn = True,
                n_hidden_feats=50,
                activation=F.relu,
                attn_activation = nn.LeakyReLU(negative_slope=0.2),
                attn_dropout = 0,
                feat_dropout = 0,
                xavier_normal = False,
                n_tasks=2,
        ):
        super(HGNNPredictor, self).__init__()
        self.gnn = HGNN(
                        n_node_feats=n_node_feats,
                        n_edge_feats=n_edge_feats,
                        num_layers=num_layers,
                        n_heads=n_heads,
                        node_gat = node_gat,
                        edge_gat = edge_gat,
                        weave=weave,
                        mpnn = mpnn,
                        n_hidden_feats=n_hidden_feats,
                        activation=activation,
                        attn_activation = attn_activation,
                        attn_dropout=attn_dropout,
                        feat_dropout=feat_dropout,
                        xavier_normal = xavier_normal,
                        )
        self.readout = WeightedSumAndMax(in_feats=n_hidden_feats)
        self.predict = nn.Sequential(
            nn.Linear(2*n_hidden_feats, 64),   
            nn.Linear(64, n_tasks),
        )

    def forward(self, g, node_feats, edge_feats):
        node_feats, edge_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)
        return self.predict(graph_feats)
    
def main():
    u, v = torch.tensor([0, 1]), torch.tensor([1, 0])
    g = dgl.graph((u, v))
    nfeats = torch.tensor([[1., 1.],[1., 1.]])
    efeats = torch.tensor([[1., 1.],[1., 1.]])
    model = HGNNPredictor(2, 2, n_tasks=1)
    print(model(g, nfeats, efeats))

if __name__ == "__main__":
    main()

