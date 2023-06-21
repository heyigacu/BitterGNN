import numpy as np

import dgl
from dgl.utils import expand_as_pair
from dgl.nn.functional import edge_softmax
from dgllife.utils import mol_to_complete_graph
import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgllife.model.model_zoo import MLPPredictor
from dgllife.model.readout import MLPNodeReadout,SumAndMax,WeightedSumAndMax
__all__ = ['MyLayer','MyGNN', 'MyPredictor']

class MyLayer(nn.Module):
    def __init__(self,
                 n_node_feats = 1,
                 n_edge_feats = 1,
                 n_heads = 1,
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
        super(MyLayer, self).__init__()
        
        self.n_node2node_hidden_feats = n_node2node_hidden_feats
        self.n_edge2node_hidden_feats = n_edge2node_hidden_feats
        self.n_node_out_feats = n_node_out_feats
        self.n_node2edge_hidden_feats = n_node2edge_hidden_feats
        self.n_edge2edge_hidden_feats = n_edge2edge_hidden_feats
        self.n_edge_out_feats = n_edge_out_feats

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
        """
        ##implement in original paper
        # self.attn_node_to_node = nn.Parameter(torch.FloatTensor(size=(1, n_heads, 2, n_node2node_hidden_feats)))
        # g.apply_edges(fn.copy_u('el' ,'el'))
        # g.apply_edges(fn.copy_u('er', 'er'))
        # g.edata['e'] = torch.stack((g.edata['el'],g.edata['er']),dim=2) # [wh_i||wh_j] for i for j: (n_edges, n_heads, out_dim, 2)
        # e = self.leaky_relu(self.attn_node_to_node.transpose(2,3) * g.edata.pop("e")) # e = leakyrelu(a^T * [wh_i||wh_j]) for i for j: (n_edges, n_heads, out_dim, 2)
        # g.edata["a"] = self.attn_drop(edge_softmax(g, e))  # a = softmax(e): (num_edge, num_heads, out_dim, 2)

        ##In fact, a_{ij} should shape as (num_edge, num_heads) so need sum, mean or max last 2 dimension, so we use sum to implement it 

        # g.update_all(fn.u_mul_e("el", "a", "m"), fn.sum("m", "ft")) #h_i^' = sigma(awh_j)
        # rst = g.dstdata["ft"]
        """

        g = g.local_var()
        g_self = dgl.add_self_loop(g)

        edge_feats = self.feat_dropout(edge_feats)
        node_feats = self.feat_dropout(node_feats)
        # node 
        h_src = h_dst = node_feats
        h_e = edge_feats
        feat_src = self.node_to_node_src(h_src).view(-1, self.n_heads, self.n_node2node_hidden_feats) # Wh_i: (n_nodes, n_heads, n_node2node_hidden_feats)
        if self.share_weights:
            feat_dst = feat_src # Wh_i =  Wh_j = W1h_i
        else:
            feat_dst = self.node_to_node_dst(h_dst).view(-1, self._num_heads, self._out_feats) # Wh_j = W2h_j
        g.srcdata.update({"h_n2n_l": feat_src}) # Wh_i: (num_src_edge, num_heads, out_dim)
        g.dstdata.update({"h_n2n_r": feat_dst}) # Wh_j: (num_src_edge, num_heads, out_dim)
        # edge
        g.edata['e'] = self.edge_to_node(edge_feats).view(-1, self.n_heads, self.n_edge2node_hidden_feats) #(n_edges, n_heads, n_edge2node_hidden_feats)
        feat_e = g.edata['e'] #(n_edges, n_heads, n_edge2node_hidden_feats)


        #####################
        # update node
        #####################
        # node to node
        if self.node_gat:
            g.apply_edges(fn.u_add_v("h_n2n_l", "h_n2n_r", "e_n2n")) # [wh_i||wh_j]: (n_edges, n_heads, n_node2node_hidden_feats)
            e_n2n = g.edata.pop("e_n2n") # (n_edges, n_heads, n_node2node_hidden_feats)
            # self.attn_node_to_node: (1, n_heads, n_node2node_hidden_feats) 
            e_n2n = self.attn_activation((e_n2n * self.attn_node_to_node).sum(dim=-1).unsqueeze(dim=2)) # leakyrelu(a^T[wh_i||wh_j]): (n_edges, n_heads, 1)
            g.edata["a_n2n"] = self.attn_dropout(edge_softmax(g, e_n2n)) # softmax(e(h_i,h_j)): (n_edges, n_heads, 1)
            g.update_all(fn.u_mul_e("h_n2n_l", "a_n2n", "m"), fn.sum("m", "h_n2n")) # a*wh_i
            a_n2n = g.edata.pop("a_n2n") # delete g.edata["a_n2n"]
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
            g.edata['e2n'] = a_e2n * feat_e #(n_edges, n_heads, n_edge2node_hidden_feats)
            g.update_all(fn.copy_e('e2n', 'm'), fn.sum('m', 'e2n'))
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

class MyGNN(nn.Module):
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
        super(MyGNN, self).__init__()

        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gnn_layers.append(MyLayer(n_node_feats = n_node_feats,
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
                self.gnn_layers.append(MyLayer(n_node_feats = n_hidden_feats,
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

    def forward(self, g, node_feats, edge_feats):
        """Updates node representations (and edge representations)."""
        for i in range(len(self.gnn_layers)):
            node_feats, edge_feats = self.gnn_layers[i](g, node_feats, edge_feats)
            np.savetxt('l{}b.txt'.format(i), edge_feats.detach().numpy())
            np.savetxt('l{}a.txt'.format(i), node_feats.detach().numpy())
        return node_feats, edge_feats
    

class MyPredictor(nn.Module):
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
                readout = "SumAndMax",
                predict = "MLPPredictor",
        ):
        super(MyPredictor, self).__init__()
        self.gnn = MyGNN(
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
        gnn_out_feats = n_hidden_feats
        if readout == "WeightedSumAndMax":
            self.readout = WeightedSumAndMax(gnn_out_feats)
        elif readout == "SumAndMax":
            self.readout = SumAndMax()
        else:
            raise ValueError
        if predict == "MLPPredictor":
            predictor_out_feats=128,
            predictor_dropout=0.
            self.predict = MLPPredictor(2*gnn_out_feats, 128,
                                        n_tasks, predictor_dropout)
        
    def forward(self, g, node_feats, edge_feats ):
        node_feats, edge_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)
        return self.predict(graph_feats)
    
def main():
    from features import featurize_atoms, featurize_bonds1
    from features import atom_number_featurizer, bond_1_featurizer, featurize_atoms, featurize_bonds
    node_featurizer, n_nfeats = atom_number_featurizer()
    edge_featurizer, n_efeats = bond_1_featurizer()
    from dgllife.utils import smiles_to_bigraph
    g = smiles_to_bigraph('CC=O', node_featurizer=node_featurizer, edge_featurizer=edge_featurizer)
    print(g)
    node_feats, edge_feats = g.ndata['h'], g.edata['e']
    print(node_feats)
    print(edge_feats)
    model = MyPredictor(n_heads=1, n_node_feats=n_nfeats, n_edge_feats=n_efeats, readout="WeightedSumAndMax")
    model.eval()
    rst = model(g,node_feats,edge_feats)
    print(rst)

if __name__ == "__main__":
    main()

