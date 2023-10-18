import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgllife.model.readout import WeightedSumAndMax
from dgllife.model.gnn import GraphSAGE


class GraphSAGEPredictor(nn.Module):
    def __init__(self, in_feats=74, hidden_feats=[100,100], activation=None, dropout=None, aggregator_type=None, n_tasks=1):
        super(GraphSAGEPredictor, self).__init__()
        self.gnn = GraphSAGE(in_feats=in_feats, hidden_feats=hidden_feats, activation=activation, dropout=dropout, aggregator_type=aggregator_type,)
        self.readout = WeightedSumAndMax(in_feats=hidden_feats[-1])
        self.predict = nn.Sequential(
            nn.Linear(2*hidden_feats[-1], 64),   
            nn.Linear(64, n_tasks),
        )
    def forward(self, g, node_feats):
        node_feats = self.gnn(g, node_feats)
        graph_feats = self.readout(g, node_feats)
        return self.predict(graph_feats)