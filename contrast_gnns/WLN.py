import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgllife.model.readout import WeightedSumAndMax
from dgllife.model.gnn import WLN


class WLNPredictor(nn.Module):
    def __init__(self, node_in_feats=74, edge_in_feats=12, node_out_feats=100, n_layers=2, project_in_feats=True, set_comparison=True, n_tasks=1):
        super(WLNPredictor, self).__init__()
        self.gnn = WLN(node_in_feats=node_in_feats, edge_in_feats=edge_in_feats, node_out_feats=node_out_feats, n_layers=n_layers, project_in_feats=project_in_feats, set_comparison=set_comparison)
        self.readout = WeightedSumAndMax(in_feats=node_out_feats)
        self.predict = nn.Sequential(
            nn.Linear(2*node_out_feats, 64),   
            nn.Linear(64, n_tasks),
        )
    def forward(self, g, node_feats, edge_feats):
        node_feats = self.gnn(g, node_feats, edge_feats)
        graph_feats = self.readout(g, node_feats)
        return self.predict(graph_feats)

