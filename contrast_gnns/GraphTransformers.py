import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgllife.model.readout import WeightedSumAndMax


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(nn.ModuleList([
                nn.Linear(in_dim, hidden_dim, bias=False),  # For query
                nn.Linear(in_dim, hidden_dim, bias=False),  # For key
                nn.Linear(in_dim, hidden_dim, bias=False)   # For value
            ]))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def edge_attention(self, edges):
        query = edges.src['query']
        key = edges.dst['key']
        value = edges.src['value']
        e = self.leaky_relu((query * key).sum(-1, keepdim=True))
        return {'e': e, 'value': value}

    def message_func(self, edges):
        return {'e': edges.data['e'], 'value': edges.data['value']}

    def reduce_func(self, nodes):
        alpha = self.softmax(nodes.mailbox['e'])
        h = (alpha * nodes.mailbox['value']).sum(dim=1)
        return {'h': h}

    def forward(self, g, h):
        head_outs = []
        for linear_query, linear_key, linear_value in self.heads:
            query, key, value = linear_query(h), linear_key(h), linear_value(h)
            g.ndata.update({'query': query, 'key': key, 'value': value})
            g.apply_edges(self.edge_attention)
            g.update_all(self.message_func, self.reduce_func)
            head_outs.append(g.ndata.pop('h'))
        return torch.cat(head_outs, dim=-1)  # Concatenate heads' outputs


class GraphTransformer(nn.Module):
    def __init__(self, in_dim=50, hidden_dim=50, out_dim=2, num_layers=3, num_heads=5):
        super(GraphTransformer, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(MultiHeadGATLayer(in_dim, hidden_dim, num_heads))
        for _ in range(num_layers - 1):
            self.layers.append(MultiHeadGATLayer(hidden_dim * num_heads, hidden_dim, num_heads))
        self.out_proj = nn.Linear(hidden_dim * num_heads, out_dim)

    def forward(self, g, h):
        for layer in self.layers:
            h = layer(g, h)
        return self.out_proj(h)


class GraphTransformerPredictor(nn.Module):
    def __init__(self, in_dim=50, hidden_dim=100, out_dim=100, n_tasks=1, num_layers=2, num_heads=5):
        super(GraphTransformerPredictor, self).__init__()
        self.gnn = GraphTransformer(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, num_layers=num_layers, num_heads=num_heads)
        self.readout = WeightedSumAndMax(in_feats=hidden_dim)
        self.predict = nn.Sequential(
            nn.Linear(2*hidden_dim, 64),   
            nn.Linear(64, n_tasks),
        )

    def forward(self, g, node_feats):
        node_feats = self.gnn(g, node_feats)
        graph_feats = self.readout(g, node_feats)
        return self.predict(graph_feats)

def main():
    # Test code
    g = dgl.graph(([0, 1, 2], [1, 2, 0]))  
    h = torch.randn((g.num_nodes(), 10)) 
    model = GraphTransformerPredictor(10)  
    out = model(g, h)  
    print(out) 

if __name__=="__main__":
    main()