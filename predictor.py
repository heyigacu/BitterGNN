import argparse
import torch
import joblib
import os
import numpy as np
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from hgnn.model import HGNNPredictor
from hgnn.features import Graph_smiles

import dgl
from dgl import DGLGraph
from  dgllife.model.gnn import WeaveGNN
from  dgllife.model.readout import WeightedSumAndMax,SumAndMax
from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit.Chem import Descriptors,GraphDescriptors,MolSurf,Crippen,Fragments,Lipinski,QED,rdMolDescriptors
from rdkit.Chem.EState import EState_VSA,EState

from contrast_gnns.GraphSAGE import *
from contrast_gnns.WLN import *
from dgllife.model.model_zoo import AttentiveFPPredictor, PAGTNPredictor


work_dir = os.path.abspath(os.path.dirname(__file__))


parser = argparse.ArgumentParser(description='kokumi and flavous predictor')
parser.add_argument("-t", "--type", type=int, choices=[0,1,2], default=0,
                    help="0 is Bitter/Nonbitter prediction, 1 is Bitter/Sweet prediction, 2 is Multi-flavor"
                    )
parser.add_argument("-m", "--model", type=int, choices=[0,1,2,3], default=3,
                    help="0 is HGNN, 1 is WLN, 2 is AttentiveFP, 3 is GraphSAGE"
                    )
parser.add_argument("-i", "--file", type=str, default=work_dir+'/test.csv', help="input smiles file, don't have a header, only a column smiles")
parser.add_argument("-o", "--out", type=str, default=work_dir+'/result.csv',help="output file")
args = parser.parse_args()





def collate_smiles(graphs):
    batched_graph = dgl.batch(graphs)
    batched_graph.set_n_initializer(dgl.init.zero_initializer)
    batched_graph.set_e_initializer(dgl.init.zero_initializer)
    return batched_graph

    
"""
for i in range(len(self.gnn_layers) - 1):
    node_feats, edge_feats = self.gnn_layers[i](g, node_feats, edge_feats)
import numpy as np
# np.savetxt('/home/hy/Documents/Project/Astringent/predict/n2.txt', node_feats.detach().cpu().numpy())
# np.savetxt('/home/hy/Documents/Project/Astringent/predict/b2.txt', edge_feats.detach().cpu().numpy())
return self.gnn_layers[-1](g, node_feats, edge_feats, node_only)
"""



##################
# predict
##################


def gnn_bn(model, path, smileses, edge ):
    state_dict = torch.load(os.path.join(work_dir,path))
    model.load_state_dict(state_dict)
    model.eval()
    total = []
    n=0
    for smiles in smileses:
        try:
            n+=1
            for i in list(DataLoader([Graph_smiles(smiles)], batch_size=1, shuffle=False, collate_fn=collate_smiles, drop_last=False)):
                graphs = i
            if edge:
                rst = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
            else:
                rst = model(graphs, graphs.ndata.pop('h'))
            rst =  F.softmax(rst,dim=1).detach().numpy()[0]
            labels = ['Non-Bitter','Bitter']
            string = labels[rst.argmax()]
            ls = []
            ls.append(string)
            for value in rst:
                ls.append('{:.4f}'.format(value))
            total.append(ls)
        except:
            total.append(['error smiles', 'nan', 'nan'])
    return total

def gnn_bs(model, path, smileses, edge ):
    state_dict = torch.load(os.path.join(work_dir,path))
    model.load_state_dict(state_dict)
    model.eval()
    total = []
    for smiles in smileses:
        try:
            for i in list(DataLoader([Graph_smiles(smiles)], batch_size=1, shuffle=False, collate_fn=collate_smiles, drop_last=False)):
                graphs = i
            if edge:
                rst = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
            else:
                rst = model(graphs, graphs.ndata.pop('h'))
            rst =  F.softmax(rst,dim=1).detach().numpy()[0]
            labels = ['Sweet','Bitter']
            string = labels[rst.argmax()]
            ls = []
            ls.append(string)
            for value in rst:
                ls.append('{:.4f}'.format(value))
            total.append(ls)
        except:
            total.append(['error smiles', 'nan', 'nan'])
    return total

def gnn_multi(model, path, smileses, edge ):
    state_dict = torch.load(os.path.join(work_dir,path))
    model.load_state_dict(state_dict)
    model.eval()
    total = []
    for smiles in smileses:
        try:
            for i in list(DataLoader([Graph_smiles(smiles)], batch_size=1, shuffle=False, collate_fn=collate_smiles, drop_last=False)):
                graphs = i
            if edge:
                rst = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
            else:
                rst = model(graphs, graphs.ndata.pop('h'))
            rst =  F.softmax(rst,dim=1).detach().numpy()[0]
            labels = ['Bitter','Sweet','Sour', 'Salty','Umami', 'Kokumi', 'Astringent','Tasteless']
            string = labels[rst.argmax()]
            ls = []
            ls.append(string)
            for value in rst:
                ls.append('{:.4f}'.format(value))
            total.append(ls)
        except:
            total.append(['error smiles', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'])
    return total


smileses = []
with open(args.file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        smileses.append(line.split()[0].strip())
# smileses = ['C1=C(C=C(C(=C1O)O)O)C(=O)OC2=CC(=CC(=C2O)O)C(=O)OCC3C(C(C(C(O3)OC(=O)C4=CC(=C(C(=C4)OC(=O)C5=CC(=C(C(=C5)O)O)O)O)O)OC(=O)C6=CC(=C(C(=C6)OC(=O)C7=CC(=C(C(=C7)O)O)O)O)O)OC(=O)C8=CC(=C(C(=C8)OC(=O)C9=CC(=C(C(=C9)O)O)O)O)O)OC(=O)C1=CC(=C(C(=C1)OC(=O)C1=CC(=C(C(=C1)O)O)O)O)O']


n_node_feats=74
n_edge_feats=12



if args.type == 0:
    
    n_tasks=2
    if args.model == 0:
        print("HGNN")
        model = HGNNPredictor(
                            node_gat = True, 
                            edge_gat = True, 
                            weave = True, 
                            mpnn = True, 
                            n_node_feats=n_node_feats, n_edge_feats=n_edge_feats, num_layers=2, n_heads=5, n_hidden_feats=100, activation=F.relu, attn_activation=nn.LeakyReLU(negative_slope=0.2), attn_dropout=0, feat_dropout=0, xavier_normal=False, n_tasks=n_tasks,
                            )
        path= "pretrained/bn_hgnn.pth"
        edge= True
    elif args.model == 1:
        model= model = WLNPredictor(node_in_feats=n_node_feats, edge_in_feats=n_edge_feats, node_out_feats=100, n_layers=2, project_in_feats=True, set_comparison=True, n_tasks=n_tasks)
        path= "pretrained/bn_wln.pth"
        edge=True
    elif args.model == 2:
        model= AttentiveFPPredictor(node_feat_size=n_node_feats,edge_feat_size=n_edge_feats,num_layers=2,num_timesteps=2,graph_feat_size=100,n_tasks=n_tasks,dropout=0.) 
        path= "pretrained/bn_afp.pth"
        edge=True
    elif args.model == 3:
        model = GraphSAGEPredictor(in_feats=n_node_feats, hidden_feats=[100,100], activation=None, dropout=None, aggregator_type=None, n_tasks=n_tasks)
        path="pretrained/bn_gsage.pth"
        edge=False
    else:
        raise ValueError
    total = gnn_bn(model, path, smileses, edge)
    df = pd.DataFrame(total)
    df.columns =  ['Taste','Non-Bitter', 'Bitter']
    df.insert(0,'Smiles',smileses)
    df.to_csv(args.out,index=False,header=True,sep='\t')
elif args.type == 1:
    n_tasks=2
    if args.model == 0:
        model = HGNNPredictor(
                            node_gat = True, 
                            edge_gat = True, 
                            weave = True, 
                            mpnn = True, 
                            n_node_feats=n_node_feats, n_edge_feats=n_edge_feats, num_layers=2, n_heads=5, n_hidden_feats=100, activation=F.relu, attn_activation=nn.LeakyReLU(negative_slope=0.2), attn_dropout=0, feat_dropout=0, xavier_normal=False, n_tasks=n_tasks,
                            )
        path= "pretrained/bs_hgnn.pth"
        edge= True
    elif args.model == 1:
        model= model = WLNPredictor(node_in_feats=n_node_feats, edge_in_feats=n_edge_feats, node_out_feats=100, n_layers=2, project_in_feats=True, set_comparison=True, n_tasks=n_tasks)
        path= "pretrained/bs_wln.pth"
        edge=True
    elif args.model == 2:
        model= AttentiveFPPredictor(node_feat_size=n_node_feats,edge_feat_size=n_edge_feats,num_layers=2,num_timesteps=2,graph_feat_size=100,n_tasks=n_tasks,dropout=0.) 
        path= "pretrained/bs_afp.pth"
        edge=True
    elif args.model == 3:
        model = GraphSAGEPredictor(in_feats=n_node_feats, hidden_feats=[100,100], activation=None, dropout=None, aggregator_type=None, n_tasks=n_tasks)
        path="pretrained/bs_gsage.pth"
        edge=False
    else:
        raise ValueError
    total = gnn_bs(model, path, smileses, edge)
    df = pd.DataFrame(total)
    df.columns =  ['Taste', 'Sweet', 'Bitter']
    df.insert(0,'Smiles',smileses)
    df.to_csv(args.out,index=False,header=True,sep='\t')

elif args.type == 2:
    n_tasks=8
    if args.model == 0:
        model = HGNNPredictor(
                            node_gat = True, 
                            edge_gat = True, 
                            weave = True, 
                            mpnn = True, 
                            n_node_feats=n_node_feats, n_edge_feats=n_edge_feats, num_layers=2, n_heads=5, n_hidden_feats=100, activation=F.relu, attn_activation=nn.LeakyReLU(negative_slope=0.2), attn_dropout=0, feat_dropout=0, xavier_normal=False, n_tasks=n_tasks,
                            )
        path= "pretrained/multi_hgnn.pth"
        edge= True
    elif args.model == 1:
        model= model = WLNPredictor(node_in_feats=n_node_feats, edge_in_feats=n_edge_feats, node_out_feats=100, n_layers=2, project_in_feats=True, set_comparison=True, n_tasks=n_tasks)
        path= "pretrained/multi_wln.pth"
        edge=True
    elif args.model == 2:
        model= AttentiveFPPredictor(node_feat_size=n_node_feats,edge_feat_size=n_edge_feats,num_layers=2,num_timesteps=2,graph_feat_size=100,n_tasks=n_tasks,dropout=0.) 
        path= "pretrained/multi_afp.pth"
        edge=True
    elif args.model == 3:
        model = GraphSAGEPredictor(in_feats=n_node_feats, hidden_feats=[100,100], activation=None, dropout=None, aggregator_type=None, n_tasks=n_tasks)
        path="pretrained/multi_gsage.pth"
        edge=False
    else:
        raise ValueError
    total = gnn_multi(model, path, smileses, edge)
    df = pd.DataFrame(total)
    df.columns =  ['Taste', 'Bitter','Sweet','Sour', 'Salty','Umami', 'Kokumi', 'Astringent','Tasteless']
    df.insert(0,'Smiles',smileses)
    df.to_csv(args.out,index=False,header=True,sep='\t')
else:
    pass
