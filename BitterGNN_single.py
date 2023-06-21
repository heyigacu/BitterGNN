import torch
from model import MyPredictor
from rdkit import Chem
from dgllife.utils import mol_to_complete_graph,mol_to_bigraph
from load_data import load_data
import torch.nn as nn
import torch.nn.functional as F
from features import atom_number_featurizer, bond_1_featurizer, canonical_atom_featurlizer, canonical_bond_featurlizer
from train import Train_Trainer
import os
from model import MyPredictor
from torch.nn import CrossEntropyLoss
import dgl
path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) 

model_save_folder = os.path.join(path,'result/model/')


node_featurizer, n_nfeats = atom_number_featurizer()
edge_featurizer, n_efeats = bond_1_featurizer()

# paras_dic_bn_single ={
# 'num_layers':[3],
# 'n_heads':[5],
# 'n_hidden_feats':[200], 
# 'feat_dropout':[0.03],
# 'lr': [0.0005],
# }
model =  MyPredictor(
        n_node_feats=n_nfeats, 
        n_edge_feats=n_efeats,
        num_layers=3,
        n_heads=5,
        node_gat = True,
        edge_gat = True,
        weave = True,
        mpnn = True,
        n_hidden_feats=200,
        activation=F.relu,
        attn_activation = nn.LeakyReLU(negative_slope=0.2),
        attn_dropout = 0,
        feat_dropout = 0.03,
        n_tasks=2,
        readout="WeightedSumAndMax"
        )

def train():
    loader = load_data(node_featurizer=node_featurizer,
                    edge_featurizer=edge_featurizer,
                    batch_size=128,
                    split_method='all',
                    k=0)
    train_loader =loader.load_oringal()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    epochs = 460
    model.cuda()
    for epoch in range(1, epochs+1):
        # train
        model.train()
        print('epoch:',epoch)
        for i, (graphs, labels) in enumerate(train_loader):
            labels = labels.cuda()
            graphs = graphs.to('cuda:0')
            preds = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
            optimizer.zero_grad()
            loss = CrossEntropyLoss()(preds, labels)
            loss.backward()
            optimizer.step()
    torch.save({'model': model.state_dict()}, os.path.join(model_save_folder,'temp.pth'))

# train()       


class PretrainedPredictor():
    """
    a class for predict smiles to which class from pretrained model
    """
    # read model
    model = MyPredictor(
                n_node_feats=n_nfeats, 
                n_edge_feats=n_efeats,
                num_layers=3,
                n_heads=5,
                node_gat = True,
                edge_gat = True,
                weave = True,
                mpnn = True,
                n_hidden_feats=200,
                activation=F.relu,
                attn_activation = nn.LeakyReLU(negative_slope=0.2),
                attn_dropout = 0,
                feat_dropout = 0.03,
                n_tasks=2,
                readout="WeightedSumAndMax"
                    )
    state_dict = torch.load(os.path.join(model_save_folder,'temp.pth'))
    model.load_state_dict(state_dict['model'])
    model.eval()

    @classmethod
    def predit(cls,smiles):
        """
        args:
            smiles <str>: simles format of molecule
        return:
            class of predictting 
        """
        ROMol = Chem.MolFromSmiles(smiles)
        graph= mol_to_bigraph(ROMol, node_featurizer=node_featurizer, edge_featurizer=edge_featurizer)
        atom_feats = graph.ndata.pop('h')
        edge_feats = graph.edata.pop('e')
        pred = cls.model(graph, atom_feats, edge_feats)
        pred_cls = pred.argmax(-1).detach().numpy()
        return pred_cls[0]

print(PretrainedPredictor.predit('Cn1c(=O)c2c(ncn2C)n(C)c1=O'))

