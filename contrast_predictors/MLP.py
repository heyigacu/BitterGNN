"""
For a comparasion, we reachieved the CNN and MLP of Bo et al;
https://www.sciencedirect.com/science/article/pii/S096399692200031X#f0005

written by Yi He, May 2023
"""
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from sklearn.model_selection import KFold

from rdkit import Chem,DataStructs
from rdkit.Chem import AllChem,Draw,PandasTools
from rdkit.Chem.EState import EState_VSA,EState
from rdkit.Chem import Descriptors,GraphDescriptors,MolSurf,QED,Crippen,Fragments,GraphDescriptors,Lipinski

path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def loader(filepath):
    df = pd.read_csv(filepath, header=0, sep='\t') #smiles,Taste
    return list(zip(df['Smiles'],df['Label']))

def RDKFP(tuple_ls):
    fps = [(torch.from_numpy(np.array(Chem.RDKFingerprint(Chem.MolFromSmiles(smiles)))),label) for (smiles,label) in tuple_ls]
    return fps

def Descriptors_bitter_sweet(tuple_ls):
    return [
        (torch.from_numpy(np.array([
            EState_VSA.VSA_EState9(Chem.MolFromSmiles(smiles)),
            Crippen.MolMR(Chem.MolFromSmiles(smiles)),
            Descriptors.NumValenceElectrons(Chem.MolFromSmiles(smiles)),
            MolSurf.LabuteASA(Chem.MolFromSmiles(smiles)),
            EState_VSA.EState_VSA1(Chem.MolFromSmiles(smiles)),
            MolSurf.SlogP_VSA2(Chem.MolFromSmiles(smiles)),
            MolSurf.SMR_VSA5(Chem.MolFromSmiles(smiles)),
            Descriptors.TPSA(Chem.MolFromSmiles(smiles)),
            Descriptors.HeavyAtomMolWt(Chem.MolFromSmiles(smiles)),
            Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles)),
            Descriptors.MolWt(Chem.MolFromSmiles(smiles)),
            GraphDescriptors.Chi0(Chem.MolFromSmiles(smiles)),
            Fragments.fr_ether(Chem.MolFromSmiles(smiles)),
            GraphDescriptors.BertzCT(Chem.MolFromSmiles(smiles)),
            GraphDescriptors.HallKierAlpha(Chem.MolFromSmiles(smiles)),
            Fragments.fr_C_O_noCOO(Chem.MolFromSmiles(smiles)),
            Fragments.fr_C_O(Chem.MolFromSmiles(smiles))
        ])),label)
        for (smiles,label) in tuple_ls
    ]

def Descriptors_bitter_nonbitter(tuple_ls):
    return [
        (torch.from_numpy(np.array([
            Descriptors.HeavyAtomMolWt(Chem.MolFromSmiles(smiles)),
            Lipinski.NumHDonors(Chem.MolFromSmiles(smiles)),
            Fragments.fr_ether(Chem.MolFromSmiles(smiles)),
            MolSurf.PEOE_VSA10(Chem.MolFromSmiles(smiles)),
            Lipinski.NHOHCount(Chem.MolFromSmiles(smiles)),
            EState_VSA.EState_VSA9(Chem.MolFromSmiles(smiles)),
            MolSurf.SlogP_VSA3(Chem.MolFromSmiles(smiles)),
            Lipinski.NumRotatableBonds(Chem.MolFromSmiles(smiles)),
            MolSurf.SMR_VSA6(Chem.MolFromSmiles(smiles)),
            MolSurf.PEOE_VSA9(Chem.MolFromSmiles(smiles)),
            GraphDescriptors.Kappa2(Chem.MolFromSmiles(smiles)),
            Crippen.MolLogP(Chem.MolFromSmiles(smiles)),
            Descriptors.FpDensityMorgan3(Chem.MolFromSmiles(smiles)),
            Descriptors.FpDensityMorgan2(Chem.MolFromSmiles(smiles)),
            Descriptors.FpDensityMorgan1(Chem.MolFromSmiles(smiles)),
            EState.MaxEStateIndex(Chem.MolFromSmiles(smiles)),
            EState.MaxAbsEStateIndex(Chem.MolFromSmiles(smiles)),
            GraphDescriptors.HallKierAlpha(Chem.MolFromSmiles(smiles)),
            EState_VSA.EState_VSA8(Chem.MolFromSmiles(smiles)),
            Lipinski.FractionCSP3(Chem.MolFromSmiles(smiles)),
            GraphDescriptors.Kappa3(Chem.MolFromSmiles(smiles)),
        ])),label)
        for (smiles,label) in tuple_ls
    ]

def kfolds_split_batchsize(ls, batchsize):
    kf = KFold(n_splits=5)
    kfolds=[]
    for train_idxs,val_idxs in kf.split(ls):
        trains = [ls[index] for index in train_idxs]
        trains = DataLoader(trains, batch_size=batchsize, shuffle=True, collate_fn=None, drop_last=False)
        val = [ls[index] for index in val_idxs]
        mols, labels = zip(*val)
        mols_tensor = torch.Tensor()
        for mol in list(mols):
            mols_tensor = torch.cat((mols_tensor, mol.unsqueeze(0)), 0)
        labels_tensor = torch.from_numpy(np.array(list(labels)))
        val = [mols_tensor, labels_tensor]
        kfolds.append((trains,val))
    return kfolds

def all_batchsize(ls, batchsize):
    trains = DataLoader(ls, batch_size=batchsize, shuffle=True, collate_fn=None, drop_last=False)
    return trains

class BoMLP(nn.Module):
    def __init__(self, n_feats):
        super(BoMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_feats, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
        )
        self.criteon = nn.CrossEntropyLoss()
    def forward(self, x):
        x = self.model(x.to(torch.float32))
        return x
    
