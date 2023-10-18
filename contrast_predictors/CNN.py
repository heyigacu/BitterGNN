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
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from sklearn.model_selection import KFold
from sklearn.utils import shuffle

from rdkit import Chem,DataStructs
from rdkit.Chem import AllChem,Draw,PandasTools
from rdkit.Chem.EState import EState_VSA,EState
from rdkit.Chem import Descriptors,GraphDescriptors,MolSurf,QED,Crippen,Fragments,GraphDescriptors,Lipinski

path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

##########################################
## load_data
##########################################
def loader(filepath):
    df = pd.read_csv(filepath, header=0, sep='\t') #smiles,Taste
    return list(zip(df['Smiles'],df['Label']))

##########################################
## feature
##########################################

def imgpath2tensor(tuple_ls, transform):
    return [(transform(Image.open(img_path).convert('RGB')),label) for (img_path,label) in tuple_ls]

def cnn_prepare(tuple_ls, imgs_dir, data_aug=True):
    # generate_figures
    ls = []
    for i,(smiles,label) in enumerate(tuple_ls):
        draw = Draw.MolToImage(Chem.MolFromSmiles(smiles), size=(32,32))
        draw.save(imgs_dir+'fig{}.png'.format(i))
        ls.append((imgs_dir+'fig{}.png'.format(i), label))
    # data_augmentation
    original_transform = transforms.Compose([transforms.ToTensor()])
    original = imgpath2tensor(ls,original_transform)
    if not data_aug:
        return original
    flip_transform = transforms.Compose([
        transforms.RandomChoice([
            transforms.RandomHorizontalFlip(p=1), 
            transforms.RandomVerticalFlip(p=1),
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=1), 
                transforms.RandomHorizontalFlip(p=1)
            ]),      
        ]),
        transforms.ToTensor(),
        ])
    rotate_transform = transforms.Compose([
        transforms.RandomChoice([
        transforms.RandomRotation(degrees=(90, 90)) ,
        transforms.RandomRotation(degrees=(180, 180)),
        transforms.RandomRotation(degrees=(270, 270)),
        ]),
        transforms.ToTensor(),
    ])
    zoom_transform = transforms.Compose([
        transforms.RandomChoice([
            transforms.Compose([
                transforms.Resize((16, 16)), 
                transforms.Pad(padding=8, fill=(255,255,255), padding_mode="constant"),
            ]),
            transforms.Compose([
                transforms.Resize((28, 28)), 
                transforms.Pad(padding=2, fill=(255,255,255), padding_mode="constant"),
            ]),
            transforms.Compose([
                transforms.Resize((24, 24)), 
                transforms.Pad(padding=4, fill=(255,255,255), padding_mode="constant"),
            ]),
            transforms.Compose([
                transforms.Resize((20, 20)), 
                transforms.Pad(padding=6, fill=(255,255,255), padding_mode="constant"),
            ]),     
        ]),
        transforms.ToTensor(),
    ])

    flip_tuple_ls= [ls[i] for i in [np.random.randint(0,len(ls)) for i in range(int(len(ls)/2))]]
    flip = imgpath2tensor(flip_tuple_ls,flip_transform)

    rotate_tuple_ls= [ls[i] for i in [np.random.randint(0,len(ls)) for i in range(int(len(ls)/2))]]
    rotate = imgpath2tensor(rotate_tuple_ls,rotate_transform)

    zoom_tuple_ls= [ls[i] for i in [np.random.randint(0,len(ls)) for i in range(int(len(ls)/2))]]
    zoom = imgpath2tensor(zoom_tuple_ls,zoom_transform)

    return original+flip+rotate+zoom


##########################################
## processing data: kfold and batchsize
##########################################
def kfolds_split_batchsize(ls, batchsize):
    kf = KFold(n_splits=5)
    kfolds=[]
    for train_idxs,val_idxs in kf.split(ls):
        trains = [ls[index] for index in train_idxs]
        trains = shuffle(trains)
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

##########################################
## models
##########################################
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
        x = self.model(x)
        return x
    
class BoCNN(nn.Module):
    def  __init__(self):
        super(BoCNN,self).__init__()
        self.conv_unit = nn.Sequential(
            # x: [b,3,32,32] => [b,32,30,30] => [b,32,15,15]
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),
            # x: [b,32,15,15] => [b,64,13,13] => [b,64,6,6]            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),
            # x: [b,64,6,6] => [b,32,4,4]
            nn.Conv2d(64, 32,kernel_size=3,stride=1,padding=0)
        )
        self.fc_unit = nn.Sequential(
            nn.Linear(32*4*4,120),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(120,2),
        )
        self.criteon = nn.CrossEntropyLoss()

    def forward(self,x):
        batchsz = x.size(0)
        # [b,3,32,32] => [b,32,4,4]
        x=self.conv_unit(x)
        # [b,32,4,4] => [b,32*4*4]
        x=x.view(batchsz,32*4*4)
        # [b,32*4*4] => [b,2]
        logits =self.fc_unit(x)
        pred=F.softmax(logits,dim=1)
        loss =self.criteon(logits, pred)
        return logits
