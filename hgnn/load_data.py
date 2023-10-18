import dgl
import torch
import random
import numpy as np
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import KFold, StratifiedKFold


def collate(sample):
    graphs, labels = map(list,zip(*sample))
    batched_graph = dgl.batch(graphs)
    batched_graph.set_n_initializer(dgl.init.zero_initializer)
    batched_graph.set_e_initializer(dgl.init.zero_initializer)
    return batched_graph, torch.tensor(labels)

def load_data_all_notorch(tuple_ls):
    """
    args:
        ls: [(feature,label)]
    """
    random.shuffle(tuple_ls)
    return tuple_ls

def load_data_all_batchsize(tuple_ls, batchsize, graph=False, drop_last=False):
    """
    args:
        ls: [(feature,label)]
        batchsize: int
    """
    if not graph:
        return DataLoader(tuple_ls, batch_size=batchsize, shuffle=True, collate_fn=None, drop_last=drop_last)
    else:
        return DataLoader(tuple_ls,batch_size=batchsize, shuffle=True, collate_fn=collate, drop_last=drop_last)

def load_data_kfold_notorch(tuple_ls, Stratify):
    """
    args:
        ls: [(feature,label)]
    """
    random.shuffle(tuple_ls)
    features, labels = list(zip(*tuple_ls))
    if Stratify:
        kf = StratifiedKFold(n_splits=5,shuffle=True)
    else:
        kf = KFold(n_splits=5,shuffle=True)
    kfolds=[]
    for train_idxs,val_idxs in kf.split(features, labels):
        trains = [tuple_ls[index] for index in train_idxs]
        vals = [tuple_ls[index] for index in val_idxs]
        kfolds.append((trains,vals))
    return kfolds

def load_data_kfold_batchsize(tuple_ls, batchsize, Stratify=True, graph=False, drop_last=False):
    """
    args:
        ls: [(feature,label)]
        batchsize: int
    """
    features, labels = list(zip(*tuple_ls))
    if Stratify:
        kf = StratifiedKFold(n_splits=5,shuffle=True)
    else:
        kf = KFold(n_splits=5,shuffle=True)
    kfolds=[]
    if not graph:
        for train_idxs,val_idxs in kf.split(features, labels):
            trains = [tuple_ls[index] for index in train_idxs]
            trains = DataLoader(trains, batch_size=batchsize, shuffle=True, collate_fn=None, drop_last=drop_last)
            vals = [tuple_ls[index] for index in val_idxs]
            vals = DataLoader(vals,batch_size=len(vals), shuffle=True,)
            kfolds.append((trains,vals))
    else:
        if Stratify:
            for train_idxs,val_idxs in kf.split(features, labels):
                trains = [tuple_ls[index] for index in train_idxs]
                trains = DataLoader(trains, batch_size=batchsize, shuffle=True, collate_fn=collate, drop_last=drop_last)
                vals = [tuple_ls[index] for index in val_idxs]
                vals = DataLoader(vals,batch_size=len(vals), shuffle=True, collate_fn=collate, drop_last=drop_last)
                kfolds.append((trains,vals))
        else:
            for train_idxs,val_idxs in kf.split(labels):
                trains = [tuple_ls[index] for index in train_idxs]
                trains = DataLoader(trains, batch_size=batchsize, shuffle=True, collate_fn=collate, drop_last=drop_last)
                vals = [tuple_ls[index] for index in val_idxs]
                vals = DataLoader(vals,batch_size=len(vals), shuffle=True, collate_fn=collate, drop_last=drop_last)
                kfolds.append((trains,vals))
    return kfolds




def load_data(tuple_ls, featurizer=None, if_all=False, Stratify=False, if_torch=False, batchsize=32, graph=True, drop_last=False):
    mols, labels =  map(list,zip(*tuple_ls))
    features = np.array([featurizer(mol) for mol in mols])
    tuple_ls =  list(zip(features, labels))
    if if_all:
        if if_torch:
            return load_data_all_batchsize(tuple_ls, batchsize, graph, drop_last)
        else:
            return load_data_all_notorch(tuple_ls)
    else:
        if if_torch:
            return load_data_kfold_batchsize(tuple_ls, batchsize, Stratify, graph, drop_last)
        else:
            return load_data_kfold_notorch(tuple_ls, Stratify)
        


