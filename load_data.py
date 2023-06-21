import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
import torch
from torch.utils.data import DataLoader
from dgllife.utils import RandomSplitter,mol_to_bigraph
from utils import collate



path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def drop_error_mols(df):
    """
    args:
        df <pandas.DataFrame>
    return:
        df <pandas.DataFrame>: discard unidentified molecules by rdkit
    """
    none_list=[]
    for i in range(df.shape[0]):
        if Chem.MolFromSmiles(df['smiles'][i]) is None:
            none_list.append(i)
    num_error_molecules = len(none_list)
    print("dropping {} unidentified molecules".format(num_error_molecules))
    df=df.drop(none_list)
    return df

def train_val_test_split_loader(total_g,total_y, batch_size):
    [train_tuples, val_tuples, test_tuples] = RandomSplitter.train_val_test_split(list(zip(total_g,total_y)), 0.8, 0.1, 0.1)   
    if batch_size == None:
        new_train_tuples = [(g, torch.from_numpy(np.array(label)).unsqueeze(0)) for (g, label) in list(train_tuples) ]
        new_val_tuples = [(g, torch.from_numpy(np.array(label)).unsqueeze(0)) for (g, label) in list(val_tuples) ]
        new_test_tuples = [(g, torch.from_numpy(np.array(label)).unsqueeze(0)) for (g, label) in list(test_tuples) ]
        return new_train_tuples, new_val_tuples, new_test_tuples
    # mini batch the data: (minibatch graphs, tensor of minibatch labels)
    train_loader = DataLoader(list(train_tuples), batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(list(val_tuples), batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    test_loader = DataLoader(list(test_tuples), batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    return train_loader, val_loader, test_loader


def kfold_split_loader(total_g, total_y, batch_size, k):
    k_list = list(RandomSplitter.k_fold_split(list(zip(total_g,total_y)), k=k))
    new_train_tuples_ls = []
    new_val_tuples_ls = []
    for (train_tuples, val_tuples) in k_list:
        if batch_size == None:
            new_train_tuples_ls.append( [(g, torch.from_numpy(np.array(label)).unsqueeze(0)) for (g, label) in list(train_tuples) ] ) 
            new_val_tuples_ls.append( [(g, torch.from_numpy(np.array(label)).unsqueeze(0)) for (g, label) in list(val_tuples) ] )
        else:
            train_loader = DataLoader(list(train_tuples), batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)
            val_loader = DataLoader(list(val_tuples), batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)
            new_train_tuples_ls.append(train_loader)
            new_val_tuples_ls.append(val_loader)
    # mini batch the data: (minibatch graphs, tensor of minibatch labels)
    return new_train_tuples_ls,new_val_tuples_ls

def all_loader(total_g, total_y, batch_size):
    tuples = list(zip(total_g,total_y))
    if batch_size == None:
        new_tuples = [(g, torch.from_numpy(np.array(label)).unsqueeze(0)) for (g, label) in tuples ]
        print(1)
        return new_tuples 
    # mini batch the data: (minibatch graphs, tensor of minibatch labels)
    loader = DataLoader(tuples, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    return loader

class load_data(object):
    def __init__(self,node_featurizer,edge_featurizer,batch_size,split_method, k):
        self.node_featurizer = node_featurizer
        self.edge_featurizer = edge_featurizer
        self.batch_size = batch_size
        self.split_method = split_method
        self.k = k
        
    def load_delaney(self):
        filepath = path  + '/dataset/public_dataset/regression/delaney.csv'
        df = pd.read_csv(filepath, header=0, sep=',') 
        PandasTools.AddMoleculeColumnToFrame(frame=df, smilesCol='smiles')
        df = drop_error_mols(df) # no unidentified smiles
        train_mols = list(df['ROMol'])
        total_g = [mol_to_bigraph(m, node_featurizer=self.node_featurizer, edge_featurizer=self.edge_featurizer) for m in train_mols]
        total_y = np.array(df['measured log solubility in mols per litre'], dtype=np.float64)
        if self.split_method == "train_val_test_split":
            return train_val_test_split_loader(total_g, total_y, self.batch_size)
        return kfold_split_loader(total_g, total_y, 64, self.k)
    
    def load_FreeSolv(self):
        filepath = path  + '/dataset/public_dataset/regression/FreeSolv.csv' #smiles,expt
        df = pd.read_csv(filepath, header=0, sep=',') 
        PandasTools.AddMoleculeColumnToFrame(frame=df, smilesCol='smiles')
        df = drop_error_mols(df) # no unidentified smiles
        train_mols = list(df['ROMol'])
        total_g = [mol_to_bigraph(m, node_featurizer=self.node_featurizer, edge_featurizer=self.edge_featurizer) for m in train_mols]
        total_y = np.array(df['expt'], dtype=np.float64)
        if self.split_method == "train_val_test_split":
            return train_val_test_split_loader(total_g, total_y, self.batch_size)
        return kfold_split_loader(total_g, total_y, 64,self.k)

    def load_Lipop(self):
        filepath = path  + '/dataset/public_dataset/regression/Lipop.csv' #smiles,exp
        df = pd.read_csv(filepath, header=0, sep=',') 
        PandasTools.AddMoleculeColumnToFrame(frame=df, smilesCol='smiles')
        df = drop_error_mols(df) # no unidentified smiles
        train_mols = list(df['ROMol'])
        total_g = [mol_to_bigraph(m, node_featurizer=self.node_featurizer, edge_featurizer=self.edge_featurizer) for m in train_mols]
        total_y = np.array(df['exp'], dtype=np.float64)
        if self.split_method == "train_val_test_split":
            return train_val_test_split_loader(total_g, total_y, self.batch_size)
        return kfold_split_loader(total_g, total_y, 128,self.k)
    
    def load_Malaria(self):
        filepath = path  + '/dataset/public_dataset/regression/Malaria.csv' #id,smiles,activity
        df = pd.read_csv(filepath, header=0, sep=',') 
        PandasTools.AddMoleculeColumnToFrame(frame=df, smilesCol='smiles')
        df = drop_error_mols(df) # no unidentified smiles
        train_mols = list(df['ROMol'])
        total_g = [mol_to_bigraph(m, node_featurizer=self.node_featurizer, edge_featurizer=self.edge_featurizer) for m in train_mols]
        total_y = np.array(df['activity'], dtype=np.float64)
        if self.split_method == "train_val_test_split":
            return train_val_test_split_loader(total_g, total_y, self.batch_size)
        return kfold_split_loader(total_g, total_y, 256,self.k)
    
    def load_photovoltaic(self):
        filepath = path  + '/dataset/public_dataset/regression/photovoltaic.txt' #
        df = pd.read_csv(filepath, header=0, sep=' ') 
        df.columns = ['smiles', 'value']
        PandasTools.AddMoleculeColumnToFrame(frame=df, smilesCol='smiles')
        df = drop_error_mols(df) # no unidentified smiles
        train_mols = list(df['ROMol'])
        total_g = [mol_to_bigraph(m, node_featurizer=self.node_featurizer, edge_featurizer=self.edge_featurizer) for m in train_mols]
        total_y = np.array(df['value'], dtype=np.float64)
        if self.split_method == "train_val_test_split":
            return train_val_test_split_loader(total_g, total_y, self.batch_size)
        return kfold_split_loader(total_g, total_y, 512,self.k)
     
    def load_N6512(self):
        filepath = path  + '/dataset/public_dataset/classification/smiles_cas_N6512.smi' 
        df = pd.read_csv(filepath, header=None, sep='\t') 
        df.columns = ['smiles', 'CAS_NO', 'activity']
        PandasTools.AddMoleculeColumnToFrame(frame=df, smilesCol='smiles')
        df = drop_error_mols(df) # 6 unidentified smiles
        train_mols = list(df['ROMol'])
        total_g = [mol_to_bigraph(m, node_featurizer=self.node_featurizer, edge_featurizer=self.edge_featurizer) for m in train_mols]
        total_y = np.array(df['activity'], dtype=np.int64)
        if self.split_method == "train_val_test_split":
            return train_val_test_split_loader(total_g, total_y, self.batch_size)
        return kfold_split_loader(total_g, total_y, 256,self.k)
     
    def load_HIV(self):
        filepath = path  + '/dataset/public_dataset/classification/hiv.txt' 
        df = pd.read_csv(filepath, header=None, sep=' ') 
        df.columns = ['smiles', 'class']
        print(df.count()['smiles'])
        PandasTools.AddMoleculeColumnToFrame(frame=df, smilesCol='smiles')
        df = drop_error_mols(df) # no unidentified smiles
        train_mols = list(df['ROMol'])
        total_g = [mol_to_bigraph(m, node_featurizer=self.node_featurizer, edge_featurizer=self.edge_featurizer) for m in train_mols]
        total_y = np.array(df['class'], dtype=np.int64)
        if self.split_method == "train_val_test_split":
            return train_val_test_split_loader(total_g, total_y, self.batch_size)
        return kfold_split_loader(total_g, total_y, 1024,self.k)
    
    def load_ClinTox2(self):
        filepath = path  + '/dataset/public_dataset/classification/ClinTox2.csv' 
        df = pd.read_csv(filepath, header=0, sep=',') #index,smiles,FDA_APPROVED,CT_TOX
        PandasTools.AddMoleculeColumnToFrame(frame=df, smilesCol='smiles')
        df = drop_error_mols(df) # no unidentified smiles
        train_mols = list(df['ROMol'])
        total_g = [mol_to_bigraph(m, node_featurizer=self.node_featurizer, edge_featurizer=self.edge_featurizer) for m in train_mols]
        total_y = np.array(df['CT_TOX'], dtype=np.int64)
        if self.split_method == "train_val_test_split":
            return train_val_test_split_loader(total_g, total_y, self.batch_size)
        return kfold_split_loader(total_g, total_y, 128,self.k)
                 
    def load_BBBP2(self):
        filepath = path  + '/dataset/public_dataset/classification/BBBP2.csv' 
        df = pd.read_csv(filepath, header=0, sep=',') #index,smiles,p_np
        PandasTools.AddMoleculeColumnToFrame(frame=df, smilesCol='smiles')
        df = drop_error_mols(df) # no unidentified smiles
        train_mols = list(df['ROMol'])
        total_g = [mol_to_bigraph(m, node_featurizer=self.node_featurizer, edge_featurizer=self.edge_featurizer) for m in train_mols]
        total_y = np.array(df['p_np'], dtype=np.int64)
        if self.split_method == "train_val_test_split":
            return train_val_test_split_loader(total_g, total_y, self.batch_size)
        return kfold_split_loader(total_g, total_y, 128,self.k)
    
    def load_BACE(self):
        filepath = path  + '/dataset/public_dataset/classification/BACE.csv' 
        df = pd.read_csv(filepath, header=0, sep=',') #smiles,Class
        PandasTools.AddMoleculeColumnToFrame(frame=df, smilesCol='smiles')
        df = drop_error_mols(df) # no unidentified smiles
        train_mols = list(df['ROMol'])
        total_g = [mol_to_bigraph(m, node_featurizer=self.node_featurizer, edge_featurizer=self.edge_featurizer) for m in train_mols]
        total_y = np.array(df['Class'], dtype=np.int64)
        if self.split_method == "train_val_test_split":
            return train_val_test_split_loader(total_g, total_y, self.batch_size)
        return kfold_split_loader(total_g, total_y, 128,self.k)

    # batch_size can be use 
    def load_bs_train_val(self):
        filepath = path  + '/dataset/bitter_sweet_nonbitter/bs_train_val.csv' 
        df = pd.read_csv(filepath, header=0, sep=',')
        print(df.count())
        PandasTools.AddMoleculeColumnToFrame(frame=df, smilesCol='smiles')
        df = drop_error_mols(df) # no unidentified smiles
        train_mols = list(df['ROMol'])
        total_g = [mol_to_bigraph(m, node_featurizer=self.node_featurizer, edge_featurizer=self.edge_featurizer) for m in train_mols]
        total_y = np.array(df['Taste'], dtype=np.int64)
        if self.split_method == "train_val_test_split":
            return train_val_test_split_loader(total_g, total_y, self.batch_size)
        elif self.split_method == "kfold_split":
            return kfold_split_loader(total_g, total_y, self.batch_size, self.k)
        else:
            return all_loader(total_g, total_y, self.batch_size)  

    # batch_size can be use 
    def load_bn_train_val(self):
        filepath = path  + '/dataset/bitter_sweet_nonbitter/bn_train_val.csv'
        df = pd.read_csv(filepath, header=0, sep=',') #smiles,Taste
        print(df.count())
        PandasTools.AddMoleculeColumnToFrame(frame=df, smilesCol='smiles')
        df = drop_error_mols(df) # no unidentified smiles
        train_mols = list(df['ROMol'])
        total_g = [mol_to_bigraph(m, node_featurizer=self.node_featurizer, edge_featurizer=self.edge_featurizer) for m in train_mols]
        total_y = np.array(df['Taste'], dtype=np.int64)
        if self.split_method == "train_val_test_split":
            return train_val_test_split_loader(total_g, total_y, self.batch_size)
        elif self.split_method == "kfold_split":
            return kfold_split_loader(total_g, total_y, self.batch_size, self.k)
        else:
            return all_loader(total_g, total_y, self.batch_size)  

    # all loader, no fold
    def load_bs_test(self):
        filepath = path  + '/dataset/bitter_sweet_nonbitter/bs_test.csv' 
        df = pd.read_csv(filepath, header=0, sep=',')
        num = df.shape[0]
        print(num)
        PandasTools.AddMoleculeColumnToFrame(frame=df, smilesCol='smiles')
        df = drop_error_mols(df) # no unidentified smiles
        train_mols = list(df['ROMol'])
        total_g = [mol_to_bigraph(m, node_featurizer=self.node_featurizer, edge_featurizer=self.edge_featurizer) for m in train_mols]
        total_y = np.array(df['Taste'], dtype=np.int64)   
        test_loader = all_loader(total_g, total_y, num)  
        return test_loader
    
    # all loader, no fold
    def load_bn_test(self):
        filepath = path  + '/dataset/bitter_sweet_nonbitter/bn_test.csv' 
        df = pd.read_csv(filepath, header=0, sep=',')
        num = df.shape[0]
        print(num)
        PandasTools.AddMoleculeColumnToFrame(frame=df, smilesCol='smiles')
        df = drop_error_mols(df) # no unidentified smiles
        train_mols = list(df['ROMol'])
        total_g = [mol_to_bigraph(m, node_featurizer=self.node_featurizer, edge_featurizer=self.edge_featurizer) for m in train_mols]
        total_y = np.array(df['Taste'], dtype=np.int64)   
        test_loader = all_loader(total_g, total_y, num)  
        return test_loader
    
    def load_bs_all_test(self):
        return self.load_bs_train_val(),self.load_bs_test()
    
    def load_bn_all_test(self):
        return self.load_bn_train_val(),self.load_bn_test()

    def load_oringal(self):
        filepath = path  + '/dataset/bitter_sweet_nonbitter/skinny_bitter_sweet_nonbitter.csv' 
        df = pd.read_csv(filepath, header=0, sep=',')
        df = df.replace('Sweet',0).replace('Bitter',1).replace('Non-bitter-Sweet',0)
        num = self.batch_size
        print(df)
        PandasTools.AddMoleculeColumnToFrame(frame=df, smilesCol='smiles')
        df = drop_error_mols(df) # no unidentified smiles
        train_mols = list(df['ROMol'])
        total_g = [mol_to_bigraph(m, node_featurizer=self.node_featurizer, edge_featurizer=self.edge_featurizer) for m in train_mols]
        total_y = np.array(df['Taste'], dtype=np.int64)   
        loader = all_loader(total_g, total_y, num)  
        return loader        

    def load_oringal_(self):
        filepath = path  + '/dataset/bitter_sweet_nonbitter/secondary_bsn.csv' 
        df = pd.read_csv(filepath, header=0, sep=',')
        df = df.replace('Sweet',0).replace('Bitter',1).replace('Non-bitter-Sweet',0)
        num = self.batch_size
        PandasTools.AddMoleculeColumnToFrame(frame=df, smilesCol='smiles')
        df = drop_error_mols(df) # no unidentified smiles
        train_mols = list(df['ROMol'])
        total_g = [mol_to_bigraph(m, node_featurizer=self.node_featurizer, edge_featurizer=self.edge_featurizer) for m in train_mols]
        total_y = np.array(df['Taste'], dtype=np.int64)   
        loader = all_loader(total_g, total_y, num)  
        return loader    

if __name__ == "__main__":
    from features import canonical_atom_featurlizer, canonical_bond_featurlizer, weave_bond_featurlizer, afp_bond_featurlizer
    node_featurizer,  n_nfeats  = canonical_atom_featurlizer()
    edge_featurizer,  n_efeats  = canonical_bond_featurlizer()
    l = load_data(node_featurizer, edge_featurizer, 128, "kfold_split", 5)
    # print(len(l.load_bn_test()))
    # print(len(l.load_bs_test()))
    print(l.load_oringal())











