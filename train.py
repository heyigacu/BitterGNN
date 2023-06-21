import time
import argparse

import numpy as np
import torch
from torch.optim import *
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F
from load_data import load_data
from utils import accuracy, mse, rmse, bi_classify_metrics, auc_cal
from model import MyPredictor
from features import canonical_bond_featurlizer, canonical_atom_featurlizer, featurize_atoms, featurize_bonds
import os

path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) 
default_savapath = os.path.join(path,'result/model/temp/temp.pth')
# parser
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Size of mini batch to train.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.0, help='drop out.')
parser.add_argument('--save-path', type=str, default=default_savapath,
                    help='path to save the model (should include file name)')
args = parser.parse_args(args=[])
cuda = not args.no_cuda and torch.cuda.is_available()
batch_size = args.batch_size
epochs = args.epochs 
lr = args.lr
dropout = args.dropout
savepath =args.save_path
node_featurizer=canonical_atom_featurlizer
edge_featurizer=canonical_bond_featurlizer
if_kfold = True
k=5

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if cuda:
    torch.cuda.manual_seed(args.seed)

class Train_Trainer(object):
    def __init__(self, 
                 cuda=cuda, 
                 batch_size=batch_size, 
                 dropout=dropout,
                 savepath=savepath,
                 node_featurizer=node_featurizer,
                 edge_featurizer=edge_featurizer,
                 classify_metrics=bi_classify_metrics,
                 if_kfold = if_kfold,
                 k=k,
                 dataset = "load_N6512",
                 ):
        """
        args: default arguments
        """
        self.cuda = cuda
        self.batch_size = batch_size
        self.dropout = dropout
        self.savepath = savepath
        self.k=k
        self.if_kfold =if_kfold
        self.classify_metrics = classify_metrics
        # load featurlizers
        node_featurizer, self.n_nfeats = node_featurizer()
        edge_featurizer, self.n_efeats = edge_featurizer()
        # load data
        if if_kfold:
            loader = load_data(node_featurizer=node_featurizer, edge_featurizer=edge_featurizer, batch_size=batch_size,split_method = "kfold_split", k=self.k)
            if dataset == "load_delaney":
                train_kfold, val_kfold = loader.load_delaney()
                classify = False
            elif dataset == "load_FreeSolv":
                train_kfold, val_kfold = loader.load_FreeSolv()
                classify = False
            elif dataset == "load_Lipop":
                train_kfold, val_kfold = loader.load_Lipop()
                classify = False
            elif dataset == "load_Malaria":
                train_kfold, val_kfold = loader.load_Malaria()
                classify = False
            elif dataset == "load_photovoltaic":
                train_kfold, val_kfold = loader.load_photovoltaic()
                classify = False
            elif dataset == "load_N6512":
                train_kfold, val_kfold = loader.load_N6512()
                classify = True               
            elif dataset == "load_HIV":
                train_kfold, val_kfold = loader.load_HIV()
                classify = True      
            elif dataset == "load_ClinTox2":
                train_kfold, val_kfold = loader.load_ClinTox2()
                classify = True      
            elif dataset == "load_BBBP2":
                train_kfold, val_kfold = loader.load_BBBP2()
                classify = True      
            elif dataset == "load_BACE":
                train_kfold, val_kfold = loader.load_BACE()    
                classify = True        
            elif dataset == "load_bs_train_val":
                train_kfold, val_kfold = loader.load_bs_train_val()  
                classify = True   
            elif dataset == "load_bn_train_val":
                train_kfold, val_kfold = loader.load_bn_train_val()   
                classify = True   
            elif dataset == "load_model_bitter_sweet":
                train_kfold, val_kfold = loader.load_bitter_nonbitter()    
                classify = True  
            elif dataset == "load_model_bitter_nonbitter":
                train_kfold, val_kfold = loader.load_bitter_nonbitter()    
                classify = True  
            else: 
                 raise ValueError
            self.train_kfold = train_kfold
            self.val_kfold = val_kfold
            self.classify = classify
            
        else:
            loader = load_data(node_featurizer=node_featurizer, edge_featurizer=edge_featurizer, batch_size=batch_size,split_method = "train_val_test_split", k=self.k)
            if dataset == "load_delaney":
                train_loader, val_loader, test_loader = loader.load_delaney()
                classify = False
            elif dataset == "load_FreeSolv":
                train_loader, val_loader, test_loader = loader.load_FreeSolv()
                classify = False
            elif dataset == "load_Lipop":
                train_loader, val_loader, test_loader = loader.load_Lipop()
                classify = False
            elif dataset == "load_Malaria":
                train_loader, val_loader, test_loader = loader.load_Malaria()
                classify = False
            elif dataset == "load_photovoltaic":
                train_loader, val_loader, test_loader = loader.load_photovoltaic()
                classify = False
            elif dataset == "load_N6512":
                train_loader, val_loader, test_loader = loader.load_N6512()
                classify = True               
            elif dataset == "load_HIV":
                train_loader, val_loader, test_loader = loader.load_HIV()
                classify = True      
            elif dataset == "load_ClinTox2":
                train_loader, val_loader, test_loader = loader.load_ClinTox2()
                classify = True      
            elif dataset == "load_BBBP2":
                train_loader, val_loader, test_loader = loader.load_BBBP2()
                classify = True      
            elif dataset == "load_BACE":
                train_loader, val_loader, test_loader = loader.load_BACE()    
                classify = True                  
            else: 
                 raise ValueError
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader
            self.classify = classify

    def train(self, model, if_input_edge_feat, optimizer, train_data, val_data, epochs):
        t = time.time()
        if cuda:
            model.cuda()
        mse_val_ls = []
        cls_metrics_ls = []
        for epoch in range(1, epochs+1):
            # train
            model.train()
            loss_train = 0
            acc_train = 0
            auc_train = 0
            mse_train = 0
            for i, (graphs, labels) in enumerate(train_data):
                if cuda:
                    labels = labels.cuda()
                    graphs = graphs.to('cuda:0')
                if if_input_edge_feat:
                    preds = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
                else:
                    preds = model(graphs, graphs.ndata.pop('h'))
                optimizer.zero_grad()
                if self.classify:
                    loss = CrossEntropyLoss()(preds, labels)
                else:
                    loss = nn.MSELoss(reduction='mean')(preds.float().squeeze(1), labels.float())
                loss.backward()
                optimizer.step()
                loss_train += loss.detach().item()
                if self.classify:
                    acc_train += accuracy(preds, labels)
                    auc_train += auc_cal(preds, labels)
                else:
                    mse_train += mse(preds.squeeze(1), labels)
            loss_train /= (i + 1)
            acc_train /= (i + 1)
            auc_train /= (i + 1)
            mse_train /= (i + 1)
            
            # validate
            model.eval()
            loss_val = 0.
            mse_val = 0.
            cls_metrics_val = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
            with torch.no_grad():
                for i, (graphs, labels) in enumerate(val_data):
                    if cuda:
                        labels = labels.cuda()
                        graphs = graphs.to('cuda:0')
                    if if_input_edge_feat:
                        preds = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
                    else:
                        preds = model(graphs, graphs.ndata.pop('h'))
                    if self.classify:
                        loss = CrossEntropyLoss()(preds, labels)
                    else:
                        loss = nn.MSELoss(reduction='mean')(preds.float().squeeze(1), labels.float())
                    loss_val += loss.detach().item()
                    if self.classify:
                        cls_metrics_val += self.classify_metrics(preds, labels)
                    else:
                        mse_val += mse(preds.squeeze(1), labels)
            loss_val /= (i + 1)
            mse_val /= (i + 1)
            cls_metrics_val /= (i + 1)

            if self.classify:
                print(f"epoch: {epoch}, loss_train: {loss_train:.3f}, accuracy_train: {acc_train:.3f}, loss_val: {loss_val:.3f}, accuracy_val: {cls_metrics_val[-3]:.3f}, auc_val: {cls_metrics_val[-1]:.3f}")
                if epoch % 10 == 0:
                    cls_metrics_ls.append(cls_metrics_val)
            else:
                print(f"epoch: {epoch}, loss_train: {loss_train:.3f}, mse_train: {mse_train:.3f}, loss_val: {loss_val:.3f}, mse_val: {mse_val:.3f}")
                if epoch % 10 == 0:
                    mse_val_ls.append(mse_val)                   
        torch.save({'model': model.state_dict()}, self.savepath)
        print("Optimization Finished!")
        print(f"total_cost_time: {time.time() - t:.3f}")
        if self.if_kfold:
            if self.classify:
                return cls_metrics_ls
            else:
                return mse_val_ls
        else:
            return self.test(model, if_input_edge_feat,self.test_loader)

    def test(self,model, if_input_edge_feat, test_data):
        model.eval()
        loss_test = 0
        mse_test = 0
        cls_metrics_test = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
        with torch.no_grad():
            for i, (graphs, labels) in enumerate(test_data):
                if cuda:
                    labels = labels.cuda()
                    graphs = graphs.to('cuda:0')
                if if_input_edge_feat:
                    preds = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
                else:
                    preds = model(graphs, graphs.ndata.pop('h'))
                if self.classify:
                    loss = CrossEntropyLoss()(preds, labels)
                else:
                    loss = nn.MSELoss(reduction='mean')(preds.float().squeeze(1), labels.float())
                loss_test += loss.detach().item()
                if self.classify:
                    cls_metrics_test += self.classify_metrics(preds, labels)
                else:
                    mse_test += mse(preds.squeeze(1), labels)
        loss_test /= (i + 1)
        if self.classify:
            print(f"loss_test: {loss_test:.3f}, accuracy_test: {cls_metrics_test[-3]:.3f}, auc: {cls_metrics_test[-1]:.3f}",)
            return cls_metrics_test
        else:
            print(f"loss_test: {loss_test:.3f}, mse_test: {mse_test:.3f}")
            return mse_test

# only for test!
class Test_Trainer(object):
    def __init__(self, 
                 cuda=cuda, 
                 batch_size=batch_size, 
                 dropout=dropout,
                 savepath=savepath,
                 node_featurizer=node_featurizer,
                 edge_featurizer=edge_featurizer,
                 classify_metrics=bi_classify_metrics,
                 dataset = None,
                 ):
        """
        args: default arguments
        """
        self.cuda = cuda
        self.batch_size = batch_size
        self.dropout = dropout
        self.savepath = savepath
        self.classify_metrics = classify_metrics
        # load featurlizers
        node_featurizer, self.n_nfeats = node_featurizer()
        edge_featurizer, self.n_efeats = edge_featurizer()
        loader = load_data(node_featurizer=node_featurizer, edge_featurizer=edge_featurizer, batch_size=batch_size,split_method = 'all', k=0)
        if dataset == "load_bs_all_test":
            train_data, test_data = loader.load_bs_all_test()
        else :
            train_data, test_data = loader.load_bn_all_test()
        self.train_data = train_data
        self.test_data = test_data

    def train(self, model, if_input_edge_feat, optimizer, epochs):
        t = time.time()
        if cuda:
            model.cuda()
        cls_metrics_ls = []
        for epoch in range(1, epochs+1):
            # train
            model.train()
            loss_train = 0
            cls_metrics_train = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
            for i, (graphs, labels) in enumerate(self.train_data):
                if cuda:
                    labels = labels.cuda()
                    graphs = graphs.to('cuda:0')
                if if_input_edge_feat:
                    preds = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
                else:
                    preds = model(graphs, graphs.ndata.pop('h'))
                optimizer.zero_grad()
                loss = CrossEntropyLoss()(preds, labels)
                loss.backward()
                optimizer.step()
                loss_train += loss.detach().item()
                cls_metrics_train += self.classify_metrics(preds, labels)
            loss_train /= (i + 1)
            cls_metrics_train /= (i + 1)    
            print(f"epoch: {epoch}, loss_train: {loss_train:.3f}, accuracy_train: {cls_metrics_train[-3]:.3f}, auc_train: {cls_metrics_train[-1]:.3f}")        
        # test
        model.eval()
        with torch.no_grad():
            for (graphs, labels) in self.test_data:
                if cuda:
                    labels = labels.cuda()
                    graphs = graphs.to('cuda:0')
                if if_input_edge_feat:
                    preds = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
                else:
                    preds = model(graphs, graphs.ndata.pop('h'))
                loss = CrossEntropyLoss()(preds, labels)
                cls_metrics_test = self.classify_metrics(preds, labels)
        print(self.savepath)
        torch.save({'model': model.state_dict()}, self.savepath)
        print("Optimization Finished!")
        print(f"total_cost_time: {time.time() - t:.3f}")
        return np.array(cls_metrics_test)
    







# only for test!
class All_Trainer(object):
    def __init__(self, 
                 cuda=cuda, 
                 batch_size=batch_size, 
                 dropout=dropout,
                 savepath=savepath,
                 node_featurizer=node_featurizer,
                 edge_featurizer=edge_featurizer,
                 classify_metrics=bi_classify_metrics,
                 dataset = None,
                 ):
        """
        args: default arguments
        """
        self.cuda = cuda
        self.batch_size = batch_size
        self.dropout = dropout
        self.savepath = savepath
        self.classify_metrics = classify_metrics
        # load featurlizers
        node_featurizer, self.n_nfeats = node_featurizer()
        edge_featurizer, self.n_efeats = edge_featurizer()
        loader = load_data(node_featurizer=node_featurizer, edge_featurizer=edge_featurizer, batch_size=batch_size,split_method = 'all', k=0)
        if dataset == "load_bs_all_test":
            train_data, test_data = loader.load_bs_all_test()
        else :
            train_data, test_data = loader.load_bn_all_test()
        self.train_data = train_data
        self.test_data = test_data

    def train(self, model, if_input_edge_feat, optimizer, epochs):
        t = time.time()
        if cuda:
            model.cuda()
        cls_metrics_ls = []
        for epoch in range(1, epochs+1):
            # train
            model.train()
            loss_train = 0
            cls_metrics_train = np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
            for i, (graphs, labels) in enumerate(self.train_data):
                if cuda:
                    labels = labels.cuda()
                    graphs = graphs.to('cuda:0')
                if if_input_edge_feat:
                    preds = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
                else:
                    preds = model(graphs, graphs.ndata.pop('h'))
                optimizer.zero_grad()
                loss = CrossEntropyLoss()(preds, labels)
                loss.backward()
                optimizer.step()
                loss_train += loss.detach().item()
                cls_metrics_train += self.classify_metrics(preds, labels)
            loss_train /= (i + 1)
            cls_metrics_train /= (i + 1)    
            print(f"epoch: {epoch}, loss_train: {loss_train:.3f}, accuracy_train: {cls_metrics_train[-3]:.3f}, auc_train: {cls_metrics_train[-1]:.3f}")        
        # test
        torch.save({'model': model.state_dict()}, self.savepath)

