
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import copy

from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('TkAgg')

from .early_stop import EarlyStopping
from .utils import *

PWD = os.path.abspath(os.path.dirname(__file__))

class train_gnn(object):
    ##############
    ## Binary-classify
    ##############    
    @staticmethod
    def train_bi_classify_kfolds(model, kfolds=None, edge=True, max_epochs=500, patience=10, save_folder=PWD+'/pretrained/',save_name='gnn.pth'):
        val_metrics = []
        for train_loader,val_loader in kfolds:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                device = "cpu"
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            early_stopping = EarlyStopping(save_folder,save_name,patience=patience)
            for epoch in range(1, max_epochs+1):
                model.train()
                for batch_idx,(train_graphs,train_labels) in enumerate(train_loader):
                    graphs, labels = train_graphs.to(device), train_labels.to(device)
                    if edge:
                        preds = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
                    else:
                        preds = model(graphs, graphs.ndata.pop('h'))
                    loss = CrossEntropyLoss()(preds, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    for val_graphs, val_labels in val_loader:
                        graphs, labels = val_graphs.to(device), val_labels.to(device)
                        if edge:
                            preds = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
                        else:
                            preds = model(graphs, graphs.ndata.pop('h'))
                        loss = CrossEntropyLoss()(preds, labels)
                        loss_val = loss.detach().item()
                        metrics_val = bi_classify_metrics(labels.cpu().numpy(), preds.detach().cpu().numpy())
                early_stopping(loss_val, model)
                if early_stopping.early_stop:
                    val_metrics.append(metrics_val)
                    # print("Early stopping")
                    break
                else:
                    if epoch == max_epochs:
                        val_metrics.append(metrics_val)
        # np.savetxt(PWD+'/models/val.txt', np.array(val_metrics).mean(0), fmt='%.02f')
        return np.array(val_metrics).mean(0)
    @staticmethod
    def train_bi_classify_all(model, all=None, edge=True, max_epochs=500, patience=10, save_folder=PWD+'/pretrained/',save_name='gnn.pth'):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = "cpu"
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        early_stopping = EarlyStopping(save_folder,save_name=save_name,patience=patience)
        rst = []
        for epoch in range(1, max_epochs+1):
            loss_train = 0.
            acc_train=0.
            auc_train=0.
            model.train()
            for batch_idx,(train_graphs,train_labels) in enumerate(all):
                graphs, labels = train_graphs.to(device), train_labels.to(device)
                if edge:
                    logits = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
                else:
                    logits = model(graphs, graphs.ndata.pop('h'))
                loss = CrossEntropyLoss()(logits, labels)
                loss_train += loss.detach().item()
                try:
                    acc,auc = acc_auc(labels.cpu().numpy(), logits.detach().cpu().numpy())
                except:
                    acc,auc =acc,auc
                acc_train += acc
                auc_train += auc
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_train /= (batch_idx+1)
            auc_train /= (batch_idx+1)
            acc_train /= (batch_idx+1)
            if epoch%1 == 0:
                print('loss:',loss_train,'ACC:',acc_train,'AUC:',auc_train)    
                rst.append(np.array([loss_train, acc_train, auc_train]))
            early_stopping(loss_train, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        plot_training(rst, save_folder+save_name+".png")
    @staticmethod
    def test_bi_classify(model, test=None, edge=True, save_path=PWD+'/pretrained/gnn.pth',classnames=['Bitter','Sweet']):
        state_dict = torch.load(save_path)
        model.load_state_dict(state_dict)
        model.eval()
        for i in list(test):
            graphs,labels = i
        if edge:
            preds = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
        else:
            preds = model(graphs, graphs.ndata.pop('h'))
        preds = preds.detach().cpu().numpy()
        rst = bi_classify_metrics(labels, preds, plot_cm=True, save_path=save_path, classnames=classnames)
        return rst

    ##############
    ## multi-classify
    ##############
    @staticmethod
    def train_multi_classify_kfolds(model, kfolds=None, edge=True, max_epochs=500, patience=10, save_folder=PWD+'/pretrained/',save_name='gnn.pth'):
        val_metrics = []
        for train_loader,val_loader in kfolds:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                device = "cpu"
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            early_stopping = EarlyStopping(save_folder,save_name,patience=patience)
            for epoch in range(1, max_epochs+1):
                model.train()
                for batch_idx,(train_graphs,train_labels) in enumerate(train_loader):
                    graphs, labels = train_graphs.to(device), train_labels.to(device)
                    if edge:
                        preds = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
                    else:
                        preds = model(graphs, graphs.ndata.pop('h'))
                    loss = CrossEntropyLoss()(preds, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    for val_graphs, val_labels in val_loader:
                        graphs, labels = val_graphs.to(device), val_labels.to(device)
                        if edge:
                            preds = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
                        else:
                            preds = model(graphs, graphs.ndata.pop('h'))
                        loss = CrossEntropyLoss()(preds, labels)
                        loss_val = loss.detach().item()
                        metrics_val = multi_classify_metrics(labels.cpu().numpy(), preds.detach().cpu().numpy())
                early_stopping(loss_val, model)
                if early_stopping.early_stop:
                    val_metrics.append(metrics_val)
                    # print("Early stopping")
                    break
                else:
                    if epoch == max_epochs:
                        val_metrics.append(metrics_val)
        return np.array(val_metrics).mean(0)
    @staticmethod
    def train_multi_classify_all(model, all=None, edge=True, max_epochs=500, patience=10, save_folder=PWD+'/pretrained/',save_name='gnn.pth', lr=0.001, weight_decay=0):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = "cpu"
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        early_stopping = EarlyStopping(save_folder,save_name=save_name,patience=patience)
        rst = []
        for epoch in range(1, max_epochs+1):
            loss_train = 0.
            acc_train=0.
            auc_train=0.
            model.train()
            for batch_idx,(train_graphs,train_labels) in enumerate(all):
                graphs, labels = train_graphs.to(device), train_labels.to(device)
                if edge:
                    logits = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
                else:
                    logits = model(graphs, graphs.ndata.pop('h'))
                loss = CrossEntropyLoss()(logits, labels)
                loss_train += loss.detach().item()
                acc,auc = acc_auc(labels.cpu().numpy(), logits.detach().cpu().numpy())
                acc_train += acc
                auc_train += auc
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_train /= (batch_idx+1)
            auc_train /= (batch_idx+1)
            acc_train /= (batch_idx+1)
            if epoch%1 == 0:
                print('loss:',loss_train,'ACC:',acc_train,'AUC:',auc_train)    
                rst.append(np.array([loss_train, acc_train, auc_train]))
            early_stopping(loss_train, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        plot_training(rst, save_folder+save_name+".png")
    @staticmethod
    def test_multi_classify(model, test=None, edge=True, save_path=PWD+'/pretrained/gnn.pth', classnames=['Bitter','Sweet','Sour','Salty','Umami','Kokumi','Astringent','Tasteless']):
        state_dict = torch.load(save_path)
        model.load_state_dict(state_dict)
        model.eval()
        for i in list(test):
            graphs,labels = i
        if edge:
            preds = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
        else:
            preds = model(graphs, graphs.ndata.pop('h'))
        preds = preds.detach().cpu().numpy()
        rst = multi_classify_metrics(labels, preds, plot_auc_curve=True, plot_confuse_matrix=True,savename='GNN', classnames=classnames)
        return rst
    
    ##############
    ## regress
    ##############
    @staticmethod
    def train_regress_kfolds(model0, kfolds=None, edge=True, max_epochs=500, patience=10, save_folder=PWD+'/pretrained/',save_name='gnn.pth'):
        val_metrics = []
        for train_loader,val_loader in kfolds:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                device = "cpu"
            model = copy.deepcopy(model0)
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            early_stopping = EarlyStopping(save_folder,save_name,patience=patience)
            for epoch in range(1, max_epochs+1):
                model.train()
                for batch_idx,(train_graphs,train_labels) in enumerate(train_loader):
                    graphs, labels = train_graphs.to(device), train_labels.to(device)
                    if edge:
                        preds = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
                    else:
                        preds = model(graphs, graphs.ndata.pop('h'))
                    loss = nn.MSELoss(reduction='mean')(labels.float(), preds.float().squeeze(1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    for val_graphs, val_labels in val_loader:
                        graphs, labels = val_graphs.to(device), val_labels.to(device)
                        if edge:
                            preds = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
                        else:
                            preds = model(graphs, graphs.ndata.pop('h'))
                        loss = nn.MSELoss(reduction='mean')(labels.float(), preds.float().squeeze(1))
                        loss_val = loss.detach().item()
                        metrics_val = regress_metrics(labels.cpu().numpy(), preds.detach().cpu().numpy())
                early_stopping(loss_val, model)
                if early_stopping.early_stop:
                    val_metrics.append(metrics_val)
                    # print("Early stopping")
                    break
                else:
                    if epoch == max_epochs:
                        val_metrics.append(metrics_val)        
        return np.array(val_metrics).mean(0)
    @staticmethod
    def train_regress_all(model, all=None, edge=True,max_epochs=500, patience=10, save_folder=PWD+'/pretrained/',save_name='gnn.pth'):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = "cpu"
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        early_stopping = EarlyStopping(save_folder,save_name=save_name,patience=patience)
        rst = []
        for epoch in range(1, max_epochs+1):
            loss_train = 0.
            mse_train = 0.
            model.train()
            for batch_idx,(train_graphs,train_labels) in enumerate(all):
                graphs, labels = train_graphs.to(device), train_labels.to(device)
                if edge:
                    logits = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
                else:
                    logits = model(graphs, graphs.ndata.pop('h'))
                loss = nn.MSELoss(reduction='mean')(labels.float(), logits.float().squeeze(1))
                loss_train += loss.detach().item()
                mse = mean_squared_error(labels.cpu().numpy(), logits.detach().cpu().numpy())
                mse_train += mse
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_train /= (batch_idx+1)
            mse_train /= (batch_idx+1)
            if epoch%1 == 0:
                print('loss:',loss_train,'MSE:',mse_train)    
                rst.append(np.array([loss_train, mse_train]))
            early_stopping(loss_train, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        plot_training(rst, save_folder+save_name+".png")
    @staticmethod
    def test_regress(model, test=None, edge=True, save_path=PWD+'/pretrained/gnn.pth'):
        state_dict = torch.load(save_path)
        model.load_state_dict(state_dict)
        model.eval()
        for i in list(test):
            graphs,labels = i
        if edge:
            preds = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
        else:
            preds = model(graphs, graphs.ndata.pop('h'))
        preds = preds.detach().cpu().numpy()
        rst = regress_metrics(labels, preds)
        return rst

class train_mlp_cnn(object):
    ##############
    ## Binary-classify
    ##############    
    @staticmethod
    def train_bi_classify_kfolds(model, kfolds=None, max_epochs=500, patience=10, save_folder=PWD+'/pretrained/',save_name='gnn.pth', lr=0.001, weight_decay=0):
        val_metrics = []
        for train_loader,val_loader in kfolds:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                device = "cpu"
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            early_stopping = EarlyStopping(save_folder,save_name,patience=patience)
            for epoch in range(1, max_epochs+1):
                model.train()
                for batch_idx,(train_features,train_labels) in enumerate(train_loader):
                    features, labels = train_features.to(device), train_labels.to(device)
                    preds = model(features)
                    loss = CrossEntropyLoss()(preds, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    for val_features, val_labels in val_loader:
                        features, labels = val_features.to(device), val_labels.to(device)
                        preds = model(features)
                        loss = CrossEntropyLoss()(preds, labels)
                        loss_val = loss.detach().item()
                        metrics_val = bi_classify_metrics(labels.cpu().numpy(), preds.detach().cpu().numpy())
                early_stopping(loss_val, model)
                if early_stopping.early_stop:
                    val_metrics.append(metrics_val)
                    # print("Early stopping")
                    break
                else:
                    if epoch == max_epochs:
                        val_metrics.append(metrics_val)
        # np.savetxt(PWD+'/models/val.txt', np.array(val_metrics).mean(0), fmt='%.02f')
        return np.array(val_metrics).mean(0)
    @staticmethod
    def train_bi_classify_all(model, all=None, max_epochs=500, patience=10, save_folder=PWD+'/pretrained/',save_name='gnn.pth', lr=0.001, weight_decay=0):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = "cpu"
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        early_stopping = EarlyStopping(save_folder,save_name=save_name,patience=patience)
        rst = []
        for epoch in range(1, max_epochs+1):
            loss_train = 0.
            acc_train=0.
            auc_train=0.
            model.train()
            for batch_idx,(train_features,train_labels) in enumerate(all):
                features, labels = train_features.to(device), train_labels.to(device)
                logits = model(features)
                loss = CrossEntropyLoss()(logits, labels)
                loss_train += loss.detach().item()
                try:
                    acc,auc = acc_auc(labels.cpu().numpy(), logits.detach().cpu().numpy())
                except:
                    acc,auc = acc,auc
                acc_train += acc
                auc_train += auc
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_train /= (batch_idx+1)
            auc_train /= (batch_idx+1)
            acc_train /= (batch_idx+1)
            if epoch%10 == 0:
                print('loss:',loss_train,'ACC:',acc_train,'AUC:',auc_train)    
                rst.append(np.array([loss_train, acc_train, auc_train]))
            early_stopping(loss_train, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        plot_training(rst, save_folder+save_name+".png")

    @staticmethod
    def test_bi_classify(model, test=None, plot_cm=True, save_path=PWD+'/pretrained/gnn.pth', classnames=['Bitter','Sweet']):
        state_dict = torch.load(save_path)
        model.load_state_dict(state_dict)
        model.eval()
        for i in list(test):
            features,labels = i
        preds = model(features)
        preds = preds.detach().cpu().numpy()
        rst = bi_classify_metrics(labels, preds, plot_cm=plot_cm, save_path=save_path, classnames=classnames)
        return rst

    ##############
    ## multi-classify
    ##############
    @staticmethod
    def train_multi_classify_kfolds(model, kfolds=None, max_epochs=500, patience=10, save_folder=PWD+'/pretrained/',save_name='gnn.pth'):
        val_metrics = []
        for train_loader,val_loader in kfolds:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                device = "cpu"
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            early_stopping = EarlyStopping(save_folder,save_name,patience=patience)
            for epoch in range(1, max_epochs+1):
                model.train()
                for batch_idx,(train_features,train_labels) in enumerate(train_loader):
                    features, labels = train_features.to(device), train_labels.to(device)
                    preds = model(features)
                    loss = CrossEntropyLoss()(preds, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    for val_features, val_labels in val_loader:
                        features, labels = val_features.to(device), val_labels.to(device)
                        preds = model(features)
                        loss = CrossEntropyLoss()(preds, labels)
                        loss_val = loss.detach().item()
                        metrics_val = multi_classify_metrics(labels.cpu().numpy(), preds.detach().cpu().numpy())
                early_stopping(loss_val, model)
                if early_stopping.early_stop:
                    val_metrics.append(metrics_val)
                    # print("Early stopping")
                    break
                else:
                    if epoch == max_epochs:
                        val_metrics.append(metrics_val)
        return np.array(val_metrics).mean(0)
    @classmethod
    def train_multi_classify_all(model, all=None, max_epochs=500, patience=10, save_folder=PWD+'/pretrained/',save_name='gnn.pth'):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = "cpu"
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        early_stopping = EarlyStopping(save_folder,save_name=save_name,patience=patience)
        rst = []
        for epoch in range(1, max_epochs+1):
            loss_train = 0.
            acc_train=0.
            auc_train=0.
            model.train()
            for batch_idx,(train_features,train_labels) in enumerate(all):
                features, labels = train_features.to(device), train_labels.to(device)
                logits = model(features)
                loss = CrossEntropyLoss()(logits, labels)
                loss_train += loss.detach().item()
                acc,auc = acc_auc(labels.cpu().numpy(), logits.detach().cpu().numpy())
                acc_train += acc
                auc_train += auc
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_train /= (batch_idx+1)
            auc_train /= (batch_idx+1)
            acc_train /= (batch_idx+1)
            if epoch%10 == 0:
                # print('loss:',loss_train,'ACC:',acc_train,'AUC:',auc_train)    
                rst.append(np.array([loss_train, acc_train, auc_train]))
            early_stopping(loss_train, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        plot_training(rst, save_folder+save_name+".png")
    @staticmethod
    def test_multi_classify(model, test=None, plot_auc_curve=True, plot_confuse_matrix=True, save_path=PWD+'/pretrained/gnn.pth', savename='GNN', classnames=['Bitter','Sweet','Sour','Salty','Umami','Kokumi','Astringent','Tasteless']):
        state_dict = torch.load(save_path)
        model.load_state_dict(state_dict)
        model.eval()
        for i in list(test):
            features,labels = i
        preds = model(features)
        preds = preds.detach().cpu().numpy()
        rst = multi_classify_metrics(labels, preds, plot_auc_curve=plot_auc_curve, plot_confuse_matrix=plot_confuse_matrix, savename=savename, classnames=classnames)
        return rst