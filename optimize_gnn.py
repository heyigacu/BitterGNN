import os
import sys
path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) 
sys.path.append(os.path.join(path, 'gnn'))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .train import Train_Trainer
from .model import MyPredictor



def run(trainer=None,predictor_model=MyPredictor, if_input_edge_feat=True, node_gat = True,edge_gat=True,weave=True,mpnn=True,n_tasks=1):
    if trainer.if_kfold:
        metric_ls = []
        for i in range(trainer.k):
            model =  predictor_model(
                    n_node_feats=trainer.n_nfeats, 
                    n_edge_feats=trainer.n_efeats,
                    num_layers=2,
                    n_heads=5,
                    node_gat = node_gat,
                    edge_gat = edge_gat,
                    weave=weave,
                    mpnn = mpnn,
                    n_hidden_feats=50,
                    activation=F.relu,
                    attn_activation = nn.LeakyReLU(negative_slope=0.2),
                    attn_dropout = 0,
                    feat_dropout = 0.0,
                    n_tasks=n_tasks,
                    readout="WeightedSumAndMax"
                    )
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            metric = trainer.train(model, if_input_edge_feat, optimizer, trainer.train_kfold[i], trainer.val_kfold[i], epochs=100)
            metric_ls.append(metric)
        return metric_ls
    else:
        model =  predictor_model(
                n_node_feats=trainer.n_nfeats, 
                n_edge_feats=trainer.n_efeats,
                num_layers=2,
                n_heads=5,
                node_gat = node_gat,
                edge_gat = edge_gat,
                weave = weave,
                mpnn = mpnn,
                n_hidden_feats=50,
                activation=F.relu,
                attn_activation = nn.LeakyReLU(negative_slope=0.2),
                attn_dropout = 0,
                feat_dropout = 0.0,
                n_tasks=n_tasks,
                readout="WeightedSumAndMax"
                )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        return trainer.train(model, True, optimizer,trainer.train_loader, trainer.val_loader,epochs=100)


dataset_ls1 = ['load_delaney', 
              'load_FreeSolv',
              'load_Lipop',
              'load_Malaria',
              'load_photovoltaic',
              ]
dataset_ls2 = ['load_N6512',
              'load_HIV',
              'load_ClinTox2',
              'load_BBBP2',
              'load_BACE',]
combs = []
for  i in [True,False]:
    for  j in [True,False]:
        for  k in [True,False]:
            for  l in [True,False]:
                combs.append([i,j,k,l])
def reg():
    ds_reg = []
    for dataset in dataset_ls1:
        trainer = Train_Trainer(dataset=dataset,if_kfold=True,k=5) 
        para = []
        for i,paras in enumerate(combs):
            rp=[]
            for repeat in range(10):
                print("dataset{}-model:{}-repeat:{}".format(dataset,i,repeat))                
                rst = run(trainer=trainer,predictor_model=MyPredictor, if_input_edge_feat=True, node_gat = paras[0],edge_gat=paras[1],weave=paras[2],mpnn=paras[3],n_tasks=1)
                rp.append(rst)
            para.append(rp)
        ds_reg.append(para)
    ds_reg = np.array(ds_reg)
    ds_reg = ds_reg.reshape((-1,ds_reg.shape[-1]))
    save_path = os.path.join(path,'result/compare_result/public/regression.txt')
    np.savetxt(save_path,ds_reg) 

def cla():
    ds_class = []         
    for dataset in dataset_ls2:
        trainer = Train_Trainer(dataset=dataset,if_kfold=True,k=5) 
        para = []
        for i,paras in enumerate(combs):
            rp = []
            for repeat in range(10):
                print("dataset:{}-model:{}-repeat:{}".format(dataset,i,repeat))
                rst = run(trainer=trainer,predictor_model=MyPredictor, if_input_edge_feat=True, node_gat = paras[0],edge_gat=paras[1],weave=paras[2],mpnn=paras[3],n_tasks=2)
                rp.append(rst)
            para.append(rp)
        ds_class.append(para)
    ds_class = np.array(ds_class)
    ds_class = ds_class.reshape((-1,ds_class.shape[-1]))
    save_path = os.path.join(path,'result/compare_result/public/classification.txt')
    np.savetxt(save_path,ds_class) 

if __name__ == "__main__":
    reg()
    cla()
