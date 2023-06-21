import sys
import os
path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) 
sys.path.append(os.path.join(path, 'gnn'))
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from train import Train_Trainer, Test_Trainer, MyPredictor
import os
from features import atom_number_featurizer, bond_1_featurizer, canonical_atom_featurlizer, canonical_bond_featurlizer

compare_save_folder = os.path.join(path,'result/compare/bitter/')
model_save_folder = os.path.join(path,'result/model/')

def train_run(trainer=None, 
        test=False,
        predictor_model=MyPredictor, 
        if_input_edge_feat=True, 
        num_layers =2, 
        n_heads=5, 
        n_hidden_feats=50,
        attn_dropout=0.0,
        feat_dropout=0.0,
        readout="WeightedSumAndMax",
        xavier_normal =False,
        lr = 0.001,
        epochs = 120,
        ):
    if trainer.if_kfold:
        metric_ls = []
        for i in range(trainer.k):
            model =  predictor_model(
                    n_node_feats=trainer.n_nfeats, 
                    n_edge_feats=trainer.n_efeats,
                    num_layers=num_layers,
                    n_heads=n_heads,
                    node_gat = True,
                    edge_gat = True,
                    weave= True,
                    mpnn = True,
                    n_hidden_feats=n_hidden_feats,
                    activation=F.relu,
                    attn_activation = nn.LeakyReLU(negative_slope=0.2),
                    attn_dropout = attn_dropout,
                    feat_dropout = feat_dropout,
                    xavier_normal = xavier_normal,
                    n_tasks=2,
                    readout=readout
                    )
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            metric = trainer.train(model, if_input_edge_feat, optimizer, trainer.train_kfold[i], trainer.val_kfold[i], epochs)
            metric_ls.append(metric)
        print(metric_ls)
        return metric_ls
    else:
        model =  predictor_model(
                n_node_feats=trainer.n_nfeats, 
                n_edge_feats=trainer.n_efeats,
                num_layers=2,
                n_heads=5,
                node_gat = True,
                edge_gat = True,
                weave = True,
                mpnn = True,
                n_hidden_feats=50,
                activation=F.relu,
                attn_activation = nn.LeakyReLU(negative_slope=0.2),
                attn_dropout = 0,
                feat_dropout = 0.0,
                n_tasks=2,
                readout="WeightedSumAndMax"
                )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        return trainer.train(model, True, optimizer,trainer.train_loader, trainer.val_loader, epochs)

def train_paras_run(dataset=None, save_name='bs_canonical',epochs=500, paras = {},node_featurizer=canonical_atom_featurlizer, edge_featurizer=canonical_bond_featurlizer):
    trainer = Train_Trainer(dataset=dataset,if_kfold=True,k=5, batch_size=128, node_featurizer=node_featurizer, edge_featurizer=edge_featurizer) 
    layes = []
    for num_layers in paras['num_layers']:
        heads = []
        for n_heads in paras['n_heads']:
            feats = []
            for n_hidden_feats in paras['n_hidden_feats']:
                dropout=[]
                for feat_dropout in paras['feat_dropout']:
                    lrs = []
                    for lr in paras['lr']:
                        print("num_layers:{}-n_heads:{}-feats:{}-dropout:{}-lr:{}".
                                format(num_layers,n_heads,n_hidden_feats,feat_dropout,lr))
                        rst = train_run(trainer=trainer,predictor_model=MyPredictor, if_input_edge_feat=True, 
                                num_layers =num_layers, 
                                n_heads=n_heads, 
                                n_hidden_feats=n_hidden_feats,
                                attn_dropout=0.0,
                                feat_dropout=feat_dropout,
                                readout="WeightedSumAndMax",
                                xavier_normal =True, 
                                lr = lr,
                                epochs = epochs,
                                )
                        lrs.append(rst)
                    dropout.append(lrs)
                feats.append(dropout)
            heads.append(feats)    
        layes.append(heads)
    all_rst = np.array(layes)
    all_rst_save = all_rst.reshape((-1,all_rst.shape[-1]))
    # name = dataset.split('_')[1]
    all_paras_auc=all_rst.mean(-3)[:,:,:,:,:,:,-1]
    ls=list(np.where(all_paras_auc==np.max(all_paras_auc)))
    best_paras_metrics=all_rst.mean(-3)[ls[0][0], ls[1][0],ls[2][0],ls[3][0],ls[4][0],ls[5][0]]
    best_paras = (paras['num_layers'][int(ls[0])],
                  paras['n_heads'][int(ls[1])],
                  paras['n_hidden_feats'][int(ls[2])],
                  paras['feat_dropout'][int(ls[3])],
                  paras['lr'][int(ls[4])],
                  int((ls[5]+1)*10))
    np.savetxt(compare_save_folder+f'/gnn_allparas_{save_name}.txt',all_rst_save) 
    np.savetxt(compare_save_folder+f'/gnn_bestpara_{save_name}.txt',best_paras)
    np.savetxt(compare_save_folder+f'/gnn_val_{save_name}.txt',best_paras_metrics)
    return best_paras, best_paras_metrics

def test_run(trainer=None, 
        predictor_model=MyPredictor, 
        if_input_edge_feat=True, 
        num_layers = 3, 
        n_heads=5, 
        n_hidden_feats=50,
        attn_dropout=0.0,
        feat_dropout=0.0,
        readout="WeightedSumAndMax",
        xavier_normal = True,
        lr = 0.001,
        epochs = 200,
    ):
    model =  predictor_model(
            n_node_feats=trainer.n_nfeats, 
            n_edge_feats=trainer.n_efeats,
            num_layers=num_layers,
            n_heads=n_heads,
            node_gat = True,
            edge_gat = True,
            weave=True,
            mpnn = True,
            n_hidden_feats=n_hidden_feats,
            activation=F.relu,
            attn_activation = nn.LeakyReLU(negative_slope=0.2),
            attn_dropout = attn_dropout,
            feat_dropout = feat_dropout,
            xavier_normal = xavier_normal,
            n_tasks=2,
            readout=readout
            )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    metric = trainer.train(model, if_input_edge_feat, optimizer, epochs)
    return metric

def gnn_bs_canonical(best_bs_paras):
    print(best_bs_paras)
    (num_layers, n_heads, n_hidden_feats,feat_dropout,lr,epochs)=best_bs_paras
    # all mean no kfolds here, only for testing
    dataset="load_bs_all_test"
    trainer = Test_Trainer(dataset=dataset,savepath=os.path.join(model_save_folder,'bs_model/bs_canonical_predictor.pth'),node_featurizer=canonical_atom_featurlizer,edge_featurizer=canonical_bond_featurlizer) 
    rst = test_run(trainer=trainer,predictor_model=MyPredictor, if_input_edge_feat=True, 
            num_layers = num_layers ,
            n_heads = n_heads ,
            n_hidden_feats=n_hidden_feats,
            attn_dropout=0.0,
            feat_dropout=feat_dropout,
            readout="WeightedSumAndMax",
            xavier_normal =True, 
            lr = lr,
            epochs = epochs,
            )
    rst = np.array(rst)
    np.savetxt(compare_save_folder+'/gnn_test_bs_canonical.txt',rst)
    return rst

def gnn_bn_canonical(best_bn_paras):
    (num_layers, n_heads, n_hidden_feats,feat_dropout,lr,epochs)=best_bn_paras
    # all mean no kfolds here, only for testing
    dataset="load_bn_all_test"
    trainer = Test_Trainer(dataset=dataset,savepath=os.path.join(model_save_folder,'bn_model/bn_canonical_predictor.pth'),node_featurizer=canonical_atom_featurlizer,edge_featurizer=canonical_bond_featurlizer)
    rst = test_run(trainer=trainer,predictor_model=MyPredictor, if_input_edge_feat=True, 
            num_layers = num_layers ,
            n_heads = n_heads ,
            n_hidden_feats=n_hidden_feats,
            attn_dropout=0.0,
            feat_dropout=feat_dropout,
            readout = "WeightedSumAndMax",
            xavier_normal =True, 
            lr = lr,
            epochs = epochs,
            )
    rst = np.array(rst)
    np.savetxt(compare_save_folder+'/gnn_test_bn_canonical.txt',rst)
    return rst

def gnn_bs_single(best_bs_paras):
    print(best_bs_paras)
    (num_layers, n_heads, n_hidden_feats,feat_dropout,lr,epochs)=best_bs_paras
    # all mean no kfolds here, only for testing
    dataset="load_bs_all_test"
    trainer = Test_Trainer(dataset=dataset,savepath=os.path.join(model_save_folder,'bs_model/bs_single_predictor.pth'),node_featurizer=atom_number_featurizer,edge_featurizer=bond_1_featurizer) 
    rst = test_run(trainer=trainer,predictor_model=MyPredictor, if_input_edge_feat=True, 
            num_layers = num_layers ,
            n_heads = n_heads ,
            n_hidden_feats=n_hidden_feats,
            attn_dropout=0.0,
            feat_dropout=feat_dropout,
            readout="WeightedSumAndMax",
            xavier_normal =True, 
            lr = lr,
            epochs = epochs,
            )
    rst = np.array(rst)
    print(rst)
    np.savetxt(compare_save_folder+'/gnn_test_bs_single.txt',rst)
    return rst

def gnn_bn_single(best_bn_paras):
    (num_layers, n_heads, n_hidden_feats,feat_dropout,lr,epochs)=best_bn_paras
    # all mean no kfolds here, only for testing
    dataset="load_bn_all_test"
    trainer = Test_Trainer(dataset=dataset,savepath=os.path.join(model_save_folder,'bn_model/bn_single_predictor.pth'),node_featurizer=atom_number_featurizer,edge_featurizer=bond_1_featurizer)
    rst = test_run(trainer=trainer,predictor_model=MyPredictor, if_input_edge_feat=True, 
            num_layers = num_layers ,
            n_heads = n_heads ,
            n_hidden_feats=n_hidden_feats,
            attn_dropout=0.0,
            feat_dropout=feat_dropout,
            readout = "WeightedSumAndMax",
            xavier_normal =True, 
            lr = lr,
            epochs = epochs,
            )
    rst = np.array(rst)
    np.savetxt(compare_save_folder+'/gnn_test_bn_single.txt',rst)
    return rst

if __name__ == "__main__":
    paras_dic_bs_canon ={
    'num_layers':[2],
    'n_heads':[5],
    'n_hidden_feats':[50], 
    'feat_dropout':[0.2],
    'lr': [0.005],
    }
    paras_dic_bn_canon ={
        'num_layers':[2],
        'n_heads':[5],
        'n_hidden_feats':[50], 
        'feat_dropout':[0.3],
        'lr': [0.005],
    }

    # ok 
    paras_dic_bs_single ={
    'num_layers':[3],
    'n_heads':[5],
    'n_hidden_feats':[200], 
    'feat_dropout':[0.03],
    'lr': [0.0005],
    }
    # ok
    paras_dic_bn_single ={
    'num_layers':[3],
    'n_heads':[5],
    'n_hidden_feats':[200], 
    'feat_dropout':[0.03],
    'lr': [0.0005],
    }
    best_bs_canonical_paras, best_bs_canonical_val = train_paras_run(dataset="load_bs_train_val", epochs=300, save_name='bs_canonical',paras = paras_dic_bsn_canon, node_featurizer=canonical_atom_featurlizer,edge_featurizer=bond_1_featurizer)
    best_bn_canonical_paras, best_bn_canonical_val = train_paras_run(dataset="load_bn_train_val", epochs=300, save_name='bn_canonical',paras = paras_dic_bsn_canon, node_featurizer=canonical_atom_featurlizer,edge_featurizer=bond_1_featurizer)
    best_bs_single_paras, best_bs_single_val = train_paras_run(dataset="load_bs_train_val", epochs=500, save_name='bs_single',paras = paras_dic_bsn_single, node_featurizer=atom_number_featurizer,edge_featurizer=bond_1_featurizer)
    best_bn_single_paras, best_bn_single_val = train_paras_run(dataset="load_bn_train_val", epochs=500, save_name='bn_single',paras = paras_dic_bsn_single, node_featurizer=atom_number_featurizer,edge_featurizer=bond_1_featurizer)
    gnn_bs_canonical_test = gnn_bs_canonical(best_bs_canonical_paras)
    gnn_bn_canonical_test = gnn_bn_canonical(best_bn_canonical_paras)
    gnn_bs_single_test = gnn_bs_single(best_bs_single_paras)
    gnn_bn_single_test = gnn_bn_canonical(best_bn_single_paras)









