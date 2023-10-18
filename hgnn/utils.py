import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
dir = os.path.abspath(os.path.dirname(__file__))
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef,average_precision_score, accuracy_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score,auc, roc_curve
from itertools import cycle
from sklearn.metrics import f1_score,precision_score,recall_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,explained_variance_score, mean_absolute_percentage_error
from math import sqrt

def onehot(labels,n_class):
    """
    example:
        print(onehot(np.array([0,1,2,3,7]),8))
    """
    onehot = np.zeros((labels.shape[-1], n_class))
    for i, value in enumerate(labels):
        onehot[i, value] = 1
    return onehot

def de_onehot(labels):
    return np.argmax(labels, axis=1)

def Micro_OvR_AUC(labels, preds,):
    fpr, tpr, _ = roc_curve(labels.ravel(), preds.ravel())
    roc_auc = auc(fpr, tpr)
    return  fpr, tpr, roc_auc

def Macro_OvR_AUC(labels, preds):
    n_classes = labels.shape[1]
    fpr={}
    tpr={}
    roc_auc={}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average it and compute AUC
    mean_tpr /= n_classes
    return fpr_grid, mean_tpr, auc(fpr_grid, mean_tpr)

def Weighted_OvR_AUC(labels, preds, weighted_method="amount_divided"):
    n_classes = labels.shape[1]
    ls_num_equal1 = np.array([np.sum(labels[:,i] == 1) for i in range(n_classes) ])
    if weighted_method=="equally_divided":
        #default
        reverse = 1/ls_num_equal1
        total = np.sum(reverse)
        weights = reverse / total
    else:
        weights = ls_num_equal1
    fpr={}
    tpr={}
    auc_list=[]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:, i], preds[:, i])
        auc_list.append(roc_auc_score(labels[:, i], preds[:, i]))
    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    tpr_ls=[]
    for i in range(n_classes):
        tpr_ls.append(np.interp(fpr_grid, fpr[i], tpr[i]))  # linear interpolation

    weighted_mean_tpr = np.average(tpr_ls, axis=0, weights=weights)
    # Average it and compute AUC
    # provide three implements, the values are consistent
    weighted_auc1 = auc(fpr_grid, weighted_mean_tpr)
    # weighted_auc2 = np.average(auc_list, weights=weights)
    # weighted_auc3 = roc_auc_score(labels, preds, multi_class="ovr", average="weighted",)                        
    return fpr_grid, weighted_mean_tpr, weighted_auc1

def plot_auc(labels, preds, savename="MLP",classnames=['a','b','c']):
    save_path = os.path.join(dir,'models/{}_ROC_AUC.png'.format(savename))
    n_classes = labels.shape[1]
    fpr, tpr, roc_auc = dict(), dict(), dict()
    fpr["micro"],tpr["micro"],roc_auc['micro'] = Micro_OvR_AUC(labels, preds)
    fpr["macro"],tpr["macro"],roc_auc['macro'] = Macro_OvR_AUC(labels, preds)
    fpr["weighted"],tpr["weighted"],roc_auc['weighted'] = Weighted_OvR_AUC(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )
    plt.plot(
        fpr["weighted"],
        tpr["weighted"],
        label=f"weighted-average ROC curve (AUC = {roc_auc['weighted']:.2f})",
        color="#006400",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(['#005831', '#7f7522','#f58f98','#b2d235','#145b7d','#d3d7d4','#dec674','#afdfe4'])
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            labels[:, class_id],
            preds[:, class_id],
            name=f"ROC curve for {classnames[class_id]}",
            color=color,
            ax=ax,
        )
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves of One-vs-Rest multi-classification")
    plt.legend(prop = { "size": 8 })
    plt.savefig(save_path)
    return roc_auc['micro'],roc_auc['macro'],roc_auc['weighted']

def mape_test(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / (y_true+0.001) )) * 100

def smape(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

def regression_metrics(y_true, y_pred):
    r2 = r2_score(y_true,y_pred)
    mse = mean_squared_error(y_true,y_pred)
    rmse = np.sqrt(mean_squared_error(y_true,y_pred)) # mean_squared_error(y_true,y_pred,squared=False)
    evar = explained_variance_score(y_true,y_pred)
    mae = mean_absolute_error(y_true,y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    smape = smape(y_true, y_pred)
    return np.array([r2,rmse,evar,mae,smape])

def acc_auc(labels, preds,):
    n_class = preds.shape[-1]
    if len(list(labels.shape)) == 1:
        y_true = labels
        labels = onehot(labels,n_class)
    else:
        if labels.shape[1] == 1:
            labels = labels.squeeze()
            y_true = labels
            labels = onehot(labels,n_class)
        else:
            y_true = de_onehot(labels)
    y_pred = de_onehot(preds)
    if n_class >2:
        auc = roc_auc_score(labels, preds, multi_class="ovr")
    else:
        auc = roc_auc_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    return acc,auc



def bi_classify_metrics(labels, preds, plot_cm=False, save_path='MLP.png',classnames=['Negative','Positive']):
    """
    example:
    labels = np.array([
                    [1,0], [1,0], [1,0], [0,1], [1,0],
                    [0,1], [0,1], [0,1], [0,1], [0,1],
                    ])
    preds = np.array([
                    [0.8, 0.2], [0.2, 0.8], [0.8, 0.2], [0.2, 0.8], [0.8, 0.2], 
                    [0.2, 0.8], [0.8, 0.2], [0.2, 0.8], [0.8, 0.2], [0.8, 0.2], 
                    ])

    bi_classify_metrics(labels, preds, plot_cm=True, savename='MLP',classnames=['Negative','Positive'])
    
    """
    save_path = save_path+"_cm.png"
    if len(list(labels.shape)) == 1:
        y_true = labels
        labels = onehot(labels,2)
    else:
        if labels.shape[1] == 1:
            labels = labels.squeeze()
            y_true = labels
            labels = onehot(labels,2)
        else:
            y_true = de_onehot(labels)
    y_pred = de_onehot(preds)
    cm = confusion_matrix(y_true, y_pred) # default 1 as postive
    cm = cm.astype(np.float32)
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0] 
    tp = cm[1][1]
    tpr = rec = sen = round(tp / (tp+fn+0.0001),3)
    tnr = spe = round(tn / (tn+fp+0.0001), 3)
    pre = round(tp / (tp+fp+0.0001), 3)
    acc = round((tp+tn) / (tp+fp+fn+tn+0.0001),3) # equal: accuracy_score(y_true, y_pred)
    f1 = round((2*pre*rec) / (pre+rec+0.0001),3) # equal: f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred) # equal: mcc = (tp*tn - fp*fn) / ((tp+fp)*(fn+tp)*(fn+tn)*(fp+tn))**0.5
    ap = average_precision_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    if plot_cm:
        conf_matrix = pd.DataFrame(cm, index=classnames, columns=classnames)
        # plot size setting
        fig, ax = plt.subplots(figsize = (4.5,3.5))
        sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 19}, cmap="Blues")
        plt.ylabel('True label', fontsize=18)
        plt.xlabel('Predicted label', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.savefig(save_path, bbox_inches='tight') 
    return np.array([tn,fp,fn,tp,tpr,tnr,pre,acc,ap,f1,mcc,auc])

def simple_classify_metrics(labels, preds, plot_cm=False, save_path='MLP.png',classnames=['Negative','Positive']):
    y_true=labels
    y_pred = preds
    cm = confusion_matrix(y_true, y_pred) # default 1 as postive
    cm = cm.astype(np.float32)
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0] 
    tp = cm[1][1]
    tpr = rec = sen = round(tp / (tp+fn+0.0001),3)
    tnr = spe = round(tn / (tn+fp+0.0001), 3)
    pre = round(tp / (tp+fp+0.0001), 3)
    acc = round((tp+tn) / (tp+fp+fn+tn+0.0001),3) # equal: accuracy_score(y_true, y_pred)
    f1 = round((2*pre*rec) / (pre+rec+0.0001),3) # equal: f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred) # equal: mcc = (tp*tn - fp*fn) / ((tp+fp)*(fn+tp)*(fn+tn)*(fp+tn))**0.5
    ap = average_precision_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    if plot_cm:
        conf_matrix = pd.DataFrame(cm, index=classnames, columns=classnames)
        # plot size setting
        fig, ax = plt.subplots(figsize = (4.5,3.5))
        sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 19}, cmap="Blues")
        plt.ylabel('True label', fontsize=18)
        plt.xlabel('Predicted label', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.savefig(save_path, bbox_inches='tight') 
    return np.array([tn,fp,fn,tp,tpr,tnr,pre,acc,ap,f1,mcc,auc])


def multi_classify_metrics(labels, preds, average_method='weighted', plot_auc_curve=False, plot_confuse_matrix=False, savename="MLP",classnames=['','b','d','e','']):
    """
    args:
        labels<numpy array>: shape like (n_sample,), category code;
                             shape like (n_sample, 1), category code;
                             shape like (n_sample, n_class), one hot code
        preds<numpy 2-d array>: shape like (n_sample, n_class), come from calcualting by deep learning

    note: if come from torch calculation in gpu, please detach as below:
        labels = labels.cpu().numpy()
        preds = preds.detach().cpu().numpy()
    
    return:
        np.array([pre,rec,acc,f1,mcc,auc])

    example:
        labels = np.array([
                        [1, 0, 0], [1, 0, 0], [1, 0, 0],
                        [0, 1, 0], [0,  1, 0], [0, 1, 0],
                        [0, 0, 1], [0, 0, 1], [0, 0, 1],
                        ])
        preds = np.array([
                        [0.8, 0.1, 0.1], [0.2, 0.32, 0.48], [0.6, 0.1, 0.3],
                        [0.2, 0.5, 0.3], [0.1, 0.6, 0.3], [0.2, 0.75, 0.05],
                        [0.05, 0.05, 0.9], [0.1, 0.3, 0.6], [0.12, 0.8, 0.08],
                        ])
        multi_classify_metrics(labels, preds, 'micro', plot_auc_curve=True, plot_confuse_matrix=True, savename="MLP",classnames=['a','b','c'])
    """

    n_class = preds.shape[-1]
    if len(list(labels.shape)) == 1:
        y_true = labels
        labels = onehot(labels,n_class)
    else:
        if labels.shape[1] == 1:
            labels = labels.squeeze()
            y_true = labels
            labels = onehot(labels,n_class)
        else:
            y_true = de_onehot(labels)
    y_pred = de_onehot(preds)
    ##################################
    ## plot roc and calculate auc
    ##################################
    if plot_auc_curve:
        micro_auc, macro_auc, weighted_auc = plot_auc(labels, preds, savename=savename,classnames=classnames)
    else:
        if average_method == 'weighted':
            weighted_auc = roc_auc_score(labels, preds, multi_class="ovr", average="weighted",)   
        elif average_method == 'macro':
            macro_auc = roc_auc_score(labels, preds, multi_class="ovr", average="macro",)  
        else:
            micro_auc = roc_auc_score(labels, preds, multi_class="ovr", average="micro",)    
    if average_method == 'weighted':
        auc = weighted_auc
    elif average_method == 'macro':
        auc = macro_auc
    else:
        auc = micro_auc
    acc = accuracy_score(y_true, y_pred)
    ##################################
    ## plot cm and alculate tpr(sen) and tnr(spe)
    ##################################
    cm = confusion_matrix(y_true, y_pred)
    tpr = sen = cm.diagonal() / cm.sum(axis=1)
    tnr = []
    for i in range(cm.shape[0]):
        mask = np.ones(cm.shape[0], dtype=bool)
        mask[i] = False
        tnr.append(np.sum(cm[mask][:, mask]) / np.sum(cm[mask]))
    tnr = spe = np.array(tnr)
    ## equal to below:
    # FP = cm.sum(axis=0) - np.diag(cm)  
    # FN = cm.sum(axis=1) - np.diag(cm)
    # TP = np.diag(cm )
    # TN = cm.sum() - (FP + FN + TP)
    # FP = FP.astype(float) 
    # FN = FN.astype(float)
    # TP = TP.astype(float)
    # TN = TN.astype(float)
    # TPR = TP/(TP+FN)
    # TNR = TN/(TN+FP)
    if plot_confuse_matrix:
        save_path = os.path.join(dir,'models/{}_confusion_matrix.png'.format(savename))
        conf_matrix = pd.DataFrame(cm, index=classnames, columns=classnames)
        # plot size setting
        fig, ax = plt.subplots(figsize = (4.5,3.5))
        sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 19}, cmap="Blues")
        plt.ylabel('True label', fontsize=18)
        plt.xlabel('Predicted label', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.savefig(save_path, bbox_inches='tight')

    ##################################
    ## calculate pre,rec,acc,f1,mcc
    ##################################
    if average_method == 'weighted':
        pre = precision_score(y_true, y_pred, average='weighted')
        rec = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
    elif average_method == 'macro':
        pre = precision_score(y_true, y_pred, average='macro')
        rec =recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
    else:
        pre = precision_score(y_true, y_pred, average='micro')
        rec =recall_score(y_true, y_pred, average='micro')
        f1 = f1_score(y_true, y_pred, average='micro')

    mcc = matthews_corrcoef(y_true, y_pred)

    return np.array([pre,rec,acc,f1,mcc,auc])



def regress_metrics(labels, preds):
    return np.array(
            [
            r2_score(labels, preds),
            mean_absolute_error(labels, preds),
            mean_squared_error(labels, preds),
            sqrt(mean_squared_error(labels, preds)),
            ]
    )



def plot_training(data, path):
    """
    data = np.array([
    [0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01, 0.005],
    [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    [0.55, 0.65, 0.75, 0.8, 0.85, 0.9, 0.93, 0.96]
    ])
    plot_gnn(data,"test.png")
    """
    data=np.array(data).transpose()
    plt.figure()
    plt.plot(data[0], label='Loss')
    plt.plot(data[1], label='Accuracy')
    plt.plot(data[2], label='AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.legend()
    plt.title('Training Metrics')
    plt.savefig(path, dpi=300)
    plt.close()
