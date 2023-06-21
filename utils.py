import dgl
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, average_precision_score,precision_score,f1_score,recall_score, roc_curve, auc, roc_auc_score


def collate(sample):
    graphs, labels = map(list,zip(*sample))
    batched_graph = dgl.batch(graphs)
    batched_graph.set_n_initializer(dgl.init.zero_initializer)
    batched_graph.set_e_initializer(dgl.init.zero_initializer)
    return batched_graph, torch.tensor(labels)

def accuracy(preds, labels):
    pred_cls = preds.argmax(-1).detach().cpu().numpy()
    true_label = labels.cpu().numpy()
    return sum(true_label==pred_cls) / true_label.shape[0]

def bi_classify_metrics(preds, labels):
    pred_cls = preds.argmax(-1).detach().cpu().numpy()
    true_label = labels.cpu().numpy()
    cm = confusion_matrix(true_label, pred_cls, labels=range(2))
    cm = cm.astype(np.float32)
    TP = cm[1][1]
    FP = cm[0][1]
    FN = cm[1][0] 
    TN = cm[0][0]
    SEN = round(TP / (TP + FN + 0.0001),3)
    SPE = round(TN / (TN + FP + 0.0001),3)
    REC = round(TP / (TP + FN + 0.0001),3)
    PRE = round(TP / (TP + FP + 0.0001),3)
    ACC = round((TP + TN) / (TP + FP + FN + TN + 0.0001), 3)
    F1 = round((2 * PRE * REC) / (PRE + REC + 0.0001),3)
    fpr, tpr, thresholds = roc_curve(true_label, pred_cls, pos_label=1)
    AUC = auc(fpr, tpr)
    return np.array([TP,FP,FN,TN,SEN,SPE,REC,PRE,ACC,F1,AUC])

def rmse(preds, labels):
    return torch.pow(torch.pow(labels-preds,2).mean(0),0.5)

def mse(preds, labels):
    return torch.pow(labels-preds,2).mean(0).detach().cpu().numpy()

def auc_cal(preds,labels):
    preds = preds.argmax(-1).detach().cpu().numpy()
    labels = labels.cpu().numpy()
    fpr, tpr, thresholds = roc_curve(labels, preds, pos_label=1)
    return auc(fpr, tpr)

def multiple_classify_metrics(preds, labels, n_class):
    preds = preds.argmax(-1).detach().cpu().numpy()
    trues = labels.cpu().numpy()
    scores = preds.detach().cpu().numpy()
    labels_onehot = np.eye(n_class)[trues]

    acc = accuracy_score(trues,preds)
    PR_micro = precision_score(trues, preds, average='micro')
    PR_macro = precision_score(trues, preds, average='macro')
    RC_micro = recall_score(trues, preds, average='micro')
    RC_macro = recall_score(trues, preds, average='macro')
    f1_micro = f1_score(trues, preds, average='micro')
    f1_macro = f1_score(trues, preds, average='macro')

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_class):
        fpr[i], tpr[i], _ = roc_curve(labels_onehot[:, i], scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(labels_onehot.ravel(), scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_class)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_class):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_class
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return np.array([acc, PR_micro, PR_macro, RC_micro, RC_macro, f1_micro, f1_macro ,roc_auc["micro"] ,roc_auc["macro"] ])



if __name__ == "__main__":
    # preds = torch.FloatTensor([[0.1,1],[-1,1]])
    # labels = torch.LongTensor([1,0])
    # print(preds.argmax(-1))
    #设置类别的数量
    num_classes = 10
    #需要转换的整数
    arr = [1,3,4,5,9]
    #将整数转为一个10位的one hot编码
    print(np.eye(10)[arr])

    
