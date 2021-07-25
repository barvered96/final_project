import torch
from sklearn.metrics import roc_auc_score, precision_score, average_precision_score, confusion_matrix
import numpy as np
from augmentations import embed_data_mask
import torch.nn as nn
from statistics import mean
import pandas as pd
from sklearn.preprocessing import label_binarize

def make_default_mask(x):
    mask = np.ones_like(x)
    mask[:,-1] = 0
    return mask

def tag_gen(tag,y):
    return np.repeat(tag,len(y['data']))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  


def imputations_acc_justy(model,dloader,device):
    model.eval()
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model)
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,model.num_categories-1,:]
            y_outs = model.mlpfory(y_reps)
            # import ipdb; ipdb.set_trace()
            y_test = torch.cat([y_test,x_categ[:,-1].float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(m(y_outs), dim=1).float()],dim=0)
            prob = torch.cat([prob,m(y_outs)[:,-1].float()],dim=0)
     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    labels = list(set(y_test.numpy()))
    auc = roc_auc_score(y_score=label_binarize(y_pred.cpu(), labels), y_true=label_binarize(y_test.cpu(), labels), average='micro', multi_class="ovo")
    #auc = roc_auc_score(y_score=y_pred, y_true=y_test.cpu())
    prec = precision_score(y_pred=y_pred.cpu(), y_true=y_test.cpu(), average="macro")
    #aupr = average_precision_score(y_true=y_test.cpu(), y_score=prob.cpu())
    # fp = np.sum((y_pred.numpy() == 1) & (y_test.cpu().numpy() == 0))
    # tp = np.sum((y_pred.numpy() == 1) & (y_test.cpu().numpy() == 1))
    # fn = np.sum((y_pred.numpy() == 0) & (y_test.cpu().numpy() == 1))
    # tn = np.sum((y_pred.numpy() == 0) & (y_test.cpu().numpy() == 0))
    matrix = confusion_matrix(y_test.cpu(), y_pred.cpu())
    fp = matrix.sum(axis=0) - np.diag(matrix)
    fn = matrix.sum(axis=1) - np.diag(matrix)
    tp = np.diag(matrix)
    tn = matrix.sum() - (fp + fn + tp)
    tprs_mc, fprs_mc = [], []
    fp, fn, tp, tn = fp.astype(float), fn.astype(float), tp.astype(float), tn.astype(float)
    for i in range(len(fp)):
        tprs_mc.append(tp[i] / (tp[i] + fn[i]))
        fprs_mc.append(fp[i] / (fp[i] + tn[i]))
    auprs_multiclass = []
    for i in range(len(set(y_test.cpu().numpy()))):
        try:
            all_classes_except_i = set(y_test.cpu().numpy())
            all_classes_except_i.remove(i)
            binary_y_test = pd.DataFrame(y_test.cpu().numpy()).replace(i, -1000)
            binary_y_test = binary_y_test.replace(all_classes_except_i, 1)
            binary_y_test = binary_y_test.replace(-1000, 0)
            binary_y_pred = pd.Series(y_pred.cpu().numpy()).replace(i, -1000)
            binary_y_pred = binary_y_pred.replace(all_classes_except_i, 1)
            binary_y_pred = binary_y_pred.replace(-1000, 0)
            auprs_multiclass.append(average_precision_score(binary_y_test, binary_y_pred))
        except:
            print('failed key', i)
    return acc, auc, prec, mean(auprs_multiclass), mean(fprs_mc), mean(tprs_mc)


def multiclass_acc_justy(model,dloader,device):
    model.eval()
    vision_dset = True
    m = nn.Softmax(dim=1)
    y_test = torch.empty(0).to(device)
    y_pred = torch.empty(0).to(device)
    prob = torch.empty(0).to(device)
    with torch.no_grad():
        for i, data in enumerate(dloader, 0):
            x_categ, x_cont, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device)
            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)
            reps = model.transformer(x_categ_enc, x_cont_enc)
            y_reps = reps[:,model.num_categories-1,:]
            y_outs = model.mlpfory(y_reps)
            # import ipdb; ipdb.set_trace()   
            y_test = torch.cat([y_test,x_categ[:,-1].float()],dim=0)
            y_pred = torch.cat([y_pred,torch.argmax(m(y_outs), dim=1).float()],dim=0)

     
    correct_results_sum = (y_pred == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]*100
    return acc, auc


