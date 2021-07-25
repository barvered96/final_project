import torch
from torch import nn
from models import SAINT, SAINT_vision
import time
from data import data_prep,DataSetCatCon
import pandas as pd
import wget
from pathlib import Path
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import count_parameters, imputations_acc_justy
from augmentations import embed_data_mask
from sklearn.model_selection import StratifiedKFold, KFold
import os
import numpy as np
from imblearn.over_sampling import SVMSMOTE
import torch, gc

gc.collect()
torch.cuda.empty_cache()
sm = SVMSMOTE(random_state=42, k_neighbors=2)

def data_download(dataset):
    url= None
    if dataset == '1995_income':
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        dataset_name = '1995_income'
        target = 14
    elif dataset == 'bank_marketing':
        dataset_name = 'bank-full'
        target = 16
    elif dataset == 'autos':
        dataset_name = 'autos'
        target = 25
    elif dataset == 'libras':
        dataset_name = 'libras'
        target = 90
    elif dataset == 'cloud':
        dataset_name = 'cloud'
        target = 6
    elif dataset == 'abalon':
        dataset_name = 'abalon'
        target = 8
    elif dataset == 'baseball':
        dataset_name = 'baseball'
        target = 16
    elif dataset == 'blood':
        dataset_name = 'blood'
        target = 4
    elif dataset == 'qsar_bio':
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00254/biodeg.csv'
        target = 41
        dataset_name = 'qsar_bio'
    elif dataset == 'online_shoppers':
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv'
        target = 17
        dataset_name = 'online_shoppers'
    elif dataset == 'blastchar':
        dataset_name = 'blastchar'
        target = 20
    elif dataset == 'htru2':
        dataset_name = 'htru2'
        target = 8
    elif dataset == 'shrutime':
        dataset_name = 'Churn_Modelling'
        target = 11
    elif dataset == 'spambase':
        dataset_name = 'spambase'
        target = 57
    elif dataset == 'acute-nephritis':
        dataset_name = 'acute-nephritis'
        target = 7
    elif dataset == 'loan_data':
        dataset_name = 'loan_data'
        target = 13
    elif dataset == 'arcene':
        dataset_name = 'arcene'
        target = 'target'
    elif dataset == 'creditcard':
        dataset_name = 'creditcard'
        target = 30
    elif dataset == 'arrhythmia':
        dataset_name = 'arrhythmia'
        target = 226
    elif dataset == 'forest':
        dataset_name = 'forest'
        target = 49
    elif dataset == 'kidney':
        dataset_name = 'kidney'
        target = 6
    elif dataset == 'kdd99':
        dataset_name = 'kdd99'
        target = 39
    else:
        print('TODO: HAVE TO DO THIS DATASET!')

    out = Path(os.getcwd()+'/data/'+dataset_name+'.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print("File already exists.")
    else:
        if url is not None:
            print(f"Downloading {dataset} file...")
            wget.download(url, out.as_posix())
        else:
            raise 'Download the dataset from the link mentioned in github first'

    if dataset=='1995_income':
        train = pd.read_csv(out,header=None)
    elif dataset=='blood':
        train = pd.read_csv(out,header=None)
    elif dataset=='autos':
        train = pd.read_csv(out,header=None)
    elif dataset=='libras':
        train = pd.read_csv(out,header=None)
    elif dataset=='bank_marketing':
        train = pd.read_csv(out,sep=';',header=None, skiprows=1)
    elif dataset == 'qsar_bio':
        train = pd.read_csv(out,sep=';',header=None)
    elif dataset == 'online_shoppers':
        train = pd.read_csv(out,sep=',',header=None, skiprows=1)
    elif dataset == 'blastchar':
        train = pd.read_csv(out,sep=',',header=None, skiprows=1)
    elif dataset == 'abalon':
        train = pd.read_csv(out,sep=',',header=None, skiprows=0)
    elif dataset == 'baseball':
        train = pd.read_csv(out,sep=',',header=None, skiprows=0)
    elif dataset == 'htru2':
        train = pd.read_csv(out,sep=',',header=None, skiprows=0)
    elif dataset == 'cloud':
        train = pd.read_csv(out,sep=',',header=None, skiprows=0)
    elif dataset == 'kidney':
        train = pd.read_csv(out,sep=',',header=None, skiprows=0)
    elif dataset == 'acute-nephritis':
        train = pd.read_csv(out,sep=',',header=None, skiprows=0)
    elif dataset=='shrutime':
        train = pd.read_csv(out,sep=',',header=None, skiprows=1)
        train = train.iloc[:,2:]
        train.columns = range(train.shape[1])
    elif dataset == 'spambase':
        train = pd.read_csv(out,sep=',',header=None, skiprows=0)
        train = train.iloc[:,1:]
    elif dataset == 'loan_data':
        train = pd.read_csv(out,sep=',', skiprows=0)
        train.columns = range(train.shape[1])
#         cat_cols = [0,1,6,10,11,12]
        cat_cols = [1]
        return train, target, cat_cols
    elif dataset == 'arcene':
        train = pd.read_csv(out,sep=',',skiprows=0)
        tg_list = train['target'].tolist()
        # train = train.loc[:, train.std() > 7]
        train = train.loc[:, train.std() > 150]
        train['target'] = tg_list
    elif dataset == 'creditcard':
        train = pd.read_csv(out,header=None,skiprows=1)
        train = train.iloc[:,1:]
    elif dataset == 'arrhythmia':
        train = pd.read_csv(out,header=None,skiprows=1)
    elif dataset == 'forest':
        train = pd.read_csv(out,header=None,skiprows=1)
    elif dataset == 'kdd99':
        train = pd.read_csv(out,header=None,skiprows=1)
    return train, target


def perform_nested_cv(dataset):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default=dataset, type=str, choices=['1995_income','bank_marketing','qsar_bio','online_shoppers','blastchar','htru2','shrutime','spambase','philippine','mnist','loan_data','arcene','volkert','creditcard','arrhythmia','forest','kdd99'])
    parser.add_argument('--cont_embeddings', default='MLP', type=str,choices = ['MLP','Noemb','pos_singleMLP'])
    parser.add_argument('--embedding_size', default=32, type=int)
    parser.add_argument('--transformer_depth', default=6, type=int)
    parser.add_argument('--attention_heads', default=8, type=int)
    parser.add_argument('--attention_dropout', default=0.1, type=float)
    parser.add_argument('--ff_dropout', default=0.1, type=float)
    parser.add_argument('--attentiontype', default='colrow', type=str,choices = ['col','colrow','row','justmlp','attn','attnmlp'])
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batchsize', default=256, type=int)
    parser.add_argument('--savemodelroot', default='./bestmodels', type=str)
    parser.add_argument('--run_name', default='testrun', type=str)
    parser.add_argument('--set_seed', default= 1 , type=int)
    parser.add_argument('--active_log', action = 'store_true')

    parser.add_argument('--pretrain', action = 'store_true')
    parser.add_argument('--pretrain_epochs', default=50, type=int)
    parser.add_argument('--pt_tasks', default=['contrastive','denoising'], type=str,nargs='*',choices = ['contrastive','contrastive_sim','denoising'])
    parser.add_argument('--pt_aug', default=[], type=str,nargs='*',choices = ['mixup','cutmix','gauss_noise'])
    parser.add_argument('--pt_aug_lam', default=0.1, type=float)
    parser.add_argument('--mixup_lam', default=0.3, type=float)

    parser.add_argument('--train_mask_prob', default=0, type=float)
    parser.add_argument('--mask_prob', default=0, type=float)

    parser.add_argument('--ssl_avail_y', default= 0, type=int)
    parser.add_argument('--pt_projhead_style', default='diff', type=str,choices = ['diff','same','nohead'])
    parser.add_argument('--nce_temp', default=0.7, type=float)

    parser.add_argument('--lam0', default=0.5, type=float)
    parser.add_argument('--lam1', default=10, type=float)
    parser.add_argument('--lam2', default=1, type=float)
    parser.add_argument('--lam3', default=10, type=float)
    parser.add_argument('--final_mlp_style', default='sep', type=str,choices = ['common','sep'])


    opt = parser.parse_args()
    torch.manual_seed(opt.set_seed)

    device = torch.device("cpu")
    if opt.attentiontype in ['colrow','row']:
        opt.ff_dropout = 0.8
        opt.transformer_depth = 1
        if opt.dataset in ['arrhythmia','philippine','creditcard', 'abalon', 'autos']:
            opt.embedding_size = 8
            opt.attention_heads = 4

    if opt.dataset in ['arrhythmia']:
        opt.embedding_size = 8
        if opt.attentiontype in ['col']:
            opt.transformer_depth = 1

    if opt.dataset in ['philippine', 'baseball', 'libras']:
        opt.batchsize = 128
        if opt.attentiontype in ['col']:
            opt.embedding_size = 8

    if opt.dataset in ['arcene']:
        opt.embedding_size = 4
        if opt.attentiontype in ['colrow','col']:
            opt.attention_heads = 1
            opt.transformer_depth = 2

    if opt.dataset in ['mnist']:
        opt.batchsize = 32
        opt.attention_heads = 4
        if opt.attentiontype in ['col']:
            opt.embedding_size = 12
        else:
            opt.embedding_size = 8


    print(f"Device is {device}.")

    modelsave_path = os.path.join(os.getcwd(),opt.savemodelroot,opt.dataset,opt.run_name)
    os.makedirs(modelsave_path, exist_ok=True)

    if opt.active_log:
        import wandb
        if opt.ssl_avail_y > 0 and opt.pretrain:
            wandb.init(project="saint_ssl", group=opt.run_name, name = opt.run_name + '_' + str(opt.attentiontype)+ '_' +str(opt.dataset))
        else:
            wandb.init(project="saint_all", group=opt.run_name, name = opt.run_name + '_' + str(opt.attentiontype)+ '_' +str(opt.dataset))
        wandb.config.update(opt)



    # mask parameters are used to similate missing data scenrio. Set to default 0s otherwise. (pt_mask_params is for pretraining)
    mask_params = {
        "mask_prob":opt.train_mask_prob,
        "avail_train_y": 0,
        "test_mask":opt.train_mask_prob
    }

    pt_mask_params = {
            "mask_prob":opt.mask_prob,
            "avail_train_y": 0,
            "test_mask": 0
        }

    print('Downloading and processing the dataset, it might take some time.')
    outer_kf = StratifiedKFold(10, shuffle=True)
    train, target, categorical_columns = None, None, None
    if opt.dataset in ['1995_income','bank_marketing', 'kidney', 'blood', 'libras', 'autos', 'baseball', 'abalon', 'cloud', 'qsar_bio','online_shoppers','blastchar','htru2','shrutime','spambase','arcene','creditcard','arrhythmia','forest','kdd99', 'acute-nephritis']:
        train, target = data_download(opt.dataset)
    if opt.dataset in ['loan_data']:
        train, target, categorical_columns = data_download(opt.dataset)

    split = 1
    best_params, acc_scores, roc_scores, tprs, fprs, precisions, auprs, best, train_times, infer_times = [], [], [], [], [], [], [], [], [], []
    Y = train[train.columns[-1]]
    for train_split, test_split in outer_kf.split(train, Y):
        print('Cross Validation Number:', split)
        split += 1
        train_inside, test_inside = train.iloc[train_split], train.iloc[test_split]
        if opt.dataset not in ['mnist']:
            cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std = data_prep(opt.dataset, opt.set_seed, train_inside, test_inside, target, categorical_columns, mask_params)
            continuous_mean_std = np.array([train_mean,train_std]).astype(np.float32)
            if opt.dataset == 'volkert':
                y_dim = 10
            elif opt.dataset == 'autos':
                y_dim = 5
            elif opt.dataset == 'libras':
                y_dim = 15
            elif opt.dataset in ['abalon', 'baseball']:
                y_dim = 3
            else:
                y_dim = 2
        # else:
        #     from data import vision_data_prep
        #     cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, _, _ = vision_data_prep(opt.dataset, opt.set_seed, mask_params)
        #     continuous_mean_std = None
        #     y_dim = 10

        train_bsize = opt.batchsize
        if opt.ssl_avail_y>0:
            train_pts_touse = np.random.choice(X_train['data'].shape[0], opt.ssl_avail_y)
            X_train['data'] = X_train['data'][train_pts_touse,:]
            y_train['data'] = y_train['data'][train_pts_touse]

            X_train['mask'] = X_train['mask'][train_pts_touse,:]
            y_train['mask'] = y_train['mask'][train_pts_touse]
            train_bsize = min(opt.ssl_avail_y//4,opt.batchsize)


        # valid_ds = DataSetCatCon(X_valid, y_valid, cat_idxs,continuous_mean_std, is_pretraining=True)
        # validloader = DataLoader(valid_ds, batch_size=opt.batchsize, shuffle=False,num_workers=4)

        test_ds = DataSetCatCon(X_test, y_test, cat_idxs,continuous_mean_std, is_pretraining=True)
        testloader = DataLoader(test_ds, batch_size=opt.batchsize, shuffle=False,num_workers=4)

        # Creating a different dataloader for the pretraining.
        if opt.pretrain:
            if opt.dataset not in ['mnist']:
                _, cat_idxs, _, X_train_pt, y_train_pt, _, _, _, _, train_mean, train_std = data_prep(opt.dataset, opt.set_seed, pt_mask_params)
                ctd = np.array([train_mean,train_std]).astype(np.float32)
            else:
                _, cat_idxs, _, X_train_pt, y_train_pt, _, _, _, _, _, _ = vision_data_prep(opt.dataset, opt.set_seed, pt_mask_params)
                ctd = None
            pt_train_ds = DataSetCatCon(X_train_pt, y_train_pt, cat_idxs,ctd, is_pretraining=True)
            pt_trainloader = DataLoader(pt_train_ds, batch_size=opt.batchsize, shuffle=True,num_workers=4)

        if opt.dataset in ['abalon', 'baseball']:
            cat_dims = np.append(np.array(cat_dims), np.array([3])).astype(
                int)  # unique values in cat column, with 2 appended in the end as the number of unique values of y. This is the case of binary classification
            model = SAINT(
                categories=tuple(cat_dims),
                num_continuous=len(con_idxs),
                dim=opt.embedding_size,
                dim_out=1,
                depth=opt.transformer_depth,
                heads=opt.attention_heads,
                attn_dropout=opt.attention_dropout,
                ff_dropout=opt.ff_dropout,
                mlp_hidden_mults=(4, 2),
                continuous_mean_std=continuous_mean_std,
                cont_embeddings=opt.cont_embeddings,
                attentiontype=opt.attentiontype,
                final_mlp_style=opt.final_mlp_style,
                y_dim=y_dim
            )
            vision_dset = False

        if opt.dataset == 'libras':
            cat_dims = np.append(np.array(cat_dims), np.array([15])).astype(
                int)  # unique values in cat column, with 2 appended in the end as the number of unique values of y. This is the case of binary classification
            model = SAINT(
                categories=tuple(cat_dims),
                num_continuous=len(con_idxs),
                dim=opt.embedding_size,
                dim_out=1,
                depth=opt.transformer_depth,
                heads=opt.attention_heads,
                attn_dropout=opt.attention_dropout,
                ff_dropout=opt.ff_dropout,
                mlp_hidden_mults=(4, 2),
                continuous_mean_std=continuous_mean_std,
                cont_embeddings=opt.cont_embeddings,
                attentiontype=opt.attentiontype,
                final_mlp_style=opt.final_mlp_style,
                y_dim=y_dim
            )
            vision_dset = False

        if opt.dataset == 'autos':
            cat_dims = np.append(np.array(cat_dims), np.array([5])).astype(
                int)  # unique values in cat column, with 2 appended in the end as the number of unique values of y. This is the case of binary classification
            model = SAINT(
                categories=tuple(cat_dims),
                num_continuous=len(con_idxs),
                dim=opt.embedding_size,
                dim_out=1,
                depth=opt.transformer_depth,
                heads=opt.attention_heads,
                attn_dropout=opt.attention_dropout,
                ff_dropout=opt.ff_dropout,
                mlp_hidden_mults=(4, 2),
                continuous_mean_std=continuous_mean_std,
                cont_embeddings=opt.cont_embeddings,
                attentiontype=opt.attentiontype,
                final_mlp_style=opt.final_mlp_style,
                y_dim=y_dim
            )
            vision_dset = False
        if opt.dataset not in ['mnist','volkert', 'abalon', 'baseball', 'autos', 'libras']:
            cat_dims = np.append(np.array(cat_dims),np.array([2])).astype(int) # unique values in cat column, with 2 appended in the end as the number of unique values of y. This is the case of binary classification
            model = SAINT(
            categories = tuple(cat_dims),
            num_continuous = len(con_idxs),
            dim = opt.embedding_size,
            dim_out = 1,
            depth = opt.transformer_depth,
            heads = opt.attention_heads,
            attn_dropout = opt.attention_dropout,
            ff_dropout = opt.ff_dropout,
            mlp_hidden_mults = (4, 2),
            continuous_mean_std = continuous_mean_std,
            cont_embeddings = opt.cont_embeddings,
            attentiontype = opt.attentiontype,
            final_mlp_style = opt.final_mlp_style,
            y_dim = y_dim
            )
            vision_dset = False
        elif opt.dataset == 'volkert':
            cat_dims = np.append(np.array(cat_dims),np.array([10])).astype(int)
            model = SAINT(
            categories = tuple(cat_dims),
            num_continuous = len(con_idxs),
            dim = opt.embedding_size,
            dim_out = 1,
            depth = opt.transformer_depth,
            heads = opt.attention_heads,
            attn_dropout = opt.attention_dropout,
            ff_dropout = opt.ff_dropout,
            mlp_hidden_mults = (4, 2),       # relative multiples of each hidden dimension of the last mlp to logits
            continuous_mean_std = continuous_mean_std,
            cont_embeddings = opt.cont_embeddings,
            attentiontype = opt.attentiontype,
            final_mlp_style = opt.final_mlp_style,
            y_dim = y_dim
            )
            vision_dset = False

        elif opt.dataset == 'mnist':
            cat_dims = np.append(np.array(cat_dims),np.array([10])).astype(int)
            model = SAINT_vision(
            categories = tuple(cat_dims),
            num_continuous = len(con_idxs),
            dim = opt.embedding_size,
            dim_out = 1,
            depth = opt.transformer_depth,
            heads = opt.attention_heads,
            attn_dropout = opt.attention_dropout,
            ff_dropout = opt.ff_dropout,
            mlp_hidden_mults = (4, 2),       # relative multiples of each hidden dimension of the last mlp to logits
            continuous_mean_std = continuous_mean_std,
            cont_embeddings = opt.cont_embeddings,
            attentiontype = opt.attentiontype,
            final_mlp_style = opt.final_mlp_style,
            y_dim = y_dim
            )
            vision_dset = True
        else:
            print('This dataset is not valid')


        criterion = nn.CrossEntropyLoss().to(device)
        model.to(device)

        if opt.pretrain:
            optimizer = optim.AdamW(model.parameters(),lr=opt.lr)
            pt_aug_dict = {
                'noise_type' : opt.pt_aug,
                'lambda' : opt.pt_aug_lam
            }
            criterion1 = nn.CrossEntropyLoss()
            criterion2 = nn.MSELoss()
            print("Pretraining begins!")
            for epoch in range(opt.pretrain_epochs):
                model.train()
                running_loss = 0.0
                for i, data in enumerate(pt_trainloader, 0):
                    optimizer.zero_grad()
                    x_categ, x_cont, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device)

                    # embed_data_mask function is used to embed both categorical and continuous data.
                    if 'cutmix' in opt.pt_aug:
                        from augmentations import add_noise
                        x_categ_corr, x_cont_corr = add_noise(x_categ,x_cont, noise_params = pt_aug_dict)
                        _ , x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ_corr, x_cont_corr, cat_mask, con_mask,model,vision_dset)
                    else:
                        _ , x_categ_enc_2, x_cont_enc_2 = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)
                    _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)

                    if 'mixup' in opt.pt_aug:
                        from augmentations import mixup_data
                        x_categ_enc_2, x_cont_enc_2 = mixup_data(x_categ_enc_2, x_cont_enc_2 , lam=opt.mixup_lam)

                    loss = 0
                    if 'contrastive' in opt.pt_tasks:
                        aug_features_1  = model.transformer(x_categ_enc, x_cont_enc)
                        aug_features_2 = model.transformer(x_categ_enc_2, x_cont_enc_2)
                        aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1,2)
                        aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1,2)
                        if opt.pt_projhead_style == 'diff':
                            aug_features_1 = model.pt_mlp(aug_features_1)
                            aug_features_2 = model.pt_mlp2(aug_features_2)
                        elif opt.pt_projhead_style == 'same':
                            aug_features_1 = model.pt_mlp(aug_features_1)
                            aug_features_2 = model.pt_mlp(aug_features_2)
                        else:
                            print('Not using projection head')
                        logits_per_aug1 = aug_features_1 @ aug_features_2.t()/opt.nce_temp
                        logits_per_aug2 =  aug_features_2 @ aug_features_1.t()/opt.nce_temp
                        targets = torch.arange(logits_per_aug1.size(0)).to(logits_per_aug1.device)
                        loss_1 = criterion(logits_per_aug1, targets)
                        loss_2 = criterion(logits_per_aug2, targets)
                        loss   = opt.lam0*(loss_1 + loss_2)/2
                    elif 'contrastive_sim' in opt.pt_tasks:
                        aug_features_1  = model.transformer(x_categ_enc, x_cont_enc)
                        aug_features_2 = model.transformer(x_categ_enc_2, x_cont_enc_2)
                        aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1,2)
                        aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1,2)
                        aug_features_1 = model.pt_mlp(aug_features_1)
                        aug_features_2 = model.pt_mlp2(aug_features_2)
                        c1 = aug_features_1 @ aug_features_2.t()
                        loss+= opt.lam1*torch.diagonal(-1*c1).add_(1).pow_(2).sum()
                    if 'denoising' in opt.pt_tasks:
                        cat_outs, con_outs = model(x_categ_enc_2, x_cont_enc_2)
                        con_outs =  torch.cat(con_outs,dim=1)
                        l2 = criterion2(con_outs, x_cont)
                        l1 = 0
                        for j in range(len(cat_dims)-1):
                            l1+= criterion1(cat_outs[j],x_categ[:,j])
                        loss += opt.lam2*l1 + opt.lam3*l2
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                print(f'Epoch: {epoch}, Running Loss: {running_loss}')
                if opt.active_log:
                    wandb.log({'pt_epoch': epoch ,'pretrain_epoch_loss': running_loss
                    })

        optimizer = optim.AdamW(model.parameters(),lr=opt.lr)

        best_valid_auroc = 0
        best_valid_accuracy = 0
        best_test_auroc = 0
        best_test_fpr = 0
        best_test_tpr = 0
        best_test_prec = 0
        best_test_train = 0
        best_test_infer = 0
        best_test_aupr = 0
        best_test_accuracy = 0
        best_lr = 0
        best_ep = 0

        print('Training begins now.')
        inner_kf = KFold(3)
        lrs = [0.001, 0.0001]
        eps = [0.9, 0.99]
        for lr in lrs:
            optimizer.defaults['lr'] = lr
            for ep in eps:
                optimizer.defaults['eps'] = ep
                print("Starting HyperParameter tuning using lr", lr, "and epsilon", ep)
                cv_inside = 1
                for inner_train, rest_train in inner_kf.split(X_train['data']):
                    print("Starting inside cross validation number:", cv_inside)
                    cv_inside += 1
                    inner_training_x, inner_testing_x = {'data': X_train['data'][inner_train], 'mask': X_train['mask'][inner_train]}, {'data': X_train['data'][rest_train], 'mask': X_train['mask'][rest_train]}
                    inner_training_y, inner_testing_y = {'data': y_train['data'][inner_train], 'mask': y_train['mask'][inner_train]}, {'data': y_train['data'][rest_train], 'mask': y_train['mask'][rest_train]}
                    train_ds = DataSetCatCon(inner_training_x, inner_training_y, cat_idxs, continuous_mean_std, is_pretraining=True)
                    trainloader = DataLoader(train_ds, batch_size=train_bsize, shuffle=True, num_workers=4)
                    hp_test = DataSetCatCon(inner_testing_x, inner_testing_y, cat_idxs, continuous_mean_std, is_pretraining=True)
                    hp_testloader = DataLoader(hp_test, batch_size=train_bsize, shuffle=True, num_workers=4)
                    start_train = time.time()
                    for epoch in range(10):
                        model.train()
                        running_loss = 0.0

                        for i, data in enumerate(trainloader, 0):
                            optimizer.zero_grad()
                            # x_categ is the the categorical data, with y appended as last feature. x_cont has continuous data. cat_mask is an array of ones same shape as x_categ except for last column(corresponding to y's) set to 0s. con_mask is an array of ones same shape as x_cont.
                            x_categ, x_cont, cat_mask, con_mask = data[0].to(device), data[1].to(device),data[2].to(device),data[3].to(device)
                            # We are converting the data to embeddings in the next step
                            _ , x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask,model,vision_dset)
                            reps = model.transformer(x_categ_enc, x_cont_enc)
                            # select only the representations corresponding to y and apply mlp on it in the next step to get the predictions.
                            y_reps = reps[:,len(cat_dims)-1,:]
                            y_outs = model.mlpfory(y_reps)
                            loss = criterion(y_outs,x_categ[:,len(cat_dims)-1])
                            loss.backward()
                            optimizer.step()
                            running_loss += loss.item()

                        end_train = time.time()
                        if opt.active_log:
                            wandb.log({'epoch': epoch ,'train_epoch_loss': running_loss,
                            'loss': loss.item()
                            })

                        if epoch == 9:
                                model.eval()
                                with torch.no_grad():
                                    if opt.dataset in ['mnist','volkert']:
                                        from utils import multiclass_acc_justy
                                        # accuracy, auroc = multiclass_acc_justy(model, validloader, device)
                                        test_accuracy, test_auroc = multiclass_acc_justy(model, testloader, device)
                                        if accuracy > best_valid_accuracy:
                                            best_valid_accuracy = accuracy
                                            best_test_auroc = test_auroc
                                            best_test_accuracy = test_accuracy
                                            torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
                                    else:
                                        # accuracy, auroc = imputations_acc_justy(model, validloader, device)
                                        infer_time = time.time()
                                        test_accuracy, test_auroc, prec, aupr, fpr, tpr = imputations_acc_justy(model, testloader, device)
                                        end_infer_time = time.time()
                                        print("Inference time for ", len(X_test['data']), " instances is: " ,end_infer_time-infer_time)

                                    # print('[EPOCH %d] VALID ACCURACY: %.3f, VALID AUROC: %.3f' %
                                    #     (epoch + 1, accuracy,auroc ))
                                    print('[EPOCH %d] TEST ACCURACY: %.3f, TEST AUROC: %.3f' %
                                        (epoch + 1, test_accuracy,test_auroc ))
                                    if opt.active_log:
                                        # wandb.log({'valid_accuracy': accuracy ,'valid_auroc': auroc })
                                        wandb.log({'test_accuracy': test_accuracy ,'test_auroc': test_auroc })
                                    if test_auroc > best_test_auroc:
                                        # best_valid_auroc = auroc
                                        best_test_auroc = test_auroc
                                        best_test_accuracy = test_accuracy
                                        best_test_tpr = tpr
                                        best_test_fpr = fpr
                                        best_test_prec = prec
                                        best_test_aupr = aupr
                                        best_test_infer = end_infer_time - infer_time
                                        best_test_train = end_train - start_train
                                        best_lr = lr
                                        best_ep = ep
                                        torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
                                model.train()

                        else:
                                model.eval()
                                with torch.no_grad():
                                    if opt.dataset in ['mnist','volkert']:
                                        from utils import multiclass_acc_justy
                                        # accuracy, auroc = multiclass_acc_justy(model, validloader, device)
                                        test_accuracy, test_auroc = multiclass_acc_justy(model, hp_testloader, device)
                                        if accuracy > best_valid_accuracy:
                                            best_valid_accuracy = accuracy
                                            best_test_auroc = test_auroc
                                            best_test_accuracy = test_accuracy
                                            torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
                                    else:
                                        # accuracy, auroc = imputations_acc_justy(model, validloader, device)
                                        test_accuracy, test_auroc, _, _, _, _ = imputations_acc_justy(model, hp_testloader, device)


                                    # print('[EPOCH %d] VALID ACCURACY: %.3f, VALID AUROC: %.3f' %
                                    #     (epoch + 1, accuracy,auroc ))
                                    print('[EPOCH %d] HP ACCURACY: %.3f, HP AUROC: %.3f',
                                        (epoch + 1, test_accuracy,test_auroc ))
                                    if opt.active_log:
                                        # wandb.log({'valid_accuracy': accuracy ,'valid_auroc': auroc })
                                        wandb.log({'test_accuracy': test_accuracy ,'test_auroc': test_auroc })
                                    if test_auroc > best_test_auroc:
                                        # best_valid_auroc = auroc
                                        best_test_auroc = test_auroc
                                        best_test_accuracy = test_accuracy
                                        torch.save(model.state_dict(),'%s/bestmodel.pth' % (modelsave_path))
                                model.train()




                        total_parameters = count_parameters(model)
                        print('TOTAL NUMBER OF PARAMS: %d' %(total_parameters))
                        if opt.dataset not in ['mnist','volkert']:
                            print('AUROC on best model:  %.3f' %(best_test_auroc))
                        else:
                            print('Accuracy on best model:  %.3f' %(best_test_accuracy))
                        if opt.active_log:
                            wandb.log({'total_parameters': total_parameters, 'test_auroc_bestep':best_test_auroc ,
                            'test_accuracy_bestep':best_test_accuracy,'cat_dims':len(cat_idxs) , 'con_dims':len(con_idxs) })

        roc_scores.append(best_test_auroc)
        auprs.append(best_test_aupr)
        infer_times.append(best_test_infer)
        train_times.append(best_test_train)
        precisions.append(best_test_prec)
        tprs.append(best_test_tpr)
        fprs.append(best_test_fpr)
        acc_scores.append(best_test_accuracy)
        best_params.append({'lr': best_lr, 'ep': best_ep})


    infer_times = [time * 1000 / len(X_test) for time in infer_times]
    data = {
        "Dataset": [dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset, dataset],
        "Algorithm Name": ['SAINT', 'SAINT', 'SAINT', 'SAINT', 'SAINT',
                           'SAINT', 'SAINT', 'SAINT', 'SAINT', 'SAINT'],
        "CV": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "hyperparameter": best_params,
        "Accuracy": acc_scores,
        "TPR": tprs,
        "FPR": fprs,
        "Precision": precisions,
        "AUC": roc_scores,
        "PR-curves": auprs,
        "Training Time": train_times,
        "Inference Time": infer_times
    }
    pd.DataFrame(data).to_excel('1.xlsx')


if __name__ == '__main__':
    perform_nested_cv('1995_income')