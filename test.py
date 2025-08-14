# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os.path
import torch

from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import time

from args import *
from model import *
from utils.dataset import *
from utils.distance import *
from utils.tools import *
from utils.metrics import *
from utils.evluate import *

if __name__ == '__main__':
    seed = 2022
    set_seed(seed)
    args = make_args()

    # load data
    print("Loading data...", end=" ")
    dataset = 'aBiofilm'  # MDAD  aBiofilm 
    Fd, Fm, Sd, Sm, dmi_adj, labels = load_data_mdad(dataset)
    dmi_adj = dmi_adj.astype(np.float32)    
    x_combine = np.vstack((np.hstack((Fd, np.zeros(shape=(Fd.shape[0], Fm.shape[1]), dtype=float))),
                           np.hstack((np.zeros(shape=(Fm.shape[0], Fd.shape[1]), dtype=int), Fm))))
    
    
    from sklearn.decomposition import PCA
    pca1 = PCA(n_components=args.pca_dim+500)  # 500
    feas = pca1.fit_transform(np.array(x_combine))
    feas = torch.tensor(feas,dtype=torch.float32)
    feas = normalize_rows_zscore(feas)
    
    data = getData(dmi_adj)
    k_folds = 5
    index_matrix = np.array(np.where(data.adj_matrix == 1))
    positive_num = index_matrix.shape[1]
    sample_num_per_fold = int(positive_num / k_folds)
    np.random.seed(2022)
    np.random.shuffle(index_matrix.T)

    metrics_matrix = np.zeros((1, 7))
    result_100 = []
    
    time1 = time.time()
    for k in range(k_folds):
        train_matrix = np.array(data.adj_matrix, copy=True)
        if k != k_folds - 1:
            test_index = tuple(index_matrix[:, k * sample_num_per_fold: (k + 1) * sample_num_per_fold])
        else:
            test_index = tuple(index_matrix[:, k * sample_num_per_fold:])

        train_matrix[test_index] = 0
        
        P = np.vstack((np.hstack((Sd, train_matrix)),
                      np.hstack((train_matrix.transpose(), Sm))))
        
        interaction = normalize_adj(P) # normalized the interaction matrix
        
        # device setting
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        ''' using PCA to reduce dimension'''   
        pca2 = PCA(n_components=args.pca_dim)
        concat_mic = pca2.fit_transform(np.array(interaction))
        concat_mic = torch.tensor(concat_mic,dtype=torch.float32)
        concat_mic = normalize_rows_zscore(concat_mic)
        
        H = construct_H_with_KNN(concat_mic, K_neigs=[10,20,50], split_diff_scale=True, is_probH=True, m_prob=1) 
        
        hyper_adj = generate_G_from_H(H, variable_weight=False)
        hyper_adj = (hyper_adj[0]+hyper_adj[1]+hyper_adj[2])/3
        
        n1, n2 = dmi_adj.shape[0], dmi_adj.shape[1]
        args.gw_weight = args.alpha / (1 - args.alpha) * min(n1, n2) ** 0.5
        start = time.time()
        
        hgcot = HCOT(feas, hyper_adj, Sd, Sm, train_matrix, args, device)
        pre_score, s_ot, _, _, loss = hgcot.train_with_dec()
        out, _, _, _ = hgcot.process()
        out = torch.tensor(out)        
        out1, out2 = out[0:n1, :], out[n1:, :] 
        
        pre_score = cos_sim(out1, out2)
        pre_score = torch.where(pre_score>torch.median(pre_score), torch.tensor(dmi_adj), pre_score)
        
        pre_score = pre_score.detach().numpy()        
        predict_matrix = np.array(pre_score)
        
        # Randomized Generation of Test Negatives and Model Evaluation
        prop = [1, 10, 50]
        for i in range(1,20000,1000):
            jieguo = cv_tensor_model_evaluate(data, predict_matrix, test_index, i, prop[0])
            metrics_matrix = metrics_matrix + jieguo
            result_100.append(jieguo)
        print(np.array(result_100).mean(axis=0))
    
    time2 = time.time()
    print("computational time；", time2-time1)
    
    # Get result
    result = pd.DataFrame(np.around(metrics_matrix / (20*k_folds), decimals=4)[:, 0:3], columns=['AUPR', 'AUC', 'F1'])
    result_100 = pd.DataFrame(np.array(result_100)[:, 0:3], columns=['AUPR', 'AUC', 'F1'])
    print(result)
    
    
    
    
    
 
    
