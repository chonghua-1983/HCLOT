# -*- coding: utf-8 -*-
import numpy as np
import random
import torch
import scipy.io as sio
import scipy.sparse as sp


def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat


def feature_concat(*F_list, normal_col=False):
    """
    Concatenate multiple modality feature. If the dimension of a feature matrix is more than two,
    the function will reduce it into two dimension(using the last dimension as the feature dimension,
    the other dimension will be fused as the object dimension)
    :param F_list: Feature matrix list
    :param normal_col: normalize each column of the feature
    :return: Fused feature matrix
    """
    features = None
    for f in F_list:
        if f is not None and f != []:
            # deal with the dimension that more than two
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            # normal each column
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            # facing the first feature matrix appended to fused feature matrix
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features


def hyperedge_concat(*H_list):
    """
    Concatenate hyperedge group in H_list
    :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix
    :return: Fused hypergraph incidence matrix
    """
    H = None
    for h in H_list:
        if h is not None and h != []:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)
    eps = 1e-10

    invDE = np.mat(np.diag(np.power(DE+eps, -1)))
    invDV = np.mat(np.diag(np.power(DV+eps, -1)))
    DV2 = np.mat(np.diag(np.power(DV+eps, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        # G = invDV * H * W * invDE * HT
        return G


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H


def construct_H_with_KNN(X, K_neigs=[10], split_diff_scale=False, is_probH=True, m_prob=1):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    dis_mat = Eu_dis(X)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)
    return H


def generate_mask(labels,N):  
    num = 0
    A = sp.csr_matrix((labels[:,2],(labels[:,0]-1, labels[:,1]-1)),shape=(1373,173)).toarray()
    mask = np.zeros(A.shape)
    label_neg=np.zeros((1*N,2)) 
    while(num<1*N):
        a = random.randint(0,1372)
        b = random.randint(0,172)
        if A[a,b] != 1 and mask[a,b] != 1:
            mask[a,b] = 1
            label_neg[num,0]=a
            label_neg[num,1]=b
            num += 1
    mask = np.reshape(mask,[-1,1])  
    return mask,label_neg

def test_negative_sample(labels,N,negative_mask):  
    num = 0
    A = sp.csr_matrix((labels[:,2],(labels[:,0]-1, labels[:,1]-1)),shape=(1373,173)).toarray()  
    mask = np.zeros(A.shape)
    test_neg=np.zeros((1*N,2))  
    while(num<1*N):
        a = random.randint(0,1372)
        b = random.randint(0,172)
        if A[a,b] != 1 and mask[a,b] != 1:
            mask[a,b] = 1
            test_neg[num,0]=a
            test_neg[num,1]=b
            num += 1
    return test_neg


def build_hyperedge_matrix(labels):
    labels = labels.astype(int)
    n, m = max(labels[:,0]), max(labels[:,1])
    tmp = np.zeros((n+m, len(labels)))
    for i in range(len(labels)):
        
            ind = labels[i,:]
            ind = ind + [0, n]
            tmp[ind-1, i] = 1
 
    return tmp

import torch
def normalize_rows_zscore(x: torch.Tensor) -> torch.Tensor:
    mean = torch.mean(x, dim=1, keepdim=True)
    std = torch.std(x, dim=1, keepdim=True)
    #
    std = torch.clamp(std, min=1e-12)
    #
    return (x - mean) / std


def normalize_cols_zscore(x: torch.Tensor) -> torch.Tensor:
    mean = torch.mean(x, dim=0, keepdim=True)
    std = torch.std(x, dim=0, keepdim=True)
    #
    std = torch.clamp(std, min=1e-12)
    #
    return (x - mean) / std


def process_sot(x: torch.Tensor, k) -> torch.Tensor:
    """
    transform x to a binary matrix according to knn
    Returns
    """
    x = np.array(x)
    n, m = x.shape
    ind_by_row = np.argsort(-x, axis=1)[:, 0:k]
    x_trans_row = np.zeros([n, m])
    
    for i in range(n):
        x_trans_row[i, ind_by_row[i,:]] = 1
    
    x_trans_col = np.zeros([n, m])
    ind_by_col = np.argsort(-x, axis=0)[0:k, :]
    for j in range(m):
        x_trans_col[ind_by_col[:, j], j] = 1
    
    x_trans = (x_trans_row + x_trans_col)/2
    
    return x_trans


def process_sot_byrow(x, k):
    """
    transform x to a binary matrix according to knn
    Returns
    """
    x = np.array(x)
    n, m = x.shape
    ind_by_row = np.argsort(-x, axis=1)[:, 0:k]
    x_trans_row = np.zeros([n, m])
    
    for i in range(n):
        x_trans_row[i, ind_by_row[i,:]] = 1
    
    # x_trans_col = np.zeros([n, m])
    # ind_by_col = np.argsort(-x, axis=0)[0:k, :]
    # for j in range(m):
    #     x_trans_col[ind_by_col[:, j], j] = 1
    
    x_trans = x_trans_row
    
    return x_trans


def top_sim(A, k=10):
    """
    A : torch tensor
    k : neghbor number, The default is 10.

    Returns
    A_new : fetch topk largest values in each row of A
    """
    
    result = torch.zeros_like(A)

    values, indices = torch.topk(A, k, dim=1)

    rows = torch.arange(A.size(0)).unsqueeze(1).expand(-1, k)
    result[rows, indices] = values

    # A_new = (result+result.T)/2
    A_new = result

    return A_new


def get_Gauss_Similarity(interaction_matrix):
    X = np.mat(interaction_matrix)
    delta = 1 / np.mean(np.power(X,2), 0).sum()
    alpha = np.power(X, 2).sum(axis=1)
    result = np.exp(np.multiply(-delta, alpha + alpha.T - 2 * X * X.T))
    # similarity_matrix[np.isnan(similarity_matrix)] = 0
    # result = result - np.diag(np.diag(result))
    return result

def get_Gauss_Similarity_torch(interaction_matrix,name):
    if name == 'row':
        X = interaction_matrix
    else:
        X = interaction_matrix.t()
    delta = 1 / torch.sum(torch.mean(torch.pow(X, 2), 0))
    alpha = torch.sum(torch.pow(X, 2), 1).unsqueeze(dim=1)

    result = torch.exp(torch.multiply(-delta, alpha + alpha.t() - 2 * X.mm(X.t())))
    result = torch.where(torch.isnan(result), torch.tensor(0.0).cuda(), result)
    result = torch.where(torch.isinf(result), torch.tensor(0.0).cuda(), result)
    # result[t.isnan(result)] = 0
    # result[t.isinf(re

def  integrated_similarity(GD, DS):
    #GD高斯相似性，DS另外一种相似性
    m = GD.shape[0]
    SD = np.zeros_like(GD)

    for i in range(m):
        for j in range(m):
            if DS[i,j] != 0:
                SD[i,j] = (GD[i,j]+DS[i,j])/2
            else:
                SD[i, j] = GD[i,j]
    return SD

