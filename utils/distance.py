import numpy as np
import networkx as nx
import os
from tqdm import tqdm


def get_rwr_matrix(G1, G2, anchor_links, dataset, ratio, dtype=np.float32):
    """
    Get distance matrix of the network
    :param G1: input graph 1
    :param G2: input graph 2
    :param anchor_links: anchor links
    :param dataset: dataset name
    :param ratio: training ratio
    :param dtype: data type
    :return: distance matrix (num of nodes x num of anchor nodes)
    """
    if not os.path.exists(f'datasets/rwr'):
        os.makedirs(f'datasets/rwr')

    rwr_path = f'datasets/rwr/rwr_emb_{dataset}_{ratio:.1f}.npz'
    if os.path.exists(rwr_path):
        print(f"Loading RWR scores from {rwr_path}...", end=" ")
        data = np.load(rwr_path)
        rwr1, rwr2 = data['rwr1'], data['rwr2']
        print("Done")
    else:
        rwr1, rwr2 = rwr_scores(G1, G2, anchor_links, dtype)
        print(f"Saving RWR scores to {rwr_path}...", end=" ")
        np.savez(rwr_path, rwr1=rwr1, rwr2=rwr2)
        print("Done")

    return rwr1, rwr2


def rwr_scores(G1, G2, anchor_links, dtype=np.float32):
    """
    Compute initial node embedding vectors by random walk with restart
    :param G1: network G1, i.e., networkx graph
    :param G2: network G2, i.e., networkx graph
    :param anchor_links: anchor links
    :param dtype: data type
    :return: rwr_score1, rwr_score2: RWR vectors of the networks
    """

    rwr_score1 = rwr_score(G1, anchor_links[:, 0], desc="Computing RWR scores for G1", dtype=dtype)
    rwr_score2 = rwr_score(G2, anchor_links[:, 1], desc="Computing RWR scores for G2", dtype=dtype)

    return rwr_score1, rwr_score2


def rwr_score(G, anchors, restart_prob=0.15, desc='Computing RWR scores', dtype=np.float32):
    """
    Random walk with restart for a single graph
    :param G: network G, i.e., networkx graph
    :param anchors: anchor nodes
    :param restart_prob: restart probability
    :param desc: description for tqdm
    :param dtype: data type
    :return: rwr: rwr vectors of the network
    """

    n = G.number_of_nodes()
    rwr = np.zeros((n, len(anchors))).astype(dtype)

    for i, node in enumerate(tqdm(anchors, desc=desc)):
        s = nx.pagerank(G, personalization={node: 1}, alpha=1-restart_prob)
        for k, v in s.items():
            rwr[k, i] = v

    return rwr

from utils.dataset import wei2bin
def rwr_hetergraph(dataset, G1, G2, adj, gamma=0.15, yita=0.5, lamda=0.5, k=12, x=None):
    """
    random walk on the hetegeneous graph.
    :param G1, G2: graphs/similarity network
    :param anchor_nodes: anchor nodes
    :param adj: bipartial graph interaction matrix, row ~ G1; col ~ G2
    :param gamma: restart probability, 0.15 default
    :param yita:  η∈(0,1) is used to weight the importance of each subnetwork. 0.5 default
    :param lamda: jumping probability of the rand walker jumping from one subnet to another. 0.5 default
    :param k: knn
    :return: 
    rwr: rwr score vectors of the network
    ref:
       Genome-wide inferring gene–phenotype relationship by walking on the heterogeneous network, Bioinformatics. 2010
    """
    
    G1_ind = np.argsort(-G1, axis=1)[:, 1:k+1]
    G1_knn = np.zeros(G1.shape)
    for i in range(G1.shape[0]):
        G1_knn[i, G1_ind[i,:]] = G1[i, G1_ind[i,:]]
    
    G2_ind = np.argsort(-G2, axis=1)[:, 1:k+1]
    G2_knn = np.zeros(G2.shape)
    for i in range(G2.shape[0]):
        G2_knn[i, G2_ind[i,:]] = G2[i, G2_ind[i,:]]
    
    # normalization  
    G1_norm, G2_norm = (1-lamda)*normalize_features(G1_knn), (1-lamda)*normalize_features(G2_knn)
    adj_norm, adj_trans_norm = lamda*normalize_features(adj), lamda*normalize_features(adj.T)
    combin_graph = np.vstack((np.hstack((lamda*G1_knn, adj)), 
                              np.hstack((adj.T, G2_knn))))
    
    # p0 = np.vstack((yita*np.ones(G1.shape[0])/G1.shape[0], 
    #                (1-yita)*np.ones(G2.shape[0])/G2.shape[0]))
    p0 = np.concatenate((yita*np.ones(G1.shape[0])/G1.shape[0], 
                   (1-yita)*np.ones(G2.shape[0])/G2.shape[0]))
    
    n = p0.shape[0]
    keys = range(n)
    p0 = {k: v for k, v in zip(keys, p0)}
    
    G = nx.Graph()
    if x is not None:
        G.add_nodes_from(np.arange(x.shape[0]))
    
    edge_index = wei2bin(combin_graph, 0)
    G.add_edges_from(edge_index)
    G.x = x
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    # G.anchor_nodes = anchor_nodes
    anchors = np.loadtxt(f"datasets/{dataset}/adj.txt")
    tmp = anchors[:,0:2] -1    
    anchors_ = np.hstack((tmp[:,0], tmp[:,1]+1373))    
       
    rwr = np.zeros((n, n)).astype(np.float32)
    for i, node in enumerate(tqdm(range(n), desc='Computing RWR scores')):
        s = nx.pagerank(G, personalization={node: 1}, alpha=1-gamma)
        for k, v in s.items():
            rwr[k, i] = v
    
    return rwr


# hyperedge construction by constrative learning
def construct_hpg_adj(rwr, k):
    """
    construct hpyergraph adjacint matrix from rand walk score matrix rwr
    :param: rwr: rwr score vectors of the network. row: nodes, column: anchor nodes
    :k: number of nodes in each hyperedge
    """
    n, m = rwr.shape
    ind = np.argsort(-rwr, axis=0)[0:k, :]
    hadj = np.zeros([n, m])
    
    for i in range(m):
        hadj[ind[:, i], i] = 1
        
    return hadj


import scipy.sparse as sp
def normalize_features(feat):
    degree = np.asarray(feat.sum(1)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = np.inf
    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)

    return feat_norm
