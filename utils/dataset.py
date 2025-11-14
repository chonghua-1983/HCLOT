import numpy as np
import random
import os
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

def set_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False # train speed is slower after enabling this opts.

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True)
    
    
def load_data(dataset, p, use_attr, dtype=np.float32):
    """
    Load dataset.
    :param dataset: dataset name
    :param p: training ratio
    :param use_attr: whether to use input node attributes
    :param dtype: data type
    :return:
        edge_index1, edge_index2: edge list of graph G1, G2
        x1, x2: input node attributes of graph G1, G2
        anchor_links: training node alignments, i.e., anchor links
        test_pairs: test node alignments
    """

    data = np.load(f'{dataset}_{p:.1f}.npz')
    edge_index1, edge_index2 = data['edge_index1'].T.astype(np.int64), data['edge_index2'].T.astype(np.int64)
    anchor_links, test_pairs = data['pos_pairs'].astype(np.int64), data['test_pairs'].astype(np.int64)
    if use_attr:
        x1, x2 = data['x1'].astype(dtype), data['x2'].astype(dtype)
    else:
        x1, x2 = None, None

    return edge_index1, edge_index2, x1, x2, anchor_links, test_pairs


def build_nx_graph(edge_index, anchor_nodes, x=None):
    """
    Build a networkx graph from edge list and node attributes.
    :param edge_index: edge list of the graph
    :param anchor_nodes: anchor nodes
    :param x: node attributes of the graph
    :return: a networkx graph
    """

    G = nx.Graph()
    if x is not None:
        G.add_nodes_from(np.arange(x.shape[0]))
    G.add_edges_from(edge_index)
    G.x = x
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    G.anchor_nodes = anchor_nodes
    return G


def build_tg_graph(edge_index, x, rwr, dtype=torch.float32):
    """
    Build a PyG Data object from edge list and node attributes.
    :param edge_index: edge list of the graph
    :param x: node attributes of the graph
    :param rwr: random walk with restart scores
    :param dtype: data type
    :return: a PyG Data object
    """

    edge_index_tensor = torch.from_numpy(edge_index.T).to(torch.int64)
    x_tensor = torch.from_numpy(x).to(dtype)
    data = Data(x=x_tensor, edge_index=edge_index_tensor)
    data.rwr = torch.from_numpy(rwr).to(dtype)
    data.adj = to_dense_adj(edge_index_tensor).squeeze(0)
    return data


def load_data_mdad(dataset):
    """
    Fd, Fm: featrure matrix
    Sd, Sm: similarity matrix
    labels: interaction matrix
    """
    
    Fd = np.loadtxt(f"datasets/{dataset}/drug_features.txt")
    Fm = np.loadtxt(f"datasets/{dataset}/microbe_features.txt")
    Sd = np.loadtxt(f"datasets/{dataset}/drug_similarity.txt")
    Sm = np.loadtxt(f"datasets/{dataset}/microbe_similarity.txt")
    
    print('loading labels...')
    labels = np.loadtxt(f"datasets/{dataset}/adj.txt")
    anchors = labels[:,0:2]
    
    num_drug = Fd.shape[0]
    num_microbe =Fm.shape[0]

    temp_label = np.zeros((num_drug, num_microbe))
    for temp in labels:
        temp_label[int(temp[0])-1, int(temp[1])-1] = int(temp[2])
    labels = temp_label
    
    return Fd, Fm, Sd, Sm, labels, anchors


import scipy.sparse as sp
def normalize_adj(adj):
    """
    adj: adjacent matrix
    output: the normalized similarity matrix 
    diag(D).^(-1/2) * adj * diag(D).^(-1/2)
    """
    adj = adj + np.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()


def preprocess_adj(adj):
    # adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    adj_normalized = adj +np.eye(adj.shape[0])

    return adj_normalized


# transform weighted graph to binary graph
def wei2bin(G, threold):
    """
    extract index from graph
    G : graph
    threold : edges above the threold value will be extrcted.

    Returns
    edge_index: numpy array type
    """
    np.fill_diagonal(G, 0) 
    # G[G >= threold] = 1
    edge_index = np.where(G > threold)
    edge_index = np.array(edge_index)
    edge_index = edge_index.T
    
    return edge_index


class getData(object):
    def __init__(self, adj):
        super().__init__()
        
        self.index_0 = np.array(np.where(adj == 0)).T
        self.N_0 = self.index_0.shape[0]
        self.adj_matrix = adj
   