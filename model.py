# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from tqdm import tqdm
from sklearn.cluster import KMeans

def full_block(in_features, out_features, p_drop):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        nn.ELU(),  #ELU()
        nn.Dropout(p=p_drop),
        )

import math

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):  #F.relu
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        support = support + self.bias
        # output = torch.spmm(adj, support)
        output = adj.matmul(support)
        output = self.act(output)
        return output


class Hypergcn(nn.Module):
    def __init__(self, input_dim, params):
        super(Hypergcn, self).__init__()
        self.alpha = 1.0
        self.latent_dim = params.gcn_hidden2 + params.feat_hidden2
        #self.latent_dim = params.feat_hidden2
        
        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(input_dim, params.feat_hidden1, params.p_drop))
        self.encoder.add_module('encoder_L2', full_block(params.feat_hidden1, params.feat_hidden2, params.p_drop))

        # self.decoder = nn.Sequential()
        # self.decoder.add_module('decoder_L0', full_block(self.latent_dim, input_dim, params.p_drop))
        self.decoder = GraphConvolution(self.latent_dim, input_dim, params.p_drop, act=lambda x: x)

        # GCN layers input_dim
        self.gc1 = GraphConvolution(params.feat_hidden2, params.gcn_hidden1, params.p_drop, act=F.relu) #F.relu F.leaky_relu F.prelu nn.LeakyReLU(negative_slope=0.2)
        self.gc2 = GraphConvolution(params.gcn_hidden1, params.gcn_hidden2, params.p_drop, act=lambda x: x)
        self.gc3 = GraphConvolution(params.gcn_hidden1, params.gcn_hidden2, params.p_drop, act=lambda x: x)
        # self.dc = InnerProductDecoder(params.p_drop, act=lambda x: x)
        
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(params.dec_cluster_n, params.gcn_hidden2+params.feat_hidden2))  #+params.feat_hidden2
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def encode(self, x, adj):
        feat_x = self.encoder(x)
        hidden1 = self.gc1(feat_x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj), feat_x
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar, feat_x = self.encode(x, adj)
        gnn_z = self.reparameterize(mu, logvar)
        z = torch.cat((feat_x, gnn_z), 1)
        de_feat = self.decoder(z, adj)
                
        # DEC clustering
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        
        return z, mu, logvar, de_feat, q, feat_x, gnn_z


class HCOT:
    def __init__(self, features, hyper_adj, sim_drug, sim_micobe, dmi_adj, args, device):
        self.device = device
        self.features = torch.tensor(features).to(self.device)
        self.args = args
        self.sim_drug = torch.tensor(sim_drug).to(torch.float32).to(self.device)
        self.sim_micobe = torch.tensor(sim_micobe).to(torch.float32).to(self.device)
        self.dmi_adj = dmi_adj      
        self.hyper_adj = torch.tensor(hyper_adj).to(torch.float32).to(self.device)
        self.input_dim = features.shape[1]
        self.model = Hypergcn(self.input_dim, args).to(self.device)
    
    def train_without_dec(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.model.train()
        
        num_drugs = self.sim_drug.shape[0]
        for _ in tqdm(range(self.args.epochs)):
            self.model.train()
            self.optimizer.zero_grad()
            out, mu, logvar, de_feat, out_q, feat_x, gnn_z = self.model(self.features, self.hyper_adj)
            
            loss_rec = reconstruction_loss(de_feat, self.features)
            loss_sce = sce_loss(de_feat, self.features)
            
            out1, out2 = out[0:num_drugs, :], out[num_drugs:, :] 
            out1, out2 = out1.to(self.device), out2.to(self.device)
                        
            # stru_consistence_loss(ori, emb, device)
            loss_str = (stru_consistence_loss(self.features[0:num_drugs, :], out1, self.device)
                         + stru_consistence_loss(self.features[num_drugs:, :], out2, self.device))/2
            
            loss_contra = loss_contrastive(out1, out2, self.dmi_adj)    
            
            loss = 0.1*loss_str + 0.01*loss_rec #+ 0.5*loss_contra# + 0.01*loss_sce  aBiofilm
            loss.backward()
            self.optimizer.step()
            
    def process(self):
        self.model.eval()
        latent_z, _, _, _, q, feat_x, gnn_z = self.model(self.features, self.hyper_adj)
        latent_z = latent_z.data.cpu().numpy()
        q = q.data.cpu().numpy()
        
        return latent_z, q, feat_x, gnn_z
    
    def recon(self):
        self.mode.eval()
        latent_z, _, _, de_feat, q, feat_x, gnn_z = self.model(self.features, self.hyper_adj)
        de_feat = de_feat.data.cpu().numpy()
        
        # revise std and mean
        from sklearn.preprocessing import StandardScaler
        out = StandardScaler().fit_transform(de_feat)
        return out
    
    def train_with_dec(self):
        self.train_without_dec()
        
        num_drugs = self.sim_drug.shape[0]
        kmeans = KMeans(n_clusters=self.args.dec_cluster_n, n_init=self.args.dec_cluster_n * 2, random_state=42)
        test_z, _, _, _ = self.process()
        
        y_pred_last = np.copy(kmeans.fit_predict(test_z))

        self.model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(self.device)
        self.model.train()
        
        criterion = FusedGWLoss(self.device, self.sim_drug, self.sim_micobe,
                                gw_weight=self.args.gw_weight,
                                gamma_p=self.args.gamma_p,
                                init_threshold_lambda=self.args.init_threshold_lambda,
                                in_iter=self.args.in_iter,
                                out_iter=self.args.out_iter,
                                total_epochs=self.args.epochs).to(self.device)
        
        for epoch in tqdm(range(100)):
            # DEC clutering updata
            if epoch % 20 == 0:
                 _, tmp_q, _, _ = self.process()
                 tmp_p = target_distribution(torch.Tensor(tmp_q))
                 y_pred = tmp_p.cpu().numpy().argmax(1)
                 delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                 y_pred_last = np.copy(y_pred)
                 self.model.train()
                 if epoch > 0 and delta_label < 0:
                    print('delta_label {:.4}'.format(delta_label), '< tol', 0)
                    print('Reached tolerance threshold. Stopping training.')
                    break
            
            # training model
            torch.set_grad_enabled(True)
            self.optimizer.zero_grad()
            out, _, _, de_feat, out_q, _, _ = self.model(self.features, self.hyper_adj)
            out = out/torch.sum(abs(out))

            loss_rec = reconstruction_loss(de_feat, self.features)
            out1, out2 = out[0:num_drugs, :], out[num_drugs:, :] 
            out1, out2 = out1.to(self.device), out2.to(self.device)
            
            # clustering KL loss
            loss_kl = F.kl_div(out_q.log(), torch.tensor(tmp_p).to(self.device)).to(self.device)
            
            # stru_consistence_loss(ori, emb, device)
            loss_str = (stru_consistence_loss(self.features[0:num_drugs, :], out1, self.device)
                        + stru_consistence_loss(self.features[num_drugs:, :], out2, self.device))/2
            
            # ot loss
            loss_ot, similarity, threshold_lambda = criterion(out1=out1, out2=out2)
            # loss_contra = loss_contrastive1(out1, out2, self.dmi_adj, similarity, self.device)
            loss_contra = loss_contrastive(out1, out2, self.dmi_adj)
            
            loss = 10*loss_contra + 5*loss_kl + 0.01*loss_str + 0.5*loss_ot  # MDAD  10*loss_contra +
                                                       
            loss.backward()
            self.optimizer.step()
            
            # if epoch % 20 == 0:
            #     print(f'Epoch {epoch}, Loss: {loss.item():.6f}')  
        
        pre_score = 1 / (1 + torch.exp(-out1 @ out2.T))
        pre_score = pre_score.detach().data.cpu()
        
        return pre_score, similarity, out1, out2, loss_contra   
    

class FusedGWLoss(torch.nn.Module):
    def __init__(self, device, Sd, Sm, gw_weight=20, gamma_p=1e-2, init_threshold_lambda=1, in_iter=5,
                 out_iter=10, total_epochs=250):
        super().__init__()
        self.device = device
        self.gw_weight = gw_weight
        self.gamma_p = gamma_p
        self.in_iter = in_iter
        self.out_iter = out_iter
        self.total_epochs = total_epochs

        self.n1, self.n2 = Sd.shape[0], Sm.shape[0]
        self.threshold_lambda = init_threshold_lambda / (self.n1 * self.n2)
        self.adj1, self.adj2 = torch.tensor(Sd), torch.tensor(Sm)
        self.adj1, self.adj2 = self.adj1.to(self.device), self.adj2.to(self.device)
        self.H = torch.ones(self.n1, self.n2).to(torch.float32).to(self.device)
        # self.H[anchor1, anchor2] = 0

    def forward(self, out1, out2):
        inter_c = torch.exp(-(out1 @ out2.T)).to(self.device)
        intra_c1 = torch.exp(-(out1 @ out1.T)) * self.adj1
        intra_c2 = torch.exp(-(out2 @ out2.T)) * self.adj2
                
        # inter_c = (1-cos_sim(out1, out2)).to(self.device)
        # intra_c1 = (1-cos_sim(out1, out1)) * self.adj1
        # intra_c2 = (1-cos_sim(out2, out2)) * self.adj2
        
        intra_c1, intra_c2 = intra_c1.to(self.device), intra_c2.to(self.device), 
        
        with torch.no_grad():
            s = sinkhorn_stable(inter_c, intra_c1, intra_c2,
                                gw_weight=self.gw_weight,
                                gamma_p=self.gamma_p,
                                threshold_lambda=self.threshold_lambda,
                                in_iter=self.in_iter,
                                out_iter=self.out_iter,
                                device=self.device)
            self.threshold_lambda = 0.05 * self.update_lambda(inter_c, intra_c1, intra_c2, s) + 0.95 * self.threshold_lambda

        s_hat = s - self.threshold_lambda

        # Wasserstein Loss
        w_loss = torch.sum(inter_c * s_hat)

        # Gromov-Wasserstein Loss
        a = torch.sum(s_hat, dim=1)
        b = torch.sum(s_hat, dim=0)
        gw_loss = torch.sum(
            (intra_c1 ** 2 @ a.view(-1, 1) @ torch.ones((1, self.n2)).to(torch.float32).to(self.device) +
             torch.ones((self.n1, 1)).to(torch.float32).to(self.device) @ b.view(1, -1) @ intra_c2 ** 2 -
             2 * intra_c1 @ s_hat @ intra_c2.T) * s_hat)

        loss = w_loss + self.gw_weight * gw_loss + 20
        return loss, s, self.threshold_lambda

    def update_lambda(self, inter_c, intra_c1, intra_c2, s):
        k1 = torch.sum(inter_c)

        one_mat = torch.ones(self.n1, self.n2).to(torch.float32).to(self.device)
        mid = intra_c1 ** 2 @ one_mat * self.n2 + one_mat @ intra_c2 ** 2 * self.n1 - 2 * intra_c1 @ one_mat @ intra_c2.T
        k2 = torch.sum(mid * s)
        k3 = torch.sum(mid)

        return (k1 + 2 * self.gw_weight * k2) / (2 * self.gw_weight * k3)

from utils.tools import *
def sinkhorn_stable(inter_c, intra_c1, intra_c2, threshold_lambda, in_iter=5, out_iter=10, gw_weight=20, gamma_p=1e-2,
                    device='cpu'):
    n1, n2 = inter_c.shape
    # marginal distribution
    a = torch.ones(n1).to(torch.float32).to(device) / n1
    b = torch.ones(n2).to(torch.float32).to(device) / n2
    # lagrange multiplier
    f = torch.ones(n1).to(torch.float32).to(device) / n1
    g = torch.ones(n2).to(torch.float32).to(device) / n2
    # transport plan
    s = torch.ones((n1, n2)).to(torch.float32).to(device) / (n1 * n2)

    def soft_min_row(z_in, eps):
        hard_min = torch.min(z_in, dim=1, keepdim=True)[0]
        soft_min = hard_min - eps * torch.log(torch.sum(torch.exp(-(z_in - hard_min) / eps), dim=1, keepdim=True))
        return soft_min.squeeze(-1)

    def soft_min_col(z_in, eps):
        hard_min = torch.min(z_in, dim=0, keepdim=True)[0]
        soft_min = hard_min - eps * torch.log(torch.sum(torch.exp(-(z_in - hard_min) / eps), dim=0, keepdim=True))
        return soft_min.squeeze(0)

    for i in range(out_iter):
        a_hat = torch.sum(s - threshold_lambda, dim=1)
        b_hat = torch.sum(s - threshold_lambda, dim=0)
        temp = (intra_c1 ** 2 @ a_hat.view(-1, 1) @ torch.ones((1, n2)).to(torch.float32).to(device) +
                torch.ones((n1, 1)).to(torch.float32).to(device) @ b_hat.view(1, -1) @ intra_c2 ** 2)
        L = temp - 2 * intra_c1 @ (s - threshold_lambda) @ intra_c2.T
        cost = inter_c + gw_weight * L

        Q = cost
        for j in range(in_iter):
            # log-sum-exp stabilization
            f = soft_min_row(Q - g.view(1, -1), gamma_p) + gamma_p * torch.log(a)
            g = soft_min_col(Q - f.view(-1, 1), gamma_p) + gamma_p * torch.log(b)
        s = 0.05 * s + 0.95 * torch.exp((f.view(-1, 1) + g.view(-1, 1).T - Q) / gamma_p)
        s = s.data.cpu()
        s = top_sim(s, k=50)
        s = torch.tensor(s).to(device)

    return s


class MLP_cls(nn.Module):
    def __init__(self, in_features, out_features, p_drop):
        super(MLP_cls, self).__init__()
        
        self.encoder = nn.Sequential()
        # self.encoder.add_module('encoder_L1', full_block(in_features, 1024, p_drop))
        self.encoder.add_module('encoder_L2', full_block(in_features, 512, p_drop))
        self.encoder.add_module('encoder_L3', full_block(512, 256, p_drop))
        self.encoder.add_module('encoder_L4', nn.Sequential(
            nn.Linear(256, out_features),
            nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
            nn.Sigmoid(),
            ))        

    def forward(self, X_train):
        encode_X = self.encoder(X_train)
              
        return encode_X
    
    
import numpy as np
def consine(mat1, mat2):
    """
    mat1, mat2 : numpy addary
    Returns
    -------
    cosine_similarity : TYPE
    """
    mat1_norm = mat1 / np.linalg.norm(mat1, axis=1, keepdims=True) # 
    mat2_norm = mat2 / np.linalg.norm(mat2, axis=1, keepdims=True)
    
    cosine_similarity = np.dot(mat1_norm, mat2_norm.T)
    
    return cosine_similarity

def cos_sim(mat1, mat2):
    """
    mat1, mat2 : torch tensor
    """
    x = mat1.unsqueeze(1)
    y = mat2.unsqueeze(0)
    cos_sim = F.cosine_similarity(x, y, dim=-1)  
    return cos_sim

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

def Eud(mat1, mat2):
    """
    mat1, mat2 : torch tensor
    """
    sq_1 = mat1**2
    sum_sq_1 = torch.sum(sq_1, dim=1).unsqueeze(1)
    sq_2 = mat2**2
    sum_sq_2 = torch.sum(sq_2, dim=1).unsqueeze(0)
    mat2t = mat2.t()
    dist = torch.sqrt(sum_sq_1 + sum_sq_2 - 2*mat1.mm(mat2t))

    return dist


def loss_contrastive(em1, em2, dmi_adj):
    """
    em1 : torch tensor. drug embedding
    em2 : torch tensor. mircobe embedding
    dmi_adj : np.array. drug-microbe interation matrix

    Returns
    loss : contrastive loss
    """
    dmi_adj = np.array(dmi_adj)
    n,m = dmi_adj.shape[0], dmi_adj.shape[1]
    index_matrix = np.array(np.where(dmi_adj == 1))    
    # positive_num = index_matrix.shape[1]
    index_matrix = tuple(index_matrix)
    
    pre_interaction = cos_sim(em1, em2)
    pos = pre_interaction[index_matrix]
    ori_pos = torch.tensor(dmi_adj)[index_matrix]
    
    loss = torch.sum(pos) - torch.sum(ori_pos)
    
    # loss = torch.sum(pre_interaction) - 2*torch.sum(pos)
    
    return abs(loss)/(n*m)

def loss_contrastive1(em1, em2, dmi_adj, sot, device):
    """
    em1 : torch tensor. drug embedding
    em2 : torch tensor. mircobe embedding
    dmi_adj : np.array. drug-microbe interation matrix
    sot:  ot matrix from ot computation
        
    Returns
    loss : contrastive loss
    """
    dmi_adj = np.array(dmi_adj)
    n, m = sot.shape
    index_matrix = np.array(np.where(dmi_adj == 1))    
    # positive_num = index_matrix.shape[1]
    index_matrix = tuple(index_matrix)
    
    pre_interaction = cos_sim(em1, em2)
    pos = pre_interaction[index_matrix]
    ori_pos = torch.tensor(dmi_adj)[index_matrix]
    
    loss1 = torch.sum(pos) - torch.sum(ori_pos) 
    
    neg_ind_matrix = neg_smp_from_sot(sot, 2).to(device)  # index for negative samples    
    neg = pre_interaction * neg_ind_matrix
    
    loss2 = torch.sum(neg)
    
    return (abs(loss1)+abs(loss2))/(n*m)


def neg_smp_from_sot(x: torch.Tensor, k) -> torch.Tensor:
    """
    transform x to a binary matrix according to knn
    Returns
    """
    n, m = x.shape
    ind_by_row = torch.argsort(x, dim=1)
    ind_by_row = ind_by_row[:, 0:k]
    x_trans_row = torch.zeros(n, m)
    
    for i in range(n):
        for j in range(k):
            x_trans_row[i, ind_by_row[i,j]] = 1

    return x_trans_row


# structure consistence loss
def stru_consistence_loss(ori, emb, device):
    ori = torch.tensor(ori).float().to(device)
    ori_dist = torch.cdist(ori, ori, p=2)
    ori_dist = torch.div(ori_dist, torch.max(ori_dist)).to(device)
    emb_dist = torch.cdist(emb, emb, p=2)
    emb_dist = torch.div(emb_dist, torch.max(emb_dist)).to(device)
    
    n_items = emb.size(dim=0) * emb.size(dim=0)
    loss = torch.div(torch.sum(torch.mul(1.0 - emb_dist, ori_dist)), n_items).to(device)

    return loss

def reconstruction_loss(decoded, x):
    loss_func = torch.nn.MSELoss()
    loss_rcn = loss_func(decoded, x)
    return loss_rcn

def target_distribution(q):
    p = q**2 / q.sum(0)
    return (p.t() / p.sum(1)).t()
   
def cluster_loss(p, q):
    def kld(target, pred):
        return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=-1))
    kldloss = kld(p, q)
    return kldloss

def gcn_loss(preds, labels, mu, logvar, n_nodes, norm, mask=None):
    if mask is not None:
        preds = preds * mask
        labels = labels * mask
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD

