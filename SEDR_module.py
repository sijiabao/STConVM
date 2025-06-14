#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from functools import partial

# import os
# import torch
# import numpy as np
# import scanpy as sc


# def fix_seed(seed):
#     import random
#     import torch
#     from torch.backends import cudnn

#     #seed = 666
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     cudnn.deterministic = True
#     cudnn.benchmark = False
    
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def full_block(in_features, out_features, p_drop):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        nn.ELU(),
        nn.Dropout(p=p_drop),
    )


# GCN Layer
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output





class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z, mask):
        col = mask.coalesce().indices()[0]
        row = mask.coalesce().indices()[1]
        result = self.act(torch.sum(z[col] * z[row], axis=1))

        return result
    

class AttentionLayer(nn.Module):   
    def __init__(self, in_features, out_features, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.w_omega = Parameter(torch.FloatTensor(in_features, out_features))
        self.u_omega = Parameter(torch.FloatTensor(out_features, 1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)
        
    def forward(self, emb1, emb2):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        self.emb = torch.cat(emb, dim=1)
        
        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu=  torch.matmul(self.v, self.u_omega)
        self.wqz = F.softmax(torch.squeeze(self.vu) + 1e-6)  
        
        emb_combined = torch.matmul(torch.transpose(self.emb,1,2), torch.unsqueeze(self.wqz, -1))
    
        return torch.squeeze(emb_combined), self.wqz  



class SEDR_module(nn.Module):
    def __init__(
            self,
            input_dim,
            image_feature,
            cl_emb_dim=32,
            feat_hidden1=64,
            feat_hidden2=16,
            gcn_hidden1=64,
            gcn_hidden2=16,
            p_drop=0.2,
            alpha=1.0,
            dec_clsuter_n=10,
    ):
        super(SEDR_module, self).__init__()
        self.input_dim = input_dim
        self.image_feature = image_feature
        self.cl_emb_dim = cl_emb_dim
        self.feat_hidden1 = feat_hidden1
        self.feat_hidden2 = feat_hidden2
        self.gcn_hidden1 = gcn_hidden1
        self.gcn_hidden2 = gcn_hidden2
        self.p_drop = p_drop
        self.alpha = alpha
        self.dec_cluster_n = dec_clsuter_n
        self.latent_dim = self.gcn_hidden2 + self.feat_hidden2

        
        self.fc_image = nn.Linear(self.image_feature, self.cl_emb_dim)

        # feature 
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(self.input_dim, self.feat_hidden1, self.p_drop))
        self.encoder.add_module('encoder_L2', full_block(self.feat_hidden1, self.feat_hidden2, self.p_drop))

        self.decoder = GraphConvolution(self.latent_dim, self.input_dim, self.p_drop, act=lambda x: x)
        # self.decoder = nn.Sequential()
        # self.decoder.add_module('decoder_L0', full_block(self.latent_dim, input_dim, self.p_drop))
        
        


        self.gc1 = GraphConvolution(self.feat_hidden2, self.gcn_hidden1, self.p_drop, act=F.relu)
        self.gc2 = GraphConvolution(self.gcn_hidden1, self.gcn_hidden2, self.p_drop, act=lambda x: x)
        self.gc3 = GraphConvolution(self.gcn_hidden1, self.gcn_hidden2, self.p_drop, act=lambda x: x)
        self.dc = InnerProductDecoder(self.p_drop, act=lambda x: x)
        self.atten_cross = AttentionLayer(self.gcn_hidden2, self.gcn_hidden2)
        self.fc_rna = nn.Linear(self.latent_dim, self.cl_emb_dim)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(self.dec_cluster_n, self.gcn_hidden2 + self.feat_hidden2))

        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        #########################
        self.enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))
        self._mask_rate = 0.8
        self.criterion = self.setup_loss_fn(loss_fn='sce')

    def setup_loss_fn(self, loss_fn, alpha_l=3):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=3)
        else:
            raise NotImplementedError
        return criterion

    def encode(self, x, adj1):
        feat_x = self.encoder(x)
        hidden1 = self.gc1(feat_x, adj1)
        return self.gc2(hidden1, adj1), self.gc3(hidden1, adj1),feat_x
    
    def encode1(self, x, adj2):
        feat_x = self.encoder(x)
        hidden2 = self.gc1(feat_x, adj2)
        return self.gc2(hidden2, adj2), self.gc3(hidden2, adj2),feat_x

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, image_feature, adj1, adj2):

        adj1, x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(adj1, x, self._mask_rate)  #
        adj2, x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(adj2, x, self._mask_rate)

        mu1, logvar1, feat_x = self.encode(x, adj1)
        mu2, logvar2, feat_x = self.encode1(x, adj2)
        gnn_z1 = self.reparameterize(mu1, logvar1)
        gnn_z2 = self.reparameterize(mu2, logvar2)
        
        gnn_z, alpha_1_2 = self.atten_cross(gnn_z1, gnn_z2)
        z = torch.cat((feat_x, gnn_z), 1)
        rna_emb = self.fc_rna(z)
        de_feat1 = self.decoder(rna_emb,adj1)
        de_feat2 = self.decoder(rna_emb,adj2)
        
        

        image_encoder_emb = torch.flatten(image_feature, start_dim=1)
        image_emb = self.fc_image(image_encoder_emb) 

        # DEC clustering
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        # self-construction loss
        recon1 = de_feat1.clone()  #
        x_init1 = x[mask_nodes]  #
        x_rec1 = recon1[mask_nodes]  #

        loss1 = self.criterion(x_rec1, x_init1)  #
        
        recon2 = de_feat2.clone()  #
        x_init2 = x[mask_nodes]  #
        x_rec2 = recon2[mask_nodes]  #

        loss2 = self.criterion(x_rec2, x_init2)
        loss = loss1 + loss2
        # loss = 0
        

        return z, mu1, logvar1, mu2, logvar2, de_feat1, de_feat2, q, feat_x, gnn_z, loss, rna_emb, image_emb

    def encoding_mask_noise(self, adj, x, mask_rate=0.3):
        num_nodes = adj.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        out_x = x.clone()
        token_nodes = mask_nodes
        #out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_adj = adj.clone()

        return use_adj, out_x, (mask_nodes, keep_nodes)
    


class SEDR_impute_module(nn.Module):
    def __init__(
            self,
            input_dim,
            image_feature,
            cl_emb_dim=32,
            feat_hidden1=64,
            feat_hidden2=16,
            gcn_hidden1=64,
            gcn_hidden2=16,
            p_drop=0.2,
            alpha=1.0,
            dec_clsuter_n=10,
    ):
        super(SEDR_impute_module, self).__init__()
        self.input_dim = input_dim
        self.image_feature = image_feature
        self.cl_emb_dim = cl_emb_dim
        self.feat_hidden1 = feat_hidden1
        self.feat_hidden2 = feat_hidden2
        self.gcn_hidden1 = gcn_hidden1
        self.gcn_hidden2 = gcn_hidden2
        self.p_drop = p_drop
        self.alpha = alpha
        self.dec_cluster_n = dec_clsuter_n
        self.latent_dim = self.gcn_hidden2 + self.feat_hidden2
        
        self.fc_image = nn.Linear(self.image_feature, self.cl_emb_dim)

        # feature autoencoder
        self.encoder = nn.Sequential()
        self.encoder.add_module('encoder_L1', full_block(self.input_dim, self.feat_hidden1, self.p_drop))
        self.encoder.add_module('encoder_L2', full_block(self.feat_hidden1, self.feat_hidden2, self.p_drop))

        self.decoder = GraphConvolution(self.latent_dim, self.input_dim, self.p_drop, act=lambda x: x)

        # GCN layers
        self.gc1 = GraphConvolution(self.feat_hidden2, self.gcn_hidden1, self.p_drop, act=F.relu)
        self.gc2 = GraphConvolution(self.gcn_hidden1, self.gcn_hidden2, self.p_drop, act=lambda x: x)
        self.gc3 = GraphConvolution(self.gcn_hidden1, self.gcn_hidden2, self.p_drop, act=lambda x: x)
        self.dc = InnerProductDecoder(self.p_drop, act=lambda x: x)
        self.atten_cross = AttentionLayer(self.gcn_hidden2, self.gcn_hidden2)
        self.fc_rna = nn.Linear(self.latent_dim, self.cl_emb_dim)
        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(self.dec_cluster_n, self.latent_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        #########################
        self.enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))
        self._mask_rate = 0.8
        self.criterion = self.setup_loss_fn(loss_fn='sce')

    def setup_loss_fn(self, loss_fn, alpha_l=3):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=3)
        else:
            raise NotImplementedError
        return criterion

    def encode(self, x, adj1):
        feat_x = self.encoder(x)
        hidden1 = self.gc1(feat_x, adj1)
        return self.gc2(hidden1, adj1), self.gc3(hidden1, adj1), feat_x
    
    def encode1(self, x, adj2):
        feat_x = self.encoder(x)
        hidden2 = self.gc1(feat_x, adj2)
        return self.gc2(hidden2, adj2), self.gc3(hidden2, adj2),feat_x

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, image_feature, adj1, adj2):

        adj1, x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(adj1, x, self._mask_rate)  #
        adj2, x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(adj2, x, self._mask_rate)

        mu1, logvar1, feat_x = self.encode(x, adj1)
        mu2, logvar2, feat_x = self.encode1(x, adj2)
        gnn_z1 = self.reparameterize(mu1, logvar1)
        gnn_z2 = self.reparameterize(mu2, logvar2)
        
        gnn_z, alpha_1_2 = self.atten_cross(gnn_z1, gnn_z2)
        z = torch.cat((feat_x, gnn_z), 1)
        rna_emb = self.fc_rna(z)
        de_feat1 = self.decoder(rna_emb,adj1)
        de_feat2 = self.decoder(rna_emb,adj2)
        
        

        image_encoder_emb = torch.flatten(image_feature, start_dim=1)
        image_emb = self.fc_image(image_encoder_emb) 

        # DEC clustering
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        # self-construction loss
        recon1 = de_feat1.clone()  #
        x_init1 = x[mask_nodes]  #
        x_rec1 = recon1[mask_nodes]  #

        loss1 = self.criterion(x_rec1, x_init1)  #
        
        recon2 = de_feat2.clone()  #
        x_init2 = x[mask_nodes]  #
        x_rec2 = recon2[mask_nodes]  #

        loss2 = self.criterion(x_rec2, x_init2)
        loss = loss1 + loss2
        # loss = 0
        

        return z, mu1, logvar1, mu2, logvar2, de_feat1, de_feat2, q, feat_x, gnn_z, loss, rna_emb, image_emb

    def encoding_mask_noise(self, adj, x, mask_rate=0.3):
        num_nodes = adj.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        out_x = x.clone()
        token_nodes = mask_nodes
        #out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_adj = adj.clone()

        return use_adj, out_x, (mask_nodes, keep_nodes)
