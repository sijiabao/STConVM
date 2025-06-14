#
import time
import numpy as np
import torch
import torch.nn.modules.loss
import torch.nn.functional as F
from sklearn.cluster import KMeans
from .SEDR_module import SEDR_module, SEDR_impute_module
from tqdm import tqdm
import torch.nn as nn
import SEDR
import matplotlib.pyplot as plt

import os
import torch
import numpy as np
import scanpy as sc


def fix_seed(seed):
    import random
    import torch
    from torch.backends import cudnn

    #seed = 666
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def target_distribution(batch):
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


def reconstruction_loss(decoded, x):
    loss_func = torch.nn.MSELoss()
    loss_rcn = loss_func(decoded, x)
    return loss_rcn


# def gcn_loss(preds, labels, mu, logvar, n_nodes, norm, mask=None):
#     if mask is not None:
#         preds = preds * mask
#         labels = labels * mask
#
#     cost = norm * F.binary_cross_entropy_with_logits(preds, labels)
#
#     # see Appendix B from VAE paper:
#     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#     # https://arxiv.org/abs/1312.6114
#     # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     KLD = -0.5 / n_nodes * torch.mean(torch.sum(
#         1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
#     return cost + KLD



def gcn_loss(preds, labels, mu, logvar, n_nodes, norm):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD


class Sedr:
    def __init__(
            self,
            X,
            Y,
            graph_dict,
            rec_w=15,
            gcn_w=0.1,
            w_rna_image=0.84,
            self_w=1,
            dec_kl_w=1,
            random_seed=1000,
            mode = 'clustering',
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu',
    ):
        
        self.rec_w = rec_w
        self.gcn_w = gcn_w
        self.w_rna_image = w_rna_image
        self.self_w = self_w
        self.dec_kl_w = dec_kl_w
        self.device = device
        self.mode = mode
        self.random_seed = random_seed

        if 'mask' in graph_dict:
            self.mask = True
            self.adj_mask = graph_dict['mask'].to(self.device)
        else:
            self.mask = False

        self.cell_num = len(X)

        self.X = torch.FloatTensor(X.copy()).to(self.device)
        self.Y = torch.FloatTensor(Y.copy()).to(self.device)
        self.input_dim = self.X.shape[1]

        self.adj_norm1 = graph_dict["adj_norm1"].to(self.device)
        self.adj_norm2 = graph_dict["adj_norm2"].to(self.device)
        self.adj_label1 = graph_dict["adj_label1"].to(self.device)
        self.adj_label2 = graph_dict["adj_label2"].to(self.device)
        self.image_feature = self.Y.shape[1]

        self.norm_value1 = graph_dict["norm_value1"]
        self.norm_value2 = graph_dict["norm_value2"]
        #fix_seed(self.random_seed)

        if self.mode == 'clustering':
            self.model = SEDR_module(self.input_dim, self.image_feature).to(self.device)
        elif self.mode == 'imputation':
            self.model = SEDR_impute_module(self.input_dim, self.image_feature).to(self.device)
        else:
            raise ValueError(f'{self.mode} is not currently supported!')


    def mask_generator1(self, N=1):
        idx = self.adj_label1.indices()

        list_non_neighbor = []
        for i in range(0, self.cell_num):
            neighbor = idx[1, torch.where(idx[0, :] == i)[0]]
            n_selected = len(neighbor) * N

            # non neighbors
            total_idx = torch.range(0, self.cell_num-1, dtype=torch.float32).to(self.device)
            non_neighbor = total_idx[~torch.isin(total_idx, neighbor)]
            indices = torch.randperm(len(non_neighbor), dtype=torch.float32).to(self.device)
            random_non_neighbor = indices[:n_selected]
            list_non_neighbor.append(random_non_neighbor)

        x = torch.repeat_interleave(self.adj_label1.indices()[0], N)
        y = torch.concat(list_non_neighbor)

        indices = torch.stack([x, y])
        indices = torch.concat([self.adj_label1.indices(), indices], axis=1)

        value = torch.concat([self.adj_label1.values(), torch.zeros(len(x), dtype=torch.float32).to(self.device)])
        adj_mask1 = torch.sparse_coo_tensor(indices, value)

        return adj_mask1
    
    def mask_generator2(self, N=1):
        idx = self.adj_label2.indices()

        list_non_neighbor = []
        for i in range(0, self.cell_num):
            neighbor = idx[1, torch.where(idx[0, :] == i)[0]]
            n_selected = len(neighbor) * N

            # non neighbors
            total_idx = torch.range(0, self.cell_num-1, dtype=torch.float32).to(self.device)
            non_neighbor = total_idx[~torch.isin(total_idx, neighbor)]
            indices = torch.randperm(len(non_neighbor), dtype=torch.float32).to(self.device)
            random_non_neighbor = indices[:n_selected]
            list_non_neighbor.append(random_non_neighbor)

        x = torch.repeat_interleave(self.adj_label2.indices()[0], N)
        y = torch.concat(list_non_neighbor)

        indices = torch.stack([x, y])
        indices = torch.concat([self.adj_label2.indices(), indices], axis=1)

        value = torch.concat([self.adj_label2.values(), torch.zeros(len(x), dtype=torch.float32).to(self.device)])
        adj_mask2 = torch.sparse_coo_tensor(indices, value)

        return adj_mask2

    def train_without_dec(
            self,
            epochs=300,
            lr=0.001,
            decay=0.01,
            N=1,
    ):
        
        self.optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=lr,
            weight_decay=decay)
        
        #fix_seed(self.random_seed)

        self.model.train()

        # list_rec = []
        # list_gcn = []
        # list_self = []
        total_losses = []
        gcn_losses = []
        rec_losses = []
        contrastive_losses = []
        losses_self = []
        
        for epoch in tqdm(range(epochs)):
            self.model.train()
            fix_seed(self.random_seed)
            self.optimizer.zero_grad()

            
            latent_z, mu1, logvar1, mu2, logvar2, de_feat1, de_feat2, _, feat_x, _, loss_self, rna_emb, image_emb = self.model(self.X, self.Y, self.adj_norm1,self.adj_norm2)

            if self.mask:
                pass
            else:

                if self.mode == 'imputation':
                    adj_mask1 = self.mask_generator1(N=0)
                    adj_mask2 = self.mask_generator2(N=0)
                else:
                    adj_mask1 = self.mask_generator1(N=1)
                    adj_mask2 = self.mask_generator2(N=1)
                self.adj_mask1 = adj_mask1
                self.adj_mask2 = adj_mask2
                self.mask = True


            loss_gcn1 = gcn_loss(
                preds=self.model.dc(latent_z, self.adj_mask1),
                # labels=self.adj_label,
                labels=self.adj_mask1.coalesce().values(),
                mu=mu1,
                logvar=logvar1,
                n_nodes=self.cell_num,
                norm=self.norm_value1,
            )
            
            loss_gcn2 = gcn_loss(
                preds=self.model.dc(latent_z, self.adj_mask2),
                # labels=self.adj_label,
                labels=self.adj_mask2.coalesce().values(),
                mu=mu2,
                logvar=logvar2,
                n_nodes=self.cell_num,
                norm=self.norm_value2,
            )
            
            loss_gcn =0.5* loss_gcn1 + 0.5*loss_gcn2

            loss_rec1 = reconstruction_loss(de_feat1, self.X)
            loss_rec2 = reconstruction_loss(de_feat2, self.X)
            loss_rec = 0.5*loss_rec1+0.5*loss_rec2
            
            ##model cl loss 

            rna_cl_train = rna_emb
            image_cl_train = image_emb

            #rna_image_cl_loss
            rna_cl_train_l2_norm = F.normalize(rna_cl_train, dim=1)


            image_cl_train_l2_norm = F.normalize(image_cl_train, dim=1)
            features_cl_rna_image = torch.cat([rna_cl_train_l2_norm.unsqueeze(1), image_cl_train_l2_norm.unsqueeze(1)], dim=1)
            criterion_cl_rna_image = SupConLoss(temperature=0.1,base_temperature=0.1)
            simclr_loss_rna_image = criterion_cl_rna_image(features_cl_rna_image)
            
            loss = self.rec_w * loss_rec + self.gcn_w * loss_gcn + self.self_w * loss_self  + simclr_loss_rna_image*self.w_rna_image
      #       print(f"Epoch [{epoch + 1}/{epochs}], "
      # f"Total Loss: {loss.item():.4f}, "
      # f"Reconstruction Loss: {loss_rec.item():.4f}, "
      # f"GCN Loss: {loss_gcn.item():.4f}, "
      # f"self Loss: {loss_self.item():.4f}, "
      # f"Contrastive Loss: {simclr_loss_rna_image.item():.4f}")
      
      
            loss_rec1 = self.rec_w * loss_rec
            loss_gcn1 = self.gcn_w * loss_gcn
            loss_self1 = self.self_w * loss_self
            simclr_loss_rna_image1 = simclr_loss_rna_image*self.w_rna_image
            
            total_losses.append(loss.detach().cpu().item())
            gcn_losses.append(loss_gcn1.detach().cpu().item())
            losses_self.append(loss_self1.detach().cpu().item())
            rec_losses.append(loss_rec1.detach().cpu().item())
            contrastive_losses.append(simclr_loss_rna_image1.detach().cpu().item())
            

            
            loss.backward()
            self.optimizer.step()
        
        #plt.plot(total_losses, label='Loss')
        plt.plot(rec_losses, label='loss_rec')
        plt.plot(gcn_losses, label='loss_gcn')
        plt.plot(losses_self, label='loss_self')
        plt.plot(contrastive_losses, label='simclr_loss_rna_image')
        plt.legend()
        plt.show()




    def save_model(self, save_model_file):
        torch.save({'state_dict': self.model.state_dict()}, save_model_file)
        print('Saving model to %s' % save_model_file)

    def load_model(self, save_model_file):
        saved_state_dict = torch.load(save_model_file)
        self.model.load_state_dict(saved_state_dict['state_dict'])
        print('Loading model from %s' % save_model_file)

    def process(self):
        self.model.eval()
        latent_z, _, _, _, _, _, _, q, feat_x, gnn_z, _ , rna_emb, image_emb= self.model(self.X, self.Y, self.adj_norm1,self.adj_norm2)
        
        latent_z = latent_z.data.cpu().numpy()
        q = q.data.cpu().numpy()
        feat_x = feat_x.data.cpu().numpy()
        gnn_z = gnn_z.data.cpu().numpy()
        rna_emb = rna_emb.data.cpu().numpy()
        image_emb = image_emb.data.cpu().numpy()
        return latent_z, q, feat_x, gnn_z, rna_emb, image_emb

    def recon(self):
        self.model.eval()
        latent_z, _, _, _, _, de_feat1, de_feat2, q, feat_x, gnn_z, _ , rna_emb, image_emb= self.model(self.X, self.Y, self.adj_norm1,self.adj_norm2)
        de_feat = 0.5*de_feat1 + 0.5*de_feat2
        de_feat = de_feat.data.cpu().numpy()

        # revise std and mean
        from sklearn.preprocessing import StandardScaler
        out = StandardScaler().fit_transform(de_feat)

        return out

    def train_with_dec(
            self,
            epochs=300,
            dec_interval=20,
            dec_tol=0.00,
            N=1,
    ):
        
        # initialize cluster parameter
        # self.train_without_dec(
        #     epochs=epochs,
        #     lr=lr,
        #     decay=decay,
        #     N=N,
        # )
        self.train_without_dec()

        kmeans = KMeans(n_clusters=self.model.dec_cluster_n, n_init=self.model.dec_cluster_n * 2, random_state=42)
        test_z, _, _, _ ,rna_emb, image_emb= self.process()
        y_pred_last = np.copy(kmeans.fit_predict(test_z))

        self.model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(self.device)
        #fix_seed(self.random_seed)
        self.model.train()
        
        total_losses = []
        gcn_losses = []
        rec_losses = []
        contrastive_losses = []
        losses_kl = []

        for epoch in tqdm(range(epochs)):
            # DEC clustering update
            if epoch % dec_interval == 0:
                _, tmp_q, _, _ ,rna_emb, image_emb= self.process()
                tmp_p = target_distribution(torch.Tensor(tmp_q))
                y_pred = tmp_p.cpu().numpy().argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                self.model.train()
                fix_seed(self.random_seed)
                if epoch > 0 and delta_label < dec_tol:
                    print('delta_label {:.4}'.format(delta_label), '< tol', dec_tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            # training model
            torch.set_grad_enabled(True)
            self.optimizer.zero_grad()
            latent_z, mu1, logvar1, mu2, logvar2, de_feat1, de_feat2, out_q, _, _, _ ,rna_emb, image_emb= self.model(self.X, self.Y, self.adj_norm1,self.adj_norm2)

            # if self.mask:
            #     pass
            # else:
            #     adj_mask = self.mask_generator(N)
            #     self.adj_mask = adj_mask
            #     self.mask = True

            loss_gcn1 = gcn_loss(
                preds=self.model.dc(latent_z, self.adj_mask1),
                # labels=self.adj_label,
                labels=self.adj_mask1.coalesce().values(),
                mu=mu1,
                logvar=logvar1,
                n_nodes=self.cell_num,
                norm=self.norm_value1,
            )
            
            loss_gcn2 = gcn_loss(
                preds=self.model.dc(latent_z, self.adj_mask2),
                # labels=self.adj_label,
                labels=self.adj_mask2.coalesce().values(),
                mu=mu2,
                logvar=logvar2,
                n_nodes=self.cell_num,
                norm=self.norm_value2,
            )
            
            loss_gcn = 0.5*loss_gcn1 + 0.5*loss_gcn2
            
            loss_rec1 = reconstruction_loss(de_feat1, self.X)
            loss_rec2 = reconstruction_loss(de_feat2, self.X)
            loss_rec = 0.5*loss_rec1+ 0.5*loss_rec2
            # clustering KL loss
            loss_kl = F.kl_div(out_q.log(), torch.tensor(tmp_p).to(self.device)).to(self.device)
            
            # ##model cl loss 

            # rna_cl_train = rna_emb
            # image_cl_train = image_emb

            # #rna_image_cl_loss
            # rna_cl_train_l2_norm = F.normalize(rna_cl_train, dim=1)


            # image_cl_train_l2_norm = F.normalize(image_cl_train, dim=1)
            # features_cl_rna_image = torch.cat([rna_cl_train_l2_norm.unsqueeze(1), image_cl_train_l2_norm.unsqueeze(1)], dim=1)
            # criterion_cl_rna_image = SupConLoss(temperature=0.1,base_temperature=0.1)
            # simclr_loss_rna_image = criterion_cl_rna_image(features_cl_rna_image)
            
            loss = self.gcn_w * loss_gcn + self.dec_kl_w * loss_kl + self.rec_w * loss_rec #+ simclr_loss_rna_image*self.w_rna_image
      #       print(f"Epoch [{epoch + 1}/{epochs}], "
      # f"Total Loss: {loss.item():.4f}, "
      # f"GCN Loss: {loss_gcn.item():.4f}, "
      # f"kl Loss: {loss_kl.item():.4f},"
      # f"Reconstruction Loss: {loss_rec.item():.4f}, "
      # f"Contrastive Loss: {simclr_loss_rna_image.item():.4f}")
      
            loss_gcn1 = self.gcn_w * loss_gcn
            loss_kl1 = self.dec_kl_w * loss_kl 
            loss_rec1 = self.rec_w * loss_rec 
            #simclr_loss_rna_image1 = simclr_loss_rna_image*self.w_rna_image
      
      
            total_losses.append(loss.detach().cpu().item())
            gcn_losses.append(loss_gcn1.detach().cpu().item())
            losses_kl.append(loss_kl1.detach().cpu().item())
            rec_losses.append(loss_rec1.detach().cpu().item())
            #contrastive_losses.append(simclr_loss_rna_image1.detach().cpu().item())
            

            
            loss.backward()
            self.optimizer.step()
            
        
        #plt.plot(total_losses, label='Loss')
        plt.plot(gcn_losses, label='loss_gcn')
        plt.plot(losses_kl, label='loss_kl')
        plt.plot(rec_losses, label='loss_rec')
        #plt.plot(contrastive_losses, label='simclr_loss_rna_image')
        plt.legend()
        plt.show()


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.1, contrast_mode='all',
                 base_temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda:0')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

