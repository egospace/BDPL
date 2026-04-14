import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, xavier_normal_
from TransformerEncoder import TransformerEncoder
import torch.nn.functional as F
import math
import numpy as np

class DAGraph(nn.Module):
    def __init__(self, args, item_num, user_num, seq_len, use_cuda=True):
        super(DAGraph, self).__init__()
        
        self.hidden_size = args.hidden_size
        self.max_seq_length = seq_len
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.user_num = user_num
        self.item_num = item_num
        self.behavior_num = 3
    
        self.user_embedding = nn.Embedding(self.user_num, self.hidden_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.item_num, self.hidden_size, padding_idx=0)
        self.behavior_embedding = nn.Embedding(self.behavior_num, self.hidden_size, padding_idx=0)
        
        self.emb_drop_out = args.emb_drop_out
        self.att_drop_out = args.att_drop_out
        self.dropout_prob = args.dropout_prob
        
        
        self.filter_drop_rate = 0.0
        self.layer_num = args.num_layer
        self.mlp_layers = args.mlp_layers
        
        self.batch_size = args.batch_size
        
        self.initializer_range = 0.02
        self.LayerNorm = nn.LayerNorm(self.hidden_size, 1e-12)
        
        self.emb_dropout = nn.Dropout(self.emb_drop_out)
        self.att_dropout = nn.Dropout(self.att_drop_out)
        self.long_dropout = nn.Dropout(self.dropout_prob)
        
        self.W_g1 = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        self.W_g2 = nn.Parameter(torch.Tensor(self.max_seq_length, self.hidden_size)) 
        
        # self.W1_g1 = nn.Parameter(torch.Tensor(self.hidden_size, 1))
        # self.W1_g2 = nn.Parameter(torch.Tensor(self.max_seq_length, self.hidden_size)) 

        self.W_s = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.W_l = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.b = nn.Parameter(torch.Tensor(1, self.hidden_size))
        self.long_w1 = nn.Parameter(torch.Tensor(self.max_seq_length, self.hidden_size)) 
        self.long_w2 = nn.Parameter(torch.Tensor(self.hidden_size, self.max_seq_length)) 
        
        # self.long1_w1 = nn.Parameter(torch.Tensor(self.max_seq_length, self.hidden_size)) 
        # self.long1_w2 = nn.Parameter(torch.Tensor(self.hidden_size, self.max_seq_length)) 
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        
        self.fc = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.fc_score = nn.Linear(self.hidden_size, self.item_num)
        
        self.loss_fuc = nn.CrossEntropyLoss()
        
        self.lmd_short = args.lmd_short
        self.lmd_long = args.lmd_long
        # self.lmd_sem = args.lmd_sem
        # self.ssl = args.contrast
        self.cl_tau = args.cl_tau
        self.sim = args.sim
        self.aug_nce_fct = nn.CrossEntropyLoss() 
        
        xavier_uniform_(self.W_g1)
        xavier_uniform_(self.W_g2)
        # xavier_uniform_(self.W1_g1)
        # xavier_uniform_(self.W1_g2)
        xavier_uniform_(self.b)
        xavier_uniform_(self.W_s)
        xavier_uniform_(self.W_l)
        xavier_uniform_(self.long_w1)
        xavier_uniform_(self.long_w2)
        # xavier_uniform_(self.long1_w1)
        # xavier_uniform_(self.long1_w2)

        
        self.GCN_out = GCNout(
            args=args, 
            hidden_dim=self.hidden_size, 
            behavior_embedding=self.behavior_embedding, 
            device=self.device)
        
        self.soft_att_out = SoftAttnout(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.hidden_size,
            session_len=self.max_seq_length,
            batch_norm=True,
            feat_drop=self.att_drop_out,
            activation=nn.PReLU(self.hidden_size),
        )
        self.att_out = Attout(
            args=args,
            max_seq_length = self.max_seq_length
        )
        
        self.apply(self._init_weights)

    
    def forward(self, inputs, graph, train_flag):
        self.graph = graph
        item_seq, user, behavior_seq, target_behavior, item_seq_len, target_item = inputs
        item_seq_len = item_seq_len.unsqueeze(1)
        if train_flag:
            target_item = target_item
            target_behavior = target_behavior
        else:
           
            target_item = item_seq.gather(1, item_seq_len - 1).squeeze(-1)
            target_behavior = (torch.zeros_like(target_item)+2).to(self.device)
            
        target_behavior_embedding = self.behavior_embedding(target_behavior)
        # calculate the mask
        mask = torch.ones(item_seq.shape, dtype=torch.float, device=item_seq.device) * item_seq.gt(0)
        
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        
        adj_mat = self.graph

       
        e2e_in, e2e_out, p2p_in, p2p_out, e2p_in, e2p_out, \
            iu, iu_p, iu_c, \
                ui, ui_p, ui_c, \
                    uu_p_10, uu_c_10 = adj_mat
       
        
        behavior_u_embs = []
        behavior_i_embs = []
        origin_user_emb, origin_item_emb = users_emb, items_emb
       
        for b in range(self.behavior_num-1, 0, -1):
            u_embs = [users_emb]
            i_embs = [items_emb]
            for layer in range(self.layer_num):
                origin_user_emb, origin_item_emb = users_emb, items_emb
                # u_embs = []
                # i_embs = []
                agg_user_emb, agg_item_emb = self.GCN_out(b, origin_user_emb, origin_item_emb, e2e_in, e2e_out, p2p_in, p2p_out, e2p_in, e2p_out, iu, iu_p, iu_c, ui, ui_p, ui_c, uu_p_10, uu_c_10)
                u_embs.append(agg_user_emb) 
                i_embs.append(agg_item_emb)
                users_emb, items_emb = agg_user_emb, agg_item_emb
            
            u_embs = torch.stack(u_embs, dim=2)
            i_embs = torch.stack(i_embs, dim=2)

            pro_u_embs = torch.mean(u_embs, dim=2)
            pro_i_embs = torch.mean(i_embs, dim=2)
        
            # users_emb, items_emb = pro_u_embs.float() @ self.W_up, pro_i_embs.float() @ self.W_ip
            users_emb, items_emb = pro_u_embs.float(), pro_i_embs.float()
            behavior_u_embs.append(pro_u_embs)
            behavior_i_embs.append(pro_i_embs)
       
        p_user_emb, p_item_emb = torch.mean(torch.stack(behavior_u_embs, dim=2), dim=2).float(), torch.mean(torch.stack(behavior_i_embs, dim=2), dim=2).float()

        mask = mask.unsqueeze(2)
        
        
        item_emb, user_emb = p_item_emb[item_seq], p_user_emb[user]
        
        Q = torch.mul(item_emb, self.sigmoid(item_emb @ self.W_g1 + (self.W_g2 @ target_behavior_embedding.T).T.unsqueeze(-1))).squeeze(-1)
        
        for layer in range(self.mlp_layers):
            Q = Q + (self.gelu(self.LayerNorm(Q).transpose(-1, -2)  @ self.long_w1) @ self.long_w2).transpose(-1, -2) 
      
        
        Q_mean = self.long_dropout(torch.mean(Q, dim=1))
        behavior_emb = self.behavior_embedding(behavior_seq)
        att_out, seq_output = self.att_out(item_emb, item_seq, item_seq_len, behavior_emb)
        
        gate = self.sigmoid(seq_output @ self.W_s + Q_mean @ self.W_l + self.b)
        out = gate * seq_output + (1 - gate) * Q_mean
        
        out = self.LayerNorm(out) 
        
        short = seq_output
      
        return out, short, self.LayerNorm(p_item_emb[target_item])
        # return out, long_att_out, long, short, self.LayerNorm(p_item_emb[target_item])
    
    
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size
    
        z = torch.cat((z_i, z_j), dim=0)
    
        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp
    
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
    
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        
        # if batch_size != self.batch_size:
        batch_size = torch.tensor(batch_size, dtype=torch.long, device=positive_samples.device)
        mask = self.mask_correlated_samples(batch_size)
        # else:
        #     mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)
            
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        
        return logits, labels
        
    def decompose(self, z_i, z_j, origin_z, batch_size):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size
    
        z = torch.cat((z_i, z_j), dim=0)
    
        # pairwise l2 distace
        sim = torch.cdist(z, z, p=2)
    
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
    
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        alignment = positive_samples.mean()

        # pairwise l2 distace
        sim = torch.cdist(origin_z, origin_z, p=2)
        mask = torch.ones((batch_size, batch_size), dtype=bool)
        mask = mask.fill_diagonal_(0)
        negative_samples = sim[mask].reshape(batch_size, -1)
        uniformity = torch.log(torch.exp(-2 * negative_samples).mean())
        
        return alignment, uniformity

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            xavier_uniform_(module.weight)
            # module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        # elif isinstance(module, nn.Embedding):
        #     hidden_size = module.weight.size()[1]
        #     bound = 6 / math.sqrt(hidden_size)
        #     nn.init.uniform_(module.weight, a=-bound, b=bound)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        elif isinstance(module, nn.LSTM):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

class GCNout(nn.Module):
    def __init__(self, args, hidden_dim, behavior_embedding, device):
        super().__init__()
        
        self.feat_drop = nn.Dropout(args.emb_drop_out)
        self.hidden_dim = hidden_dim
        self.device = device
        self.behavior_embedding = behavior_embedding
        
        # self.W_ui_p = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        # self.W_ui_c = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.W_uu = nn.Parameter(torch.Tensor(2 * self.hidden_dim, 1))
        self.W_ii = nn.Parameter(torch.Tensor(2 * self.hidden_dim, 1))
        
        # self.W_uu_p = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        # self.W_uu_c = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))

        self.I_e2e_in = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.I_e2e_out = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.I_p2p_in = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.I_p2p_out = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.I_e2p_out = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.I_e2p_in = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        
        # self.W_ig = nn.Parameter(torch.Tensor(2 * self.hidden_dim, 1))
        # self.W_ug = nn.Parameter(torch.Tensor(2 * self.hidden_dim, 1))

        self.conv_user = nn.Conv2d(1, 1, (1, 2), bias=True)
        self.conv_item = nn.Conv2d(1, 1, (1, 2), bias=True)
        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=1)
        self.LayerNorm = nn.LayerNorm(self.hidden_dim, 1e-12)
        self.sigmod = nn.Sigmoid()
        
        # xavier_uniform_(self.W_uu_p)
        # xavier_uniform_(self.W_uu_c)
        xavier_uniform_(self.W_uu)
        xavier_uniform_(self.W_ii)
        # xavier_uniform_(self.W_ig)
        # xavier_uniform_(self.W_ug)
        xavier_uniform_(self.I_e2e_in)
        xavier_uniform_(self.I_e2e_out)
        xavier_uniform_(self.I_p2p_in)
        xavier_uniform_(self.I_p2p_out)
        xavier_uniform_(self.I_e2p_out)
        xavier_uniform_(self.I_e2p_in)
        xavier_uniform_(self.conv_user.weight)
        xavier_uniform_(self.conv_item.weight)
  
    def forward(self, b, users_emb, items_emb, e2e_in, e2e_out, p2p_in, p2p_out, e2p_in, e2p_out, iu, iu_p, iu_c, ui, ui_p, ui_c, uu_p, uu_c):
        
        i_emb_ii = self.iiGNN(b, items_emb, e2e_in, e2e_out, p2p_in, p2p_out, e2p_in, e2p_out)
        # u_emb_uu = self.uuGNN(b, users_emb, uu_p, uu_c)
        u_emb_ui = self.uiGNN(b, items_emb, ui_p, ui_c)
        # i_emb_iu = self.iuGNN(b, users_emb, iu_p, iu_c)
        
        # gate_user = self.sigmod(torch.cat((u_emb_ui, u_emb_uu), -1) @ self.W_uu)
        # agg_users_emb = gate_user * u_emb_ui + (1 - gate_user) * u_emb_uu
        
        
        # gate_item =  self.sigmod(torch.cat((i_emb_ii, i_emb_iu), -1) @ self.W_ii)
        # agg_items_emb = gate_item * i_emb_iu + (1 - gate_item) * i_emb_ii
        
        # return agg_users_emb, agg_items_emb
        return u_emb_ui, i_emb_ii
    
    def iiGNN(self, b, items_emb, e2e_in, e2e_out, p2p_in, p2p_out, e2p_in, e2p_out):
        # p_emb = items_emb @ self.W_uu_p
        # e_emb = items_emb @ self.W_uu_c
        # p_emb = items_emb + self.behavior_embedding(torch.LongTensor([2]).to(self.device))
        # e_emb = items_emb + self.behavior_embedding(torch.LongTensor([1]).to(self.device))
        p_emb = items_emb
        e_emb = items_emb
        if b == 2:
           
            p2p_in_neighbor = torch.spmm(p2p_in, p_emb)
            p2p_out_neighbor = torch.spmm(p2p_out, p_emb)
            e2p_in_neighbor = torch.spmm(e2p_in, e_emb)
            
            p2p_in = self.relu(torch.mm(p_emb * p2p_in_neighbor, self.I_p2p_in))
            p2p_out = self.relu(torch.mm(p_emb * p2p_out_neighbor, self.I_p2p_out))
            e2p_in = self.relu(torch.mm(e_emb * e2p_in_neighbor, self.I_e2p_in))
            
            p2p_in_score = torch.squeeze(torch.sum((p2p_in / math.sqrt(self.hidden_dim)), dim=1), 0)
            p2p_out_score = torch.squeeze(torch.sum((p2p_out / math.sqrt(self.hidden_dim)), dim=1), 0)
            e2p_in_score = torch.squeeze(torch.sum((e2p_in / math.sqrt(self.hidden_dim)), dim=1), 0)
            
            score = self.softmax(torch.stack((p2p_in_score, p2p_out_score, e2p_in_score), dim=1))
            score_p2p_in = torch.unsqueeze(score[:, 0], dim=-1)
            score_p2p_out = torch.unsqueeze(score[:, 1], dim=-1)
            score_e2p_in = torch.unsqueeze(score[:, 2], dim=-1)
            neighbor = p2p_in_neighbor * score_p2p_in + p2p_out_neighbor * score_p2p_out + e2p_in_neighbor * score_e2p_in
            
        else:
          
            e2e_in_neighbor = torch.spmm(e2e_in, e_emb)
            e2e_out_neighbor = torch.spmm(e2e_out, e_emb)
            e2p_out_neighbor = torch.spmm(e2p_out, p_emb)
            
            e2e_in = self.relu(torch.mm(e_emb * e2e_in_neighbor, self.I_e2e_in))
            e2e_out = self.relu(torch.mm(e_emb * e2e_out_neighbor, self.I_e2e_out))
            e2p_out = self.relu(torch.mm(p_emb * e2p_out_neighbor, self.I_e2p_out))
            
            e2e_in_score = torch.squeeze(torch.sum((e2e_in / math.sqrt(self.hidden_dim)), dim=1), 0)
            e2e_out_score = torch.squeeze(torch.sum((e2e_out / math.sqrt(self.hidden_dim)), dim=1), 0)
            e2p_out_score = torch.squeeze(torch.sum((e2p_out / math.sqrt(self.hidden_dim)), dim=1), 0)
            
            score = self.softmax(torch.stack((e2e_in_score, e2e_out_score, e2p_out_score), dim=1))
        
            score_e2e_in = torch.unsqueeze(score[:, 0], dim=-1)
            score_e2e_out = torch.unsqueeze(score[:, 1], dim=-1)
            score_e2p_out = torch.unsqueeze(score[:, 2], dim=-1)
            neighbor = e2e_in_neighbor * score_e2e_in + e2e_out_neighbor * score_e2e_out + e2p_out_neighbor * score_e2p_out
            
        agg = torch.stack((items_emb, neighbor), dim=2).unsqueeze(1)
        out_conv = self.conv_item(agg)
        emb = self.feat_drop(torch.squeeze(out_conv))
            
        return neighbor
    
    def uuGNN(self, b, users_emb, uu_p, uu_c):
        # x = uu_c.to_dense()
        # print(torch.mean(users_emb,dim=1))
        # print(torch.std(users_emb, dim=1))
        
        if b == 2:
            neighbor_feature = torch.spmm(uu_p, users_emb)
        else:    
            neighbor_feature = torch.spmm(uu_c, users_emb)
        
        # print(torch.mean(neighbor_feature,dim=1))
        # print(torch.std(neighbor_feature, dim=1))
        agg = torch.stack((neighbor_feature, users_emb), dim=2).unsqueeze(1)
        out_conv = self.conv_user(agg)
        emb = self.feat_drop(torch.squeeze(out_conv))
        return emb
    
    def iuGNN(self, b, users_emb, iu_p, iu_c):
        if b == 2:
            emb = torch.spmm(iu_p, users_emb)
        else:
            emb = torch.spmm(iu_c, users_emb)
        return emb
    
    def uiGNN(self, b, items_emb, ui_p, ui_c):
        
        if b == 2:
            # tmp = torch.Tensor(ui_p.to_dense())
            # for i in range(100, 200):
            #     x = tmp.cpu().detach().numpy()[i]
            #     mask = x > 0
            #     print(x[mask])
            # topk_values, topk_indices = torch.topk(tmp, 10, dim=1)
            # ui_p = torch.zeros_like(tmp)
            # ui_p.scatter_(1, topk_indices, topk_values)
            # idx = torch.nonzero(ui_p).T  
            # data = ui_p[idx[0],idx[1]]
            # ui_p = torch.sparse_coo_tensor(idx, data, ui_p.shape)
            emb = torch.spmm(ui_p, items_emb)
        else:
            # tmp = torch.Tensor(ui_c.to_dense())
            # topk_values, topk_indices = torch.topk(tmp, 10, dim=1)
            # ui_c = torch.zeros_like(tmp)
            # ui_c.scatter_(1, topk_indices, topk_values)
            # idx = torch.nonzero(ui_c).T  
            # data = ui_c[idx[0],idx[1]]
            # ui_c = torch.sparse_coo_tensor(idx, data, ui_c.shape)
            emb = torch.spmm(ui_c, items_emb)
            
        return emb
  
    
class SoftAttnout(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            session_len,
            batch_norm=True,
            feat_drop=0.0,
            activation=None,
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(session_len) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_w1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_w2 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc_w3 = nn.Linear(hidden_dim, 1, bias=False)
        self.fc_out = (
            nn.Linear(input_dim, output_dim, bias=False)
            if output_dim != input_dim else None
        )
        self.mlp_n_ls = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feat, item_seq_len, long_term_representation, mask):
        if self.batch_norm is not None:
            feat = self.batch_norm(feat)
        feat = feat * mask
        feat = self.feat_drop(feat)
        feat_i = self.fc_w1(feat)
        feat_i = feat_i * mask
        feat_u = self.fc_w2(long_term_representation.unsqueeze(1))  # (batch_size * embedding_size)

        long = self.fc_w3(F.tanh(feat_i + feat_u)) * mask
        alph_long = self.sigmoid(long)

        alph_long_score = alph_long.squeeze()
        return alph_long_score
    
class Attout(nn.Module):

    def __init__(
            self, 
            args,
            max_seq_length):
        super().__init__()

        # load parameters info
        self.max_seq_length = max_seq_length
        self.n_layers = args.n_layers
        self.n_heads = args.n_heads
        self.hidden_size = args.hidden_size # same as embedding_size
        self.inner_size = args.hidden_size  # the dimensionality in feed-forward layer
       
        self.emb_drop_out = args.emb_drop_out
        self.att_drop_out = args.att_drop_out
        
        self.hidden_act = args.hidden_act
        self.layer_norm_eps = args.layer_norm_eps

        self.initializer_range = 0.02
        self.W_ui = nn.Parameter(torch.Tensor(2 * self.hidden_size, 1))
        # define layers and loss
        self.sigmod = nn.Sigmoid()
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.emb_drop_out,
            attn_dropout_prob=self.att_drop_out,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.emb_drop_out)
        
        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            xavier_uniform_(module.weight)
            # module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_embs, item_seq, item_seq_len, behavior_emb):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = item_embs
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        
        output1 = trm_output[-1]
        output = self.gather_indexes(output1, item_seq_len - 1)
        
        #TODO: concat  user_emb或者target_behavior_emb
        # c = torch.concat((user_emb, output), 1)
        # c = torch.concat((output, target_behavior_emb), 1)
        # c = torch.concat((output, user_emb), 1)
        # out = self.fc(c)
    
        return output1, output  # [B H]

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0., -10000.)
        return extended_attention_mask
