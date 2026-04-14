import torch
import scipy.sparse as sp
from time import time
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import os
import torch.nn as nn

class GraphDataCollector(nn.Module):
    def __init__(self, args, graphData, use_cuda):
        super().__init__()
        self.Graph = None
        self.graph_path = args.graph_path
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        
        self.Graph = None
        # self.data_path = args.data_path
        self.user_num = torch.tensor(args.user_num)
        self.item_num = torch.tensor(args.item_num)
        self.behavior_num = 3
        # self.item2Idx = graphData.item_id
        # self.user2Idx = dataset.field2token_id['item_id']
        self.userSet = graphData['user_id']
        self.item_seq = graphData['item_seq']
        self.behavior_seq = graphData['behavior_seq']

        self.tempUserSet = self.userSet.copy()
        self.tempBehaviorSeq = self.behavior_seq.copy()
        self.userIdx2itemSeq = dict(zip(self.userSet, self.item_seq))
        self.userId2behaviorSeq = dict(zip(self.tempUserSet, self.tempBehaviorSeq))

        # self.userItemSet = dict(zip(self.userSet, set(self.item_id_list)))
        self.itemSeq2user = {}
        self.user_dist = [0 for _ in range(self.user_num)]
        self.item_dist = [0 for _ in range(self.item_num)]
        # self.getSparseGraph()
        

    def getSparseGraph(self):
        print("start load adjacency matrix")
        print("#"*10)
        print("item2item")
        # self.graph_path = self.graph_path
        print("start load adjacency matrix")
        print("#"*10)
        print("item2item")
        # self.graph_path = os.path.join('/home/zgy/xiaoj/SHOCCF-baselines/datasets/Tmall/graph')
        if os.path.exists(self.graph_path + '/'+ 'e2e_in.npz'):
            e2p_in = sp.load_npz(f'{self.graph_path}/e2p_in.npz')
            e2p_out = sp.load_npz(f'{self.graph_path}/e2p_out.npz')
            e2e_in = sp.load_npz(f'{self.graph_path}/e2e_in.npz')
            e2e_out = sp.load_npz(f'{self.graph_path}/e2e_out.npz')
            p2p_in = sp.load_npz(f'{self.graph_path}/p2p_in.npz')
            p2p_out = sp.load_npz(f'{self.graph_path}/p2p_out.npz')
            print("load success")
            print("#"*10)
        else: 
            # TODO: build ItemItemNet
            print("start generating ItemItemNet")
            s_ii = time()
            c = 0
            
            ii_adj_mat = []

            for behavior in range(self.behavior_num):
                item_graph = torch.zeros(self.item_num, self.item_num, dtype=torch.int64)
                ii_adj_mat.append(item_graph)
            e2p_adj_mat = torch.zeros(self.item_num, self.item_num, dtype=torch.int64)
            
           
            for user, item_seq in self.userIdx2itemSeq.items():
                item_seq = torch.tensor(item_seq).long()
                c += 1
                if c % 1000 == 0:
                    print(f'{c} users load, {time()-s_ii}s cost')
                behavior_seq = torch.Tensor(self.userId2behaviorSeq[user])
                mask = item_seq != 0
                item_seq = torch.masked_select(item_seq, mask)
                behavior_seq = torch.masked_select(behavior_seq, mask)

                
                behavior_item_seq = [[0] for _ in range(3)]
                for behavior_type in range(1, self.behavior_num):
                    behavior_item_seq[behavior_type] = item_seq[behavior_seq == behavior_type]
                
               
                for behavior_type in range(1, self.behavior_num):

                    for i in range(len(behavior_item_seq[behavior_type])):
                        if i == 0: 
                            continue
                        now_node = behavior_item_seq[behavior_type][i] 
                        pre_node = behavior_item_seq[behavior_type][i-1]  
                        if now_node != pre_node:
                            ii_adj_mat[behavior_type][pre_node][now_node] += 1  
                        else:
                            pass
                
                #TODO: e2p
                last_purchase = None
                for i, behavior in enumerate(behavior_seq):
                    behavior = behavior.int()
                    if behavior == 2:  # purchase
                        if last_purchase is None:
                            # last_purchase = i
                            for j in range(i):
                                if behavior_seq[j] == 1:
                                    
                                    source_item = item_seq[i]
                                    target_item = item_seq[j]
                                    if source_item != target_item:
                                        e2p_adj_mat[target_item][source_item] += 1
                                    
                            if i+1 < len(behavior_seq) and behavior_seq[i+1] == 1:
                                last_purchase = i
                        else:
                            for j in range(last_purchase + 1, i):
                                if behavior_seq[j] == 1:  # view
                                    source_item = item_seq[i]
                                    target_item = item_seq[j]
                                    if source_item != target_item:
                                        e2p_adj_mat[target_item][source_item] += 1

                            if i+1 < len(behavior_seq) and behavior_seq[i+1] == 1:
                                last_purchase = i         
            
            
            for behavior in range(1, self.behavior_num):
                item_graph = ii_adj_mat[behavior]
                if behavior == 1:
                    coo_e2e_in, coo_e2e_out = self.get_degree_maxtrix(item_graph)
                else:
                    coo_p2p_in, coo_p2p_out = self.get_degree_maxtrix(item_graph)
                
            coo_e2p_in, coo_e2p_out = self.get_degree_maxtrix(e2p_adj_mat)

            sp.save_npz(f'{self.graph_path}/e2p_in.npz', coo_e2p_in)
            sp.save_npz(f'{self.graph_path}/e2p_out.npz', coo_e2p_out)
            sp.save_npz(f'{self.graph_path}/p2p_in.npz', coo_p2p_in)
            sp.save_npz(f'{self.graph_path}/p2p_out.npz', coo_p2p_out)
            sp.save_npz(f'{self.graph_path}/e2e_in.npz', coo_e2e_in)
            sp.save_npz(f'{self.graph_path}/e2e_out.npz', coo_e2e_out)
            print("save success")
            print("#"*10)
        
        # TODO: build UserUserNet
        print("user2user")
        if os.path.exists(self.graph_path + '/'+ 'uu_c.npz'):
            uu_p_10 = sp.load_npz(f'{self.graph_path}/uu_p.npz')
            uu_c_10 = sp.load_npz(f'{self.graph_path}/uu_c.npz')
            print("load success")
            print("#"*10)
        # else:
        #    
        #     print("start generating UserUserNet")
        #     purchase_mat = torch.zeros((self.user_num, self.item_num))
        #     click_mat = torch.zeros((self.user_num, self.item_num))
        #     for user, item_seq in self.userIdx2itemSeq.items():
        #         behavior_seq = torch.Tensor(self.userId2behaviorSeq[user.item()])
        #        
        #         mask = item_seq != 0
        #         item_seq = torch.masked_select(item_seq, mask)
        #         behavior_seq = torch.masked_select(behavior_seq, mask)
                
        #        
        #         purchase_mask = behavior_seq == 2
        #         purchase_seq = torch.masked_select(item_seq, purchase_mask)
        #        
        #         purchase_mat[user][purchase_seq] = 1
                
        #         click_mask = behavior_seq == 1
        #         click_seq = torch.masked_select(item_seq, click_mask)
        #         click_mat[user][click_seq] = 1

        #     uu_p_adj_mat = torch.zeros(self.user_num, self.user_num, dtype=torch.float)
        #     uu_c_adj_mat = torch.zeros(self.user_num, self.user_num, dtype=torch.float)
        #     s_uu = time()
        #  

        #     for i in range(self.user_num):
        #         if (i+1)%1000 == 0:
        #             print(f'{i+1} users load, {time()-s_uu}s cost')
        #         p_Intersection = torch.sum(purchase_mat * purchase_mat[i], 1) 
        #         p_union = torch.sum(purchase_mat + purchase_mat[i], 1) 
        #         # zero = torch.zeros_like(p_Intersection)
        #         # p_Intersection_mask = torch.where(p_Intersection < 10, zero, p_Intersection) 
        #         p_weight = p_Intersection / p_union
        #         p_weight = torch.where(torch.isnan(p_weight), torch.full_like(p_weight, 0), p_weight)
        #         uu_p_adj_mat[:,i] = p_weight

        #     for i in range(self.user_num):
        #         if (i+1)%1000 == 0:
        #             print(f'{i+1} users load, {time()-s_uu}s cost')
        #         c_Intersection = torch.sum(click_mat * click_mat[i], 1)
        #         c_union = torch.sum(click_mat + click_mat[i], 1)
        #         # zero = torch.zeros_like(c_Intersection)
        #         # c_Intersection_mask = torch.where(c_Intersection < 10, zero, c_Intersection)
        #         c_weight = c_Intersection / c_union
        #         c_weight = torch.where(torch.isnan(c_weight), torch.full_like(c_weight, 0), c_weight)
        #         uu_c_adj_mat[:][i] = c_weight

        #    
        #     # diag_p = torch.diag(uu_p_adj_mat)         
        #     # a_diag_p = torch.diag_embed(diag_p)
        #     # uu_p_adj_mat = uu_p_adj_mat - a_diag_p
            
        #     # diag_c = torch.diag(uu_c_adj_mat)         
        #     # a_diag_c = torch.diag_embed(diag_c)
        #     # uu_c_adj_mat = uu_c_adj_mat - a_diag_c
            
        #     # p_row_sum = torch.count_nonzero(uu_p_adj_mat, axis=1).unsqueeze(-1)
        #     p_row_sum = torch.sum(uu_p_adj_mat, axis = 1).unsqueeze(-1)
        #     uu_p = uu_p_adj_mat / p_row_sum
        #     uu_p = torch.where(torch.isnan(uu_p), torch.full_like(uu_p, 0), uu_p)
        
        #     # c_row_sum = torch.count_nonzero(uu_c_adj_mat, axis=1).unsqueeze(-1)
        #     c_row_sum = torch.sum(uu_c_adj_mat, axis = 1).unsqueeze(-1)
        #     uu_c = uu_c_adj_mat / c_row_sum
        #     uu_c = torch.where(torch.isnan(uu_c), torch.full_like(uu_c, 0), uu_c)
            
        #     # coo_uu_p_10 = sp.coo_matrix(uu_p_adj_mat)
        #     # coo_uu_c_10 = sp.coo_matrix(uu_c_adj_mat)
        #     # coo_uu_p_10 = sp.coo_matrix(uu_p)
        #     # coo_uu_c_10 = sp.coo_matrix(uu_c)
            
        #     coo_uu_p = sp.coo_matrix(uu_p)
        #     coo_uu_c = sp.coo_matrix(uu_c)
        #     sp.save_npz(f'{self.graph_path}/uu_p.npz', coo_uu_p)
        #     sp.save_npz(f'{self.graph_path}/uu_c.npz', coo_uu_c)
        #     print("save success")
        #     print("#"*10)
        

        # TODO: build UserItemNet
        print("user2item")
        if os.path.exists(self.graph_path + '/'+ 'iu.npz'):
            iu = sp.load_npz(f'{self.graph_path}/iu.npz')
            iu_p = sp.load_npz(f'{self.graph_path}/iu_p.npz')
            iu_c = sp.load_npz(f'{self.graph_path}/iu_c.npz')
            ui = sp.load_npz(f'{self.graph_path}/ui.npz')
            ui_p = sp.load_npz(f'{self.graph_path}/ui_p.npz')
            ui_c = sp.load_npz(f'{self.graph_path}/ui_c.npz')
            
            print("load success")
            print("#"*10)
        else:
            
            print("start generating UserItemNet")
            s_ui = time()
            users, items = [], []
            p_users, p_items = [], []
            c_users, c_items = [], []
            c = 0
            for user, item_seq in self.userIdx2itemSeq.items():
                item_seq = torch.tensor(item_seq).long()
                behavior_seq = torch.Tensor(self.userId2behaviorSeq[user])
                c += 1
                if c % 1000 == 0:
                    print(f'{c} users load, {time()-s_ui}s cost')
                # mask = torch.eq(item_seq, 0).logical_not_()
                
                seq_len = len(item_seq)
                users += [user] * seq_len
                items += item_seq
                
               
                p_mask =  torch.eq(behavior_seq, 2)
                p_item_seq = torch.masked_select(item_seq, p_mask)
                p_seq_len = len(p_item_seq)
                p_users += [user] * p_seq_len
                p_items += p_item_seq
                
               
                c_mask =  torch.eq(behavior_seq, 1)
                c_item_seq = torch.masked_select(item_seq, c_mask)
                c_seq_len = len(c_item_seq)
                c_users += [user] * c_seq_len
                c_items += c_item_seq
            
            
            
            ui = csr_matrix((np.ones(len(users)), (users, items)), shape=(self.user_num, self.item_num))
            ui_p = csr_matrix((np.ones(len(p_users)), (p_users, p_items)), shape=(self.user_num, self.item_num))
            ui_c = csr_matrix((np.ones(len(c_users)), (c_users, c_items)), shape=(self.user_num, self.item_num))
            
           
            ui_array = torch.Tensor(ui.toarray())
            row_sum = torch.sum(ui_array, axis = 1).unsqueeze(-1)
            ui_array = ui_array / row_sum
            ui_array = torch.where(torch.isnan(ui_array), torch.full_like(ui_array, 0), ui_array)
            iu_array = ui_array.T
            coo_iu = sp.coo_matrix(iu_array)
            coo_ui = sp.coo_matrix(ui_array)

            ui_p_array = torch.Tensor(ui_p.toarray())
            row_sum = torch.sum(ui_p_array, axis = 1).unsqueeze(-1)
            ui_p_array = ui_p_array / row_sum
            ui_p_array = torch.where(torch.isnan(ui_p_array), torch.full_like(ui_p_array, 0), ui_p_array)
            iu_p_array = ui_p_array.T
            coo_iu_p = sp.coo_matrix(iu_p_array)
            coo_ui_p = sp.coo_matrix(ui_p_array)
            
            ui_c_array = torch.Tensor(ui_c.toarray())
            row_sum = torch.sum(ui_c_array, axis = 1).unsqueeze(-1)
            ui_c_array = ui_c_array / row_sum
            ui_c_array = torch.where(torch.isnan(ui_c_array), torch.full_like(ui_c_array, 0), ui_c_array)
            iu_c_array = ui_c_array.T
            coo_iu_c = sp.coo_matrix(iu_c_array)
            coo_ui_c = sp.coo_matrix(ui_c_array)

            sp.save_npz(f'{self.graph_path}/iu.npz', coo_iu)
            sp.save_npz(f'{self.graph_path}/iu_p.npz', coo_iu_p)
            sp.save_npz(f'{self.graph_path}/iu_c.npz', coo_iu_c)
            sp.save_npz(f'{self.graph_path}/ui.npz', coo_ui)
            sp.save_npz(f'{self.graph_path}/ui_p.npz', coo_ui_p)
            sp.save_npz(f'{self.graph_path}/ui_c.npz', coo_ui_c)
            print("save success")
            print("#"*10)
      
 
        e2e_in = self._convert_sp_mat_to_torch_tensor(e2e_in).to(self.device)
        e2e_out = self._convert_sp_mat_to_torch_tensor(e2e_out).to(self.device)
        p2p_in = self._convert_sp_mat_to_torch_tensor(p2p_in).to(self.device)
        p2p_out = self._convert_sp_mat_to_torch_tensor(p2p_out).to(self.device)
        e2p_in = self._convert_sp_mat_to_torch_tensor(e2p_in).to(self.device)
        e2p_out = self._convert_sp_mat_to_torch_tensor(e2p_out).to(self.device)
        iu = self._convert_sp_mat_to_torch_tensor(iu).to(self.device)
        iu_p = self._convert_sp_mat_to_torch_tensor(iu_p).to(self.device)
        iu_c = self._convert_sp_mat_to_torch_tensor(iu_c).to(self.device)
        ui = self._convert_sp_mat_to_torch_tensor(ui).to(self.device)
        ui_p = self._convert_sp_mat_to_torch_tensor(ui_p).to(self.device)
        ui_c = self._convert_sp_mat_to_torch_tensor(ui_c).to(self.device)
        # uu_p_10 = self._convert_sp_mat_to_torch_tensor(uu_p_10).to(self.device)
        # uu_c_10 = self._convert_sp_mat_to_torch_tensor(uu_c_10).to(self.device)
        uu_p_10 = ui_p
        uu_c_10 = ui_c
        
        adj = (e2e_in, e2e_out, p2p_in, p2p_out, e2p_in, e2p_out, iu, iu_p, iu_c, ui, ui_p, ui_c, uu_p_10, uu_c_10)
        # adj = (e2e_in, e2e_out, p2p_in, p2p_out, e2p_in, e2p_out, iu, iu_p, iu_c, ui, ui_p, ui_c)
        return adj
        
   
    def get_degree_maxtrix(self, item_graph):
        '''
        A = [ 1, 2, 2,
            0, 4, 6,
            1, 0, 0 ]

        in = [ 0.5, 0.0, 0.5,
               0.3,  0.7,  0.0,
               1.0,  0.0,  0.0 ]

        out = [ 0.2, 0.4, 0.4,
                0.0  0.4  0.6,
                1.0  0.0  0.0 ]

        '''
       
        row_sum = torch.sum(item_graph, axis = 0).unsqueeze(0) 
      
        col_sum = torch.sum(item_graph, axis = 1).unsqueeze(1) 
        ii_in = item_graph / row_sum 
        ii_out = item_graph / col_sum 
        ii_in = torch.where(torch.isnan(ii_in), torch.full_like(ii_in, 0), ii_in)
        ii_out = torch.where(torch.isnan(ii_out), torch.full_like(ii_out, 0), ii_out)
        ii_in = ii_in.T 
        
        coo_ii_in = sp.coo_matrix(ii_in)
        coo_ii_out = sp.coo_matrix(ii_out)
        return coo_ii_in, coo_ii_out
    
    def _convert_sp_mat_to_torch_tensor(self, X):
        coo = X
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def _convert_dense_mat_to_torch_tensor(self, X):
        coo = sp.coo_matrix(X)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
    
