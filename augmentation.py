import torch
import random
import math
import numpy as np

def augment(self, item_seq, item_seq_len):
    aug_seq1 = []
    aug_len1 = []
    aug_seq2 = []
    aug_len2 = []
    for seq, length in zip(item_seq, item_seq_len):
        if length > 1:
            switch = random.sample(range(3), k=2)
        else:
            switch = [3, 3]
            aug_seq = seq
            aug_len = length
        if switch[0] == 0:
            aug_seq, aug_len = self.item_crop(seq, length)
        elif switch[0] == 1:
            aug_seq, aug_len = self.item_mask(seq, length)
        elif switch[0] == 2:
            aug_seq, aug_len = self.item_reorder(seq, length)
        
        aug_seq1.append(aug_seq)
        aug_len1.append(aug_len)
        
        if switch[1] == 0:
            aug_seq, aug_len = self.item_crop(seq, length)
        elif switch[1] == 1:
            aug_seq, aug_len = self.item_mask(seq, length)
        elif switch[1] == 2:
            aug_seq, aug_len = self.item_reorder(seq, length)

        aug_seq2.append(aug_seq)
        aug_len2.append(aug_len)
    
    return torch.stack(aug_seq1), torch.stack(aug_len1), torch.stack(aug_seq2), torch.stack(aug_len2)
    
def item_crop(self, item_seq, item_seq_len, eta=0.6):
    num_left = math.floor(item_seq_len * eta)
    crop_begin = random.randint(0, item_seq_len - num_left)
    croped_item_seq = np.zeros(item_seq.shape[0])
    if crop_begin + num_left < item_seq.shape[0]:
        croped_item_seq[:num_left] = item_seq.cpu().detach().numpy()[crop_begin:crop_begin + num_left]
    else:
        croped_item_seq[:num_left] = item_seq.cpu().detach().numpy()[crop_begin:]
    return torch.tensor(croped_item_seq, dtype=torch.long, device=item_seq.device),\
            torch.tensor(num_left, dtype=torch.long, device=item_seq.device)

def item_mask(self, item_seq, item_seq_len, gamma=0.3):
    num_mask = math.floor(item_seq_len * gamma)
    mask_index = random.sample(range(item_seq_len), k=num_mask)
    masked_item_seq = item_seq.cpu().detach().numpy().copy()
    masked_item_seq[mask_index] = self.n_items  # token 0 has been used for semantic masking
    return torch.tensor(masked_item_seq, dtype=torch.long, device=item_seq.device), item_seq_len

def item_reorder(self, item_seq, item_seq_len, beta=0.6):
    num_reorder = math.floor(item_seq_len * beta)
    reorder_begin = random.randint(0, item_seq_len - num_reorder)
    reordered_item_seq = item_seq.cpu().detach().numpy().copy()
    shuffle_index = list(range(reorder_begin, reorder_begin + num_reorder))
    random.shuffle(shuffle_index)
    reordered_item_seq[reorder_begin:reorder_begin + num_reorder] = reordered_item_seq[shuffle_index]
    return torch.tensor(reordered_item_seq, dtype=torch.long, device=item_seq.device), item_seq_len


