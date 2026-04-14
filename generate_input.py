import torch
import numpy as np

def get_input(batch, _device, is_train=None):
    if is_train:
        item_seq = list(batch['item_seq'].values())
        uid_seq = list(batch['uid'].values())
        behavior_seq = list(batch['behavior_seq'].values())
        len_seq = list(batch['len_seq'].values())
        target = list(batch['target'].values())
        target_behavior = np.ones_like(len_seq).tolist()
    else:
        item_seq = batch['init_item_seq'].values.tolist()
        behavior_seq = batch['init_behavior_seq'].values.tolist()
        uid_seq = batch['uid'].values.tolist()
        len_seq = batch['len_seq'].values.tolist()
        target = batch['target'].values.tolist()
        target_behavior = (np.ones_like(len_seq)+1).tolist()

    item_seq, uid_seq, behavior_seq, target_behavior, len_seq, target = (torch.LongTensor(item_seq), torch.LongTensor(uid_seq), torch.LongTensor(behavior_seq), torch.LongTensor(target_behavior), torch.LongTensor(len_seq), torch.LongTensor(target))

    item_seq, uid_seq, behavior_seq, target_behavior, len_seq, target = (item_seq.to(_device), uid_seq.to(_device), behavior_seq.to(_device), target_behavior.to(_device), len_seq.to(_device), target.to(_device))
    res = [item_seq, uid_seq, behavior_seq, target_behavior, len_seq, target]
    return res
