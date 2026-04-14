import argparse
import torch
import datetime
import numpy as np
import pandas as pd
import os
from dagraph import DAGraph
from GraphDataGenerator import GraphDataCollector
from generate_input import get_input
from utility import calculate_hit_ndcg
from augmentation import *
def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
# data arguments
parser.add_argument('--data', nargs='?', default='/datasets/JD/data/new_data',
                    help='data directory')
parser.add_argument('--graph_path', nargs='?', default='/datasets/JD/graph',
                    help='graph directory')
parser.add_argument('--dataset', nargs='?', default='JD')
# train arguments
parser.add_argument('--epoch', type=int, default=10000,
                    help='Number of max epochs.')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size.')
parser.add_argument('--embedding_size', type=int, default=64,
                    help='Number of hidden factors, i.e., embedding size.')
parser.add_argument('--hidden_size', type=int, default=64,
                    help='Number of hidden factors, i.e., embedding size.')
parser.add_argument('--n_layers', default=3, type=int)
parser.add_argument('--n_heads', default=1, type=int)
parser.add_argument('--num_layer', default=1, type=int)
parser.add_argument('--mlp_layers', default=3, type=int)
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--dropout_prob', default=0.5, type=float)
parser.add_argument('--att_drop_out', default=0.5, type=float)
parser.add_argument('--emb_drop_out', default=0.5, type=float)
parser.add_argument('--cl_tau', default=1, type=float)
parser.add_argument('--lmd_short', default=0.0, type=float)
parser.add_argument('--lmd_long', default=0.0, type=float)
parser.add_argument('--sim', default='dot')
parser.add_argument('--hidden_act', default='gelu')
parser.add_argument('--layer_norm_eps', default=1e-12, type=float)
parser.add_argument('--early_stop_epoch', default=20, type=int)
parser.add_argument('--alpha', default=0.8, type=float)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--is_test', type=str2bool, default=True)
parser.add_argument('--type', type=str, default='all')

args = parser.parse_args()

args.cuda = torch.cuda.is_available()
# use random seed defined
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    
def augment(args, seqs, behaviors, lengths):
    aug_seq1 = []
    aug_behavior1 = []
    aug_len1 = []
    
    for item_seq, behavior_seq, length in zip(seqs, behaviors, lengths):
            # TODO: divide sequences into sub-sequences according to purchase behavior
        new_item_seq, new_behavior_seq = item_seq[:length], behavior_seq[:length]
        mask = (new_behavior_seq[:-1] == 2) & (new_behavior_seq[1:] == 1)
        split_indices = np.where(mask.cpu())[0] + 1
        split_indices = np.insert(split_indices, 0, 0)
        split_indices = np.append(split_indices, len(new_item_seq))
        item_sequences = [new_item_seq[start:end] for start, end in zip(split_indices[:-1], split_indices[1:])]
        behavior_sequences = [new_behavior_seq[start:end] for start, end in zip(split_indices[:-1], split_indices[1:])]
    
        aug_seq, aug_behavior, aug_len = sub_sequence_reorder(item_sequences, behavior_sequences, length, args)


        aug_seq1.append(aug_seq)
        aug_behavior1.append(aug_behavior)
        aug_len1.append(aug_len)
        
    return torch.stack(aug_seq1), torch.stack(aug_behavior1), torch.stack(aug_len1)

def cal_prob(length, a=0.8, args=None):
    if isinstance(length, int):
        length = torch.tensor(length).to(args.device)
    item_indices = torch.arange(length, dtype=torch.float32, device=args.device)  # create indexes ranging from 0 to n-1
    item_importance = torch.pow(a, length - item_indices)
    total = torch.sum(item_importance)
    prob = item_importance / total
    return prob
    
def sub_sequence_reorder(seq, behavior, length, args):
    seq_ = seq.copy()
    behavior_ = behavior.copy()
    index = torch.arange(len(seq_), dtype=torch.float32, device=args.device)
    index1_prob = cal_prob(len(seq_), args=args).flip(0)  # the probability of subsequences that are sampling
    selected_item1_index = torch.tensor([random.sample(range(len(seq_)), k=1)[0]], device=args.device)
    # selected_item1_index = torch.multinomial(index1_prob, num_samples=1)
    item_importance = torch.pow(args.alpha, abs(index - selected_item1_index))
    total = torch.sum(item_importance)
    prob = item_importance / total
    selected_item2_index = torch.multinomial(prob, num_samples=1)

    seq_[selected_item1_index], seq_[selected_item2_index] = seq_[selected_item2_index], seq_[selected_item1_index]
    behavior_[selected_item1_index], behavior_[selected_item2_index] = behavior_[selected_item2_index], behavior_[selected_item1_index]
    reorder_sub_seq = torch.cat(seq_)
    reorder_sub_behaviro = torch.cat(behavior_)
    padded_reorder_item_seq = torch.cat((reorder_sub_seq, torch.zeros(args.max_seq_length - len(reorder_sub_seq), dtype=torch.long, device=args.device)))
    padded_reorder_behavior_seq = torch.cat((reorder_sub_behaviro, torch.zeros(args.max_seq_length - len(reorder_sub_behaviro), dtype=torch.long, device=args.device)))

    return  padded_reorder_item_seq, padded_reorder_behavior_seq, length
    

def calculate_loss(args, model, inputs, graph, seq_representation, short, target_item_emb, target):

    item_seq, user, behavior_seq, target_behavior, item_seq_len, target_item = inputs
    
    target_item = target.unsqueeze(1)
    target_item = target_item.squeeze()
    
    all_items_emb = model.item_embedding.weight[:model.item_num]
    
    scores = torch.matmul(seq_representation, all_items_emb.transpose(0,1))  
    # scores = model.fc_score(seq_representation)
    
    generated_seq_loss = model.loss_fuc(scores, target_item)
    
    if model.lmd_short == 0 and model.lmd_long == 0:
        return generated_seq_loss
    
    if model.lmd_short != 0:
        
        short_nce_logits, short_nce_labels = model.info_nce(short, target_item_emb, temp=model.cl_tau,
                                                batch_size=seq_representation.shape[0], sim=model.sim)
        
        short_nce_loss = model.aug_nce_fct(short_nce_logits, short_nce_labels)
        
    
    if model.lmd_long != 0:
    
        #TODO: aug --------------------------------------------------------------------------------------------------------------
        aug_item_seq1, aug_behavior_seq1, aug_len1 = augment(args, item_seq, behavior_seq, item_seq_len)
        
        aug_inputs = aug_item_seq1, user, aug_behavior_seq1, target_behavior, aug_len1, target_item
        seq_output1, _, _ = model(aug_inputs, graph, train_flag=True)
        
        long_nce_logits, long_nce_labels = model.info_nce(seq_representation, seq_output1, temp=model.cl_tau, batch_size=aug_len1.shape[0], sim=model.sim)
        long_nce_loss = model.aug_nce_fct(long_nce_logits, long_nce_labels)
    
    if model.lmd_short != 0 and model.lmd_long != 0:
        cl_loss = model.lmd_long * long_nce_loss + model.lmd_short * short_nce_loss
    
    elif model.lmd_short == 0 and model.lmd_long != 0:
        cl_loss = model.lmd_long * long_nce_loss
    else:
        cl_loss = model.lmd_short * short_nce_loss
        
    # # cl_loss = model.lmd_short * short_nce_loss
    # # cl_loss = model.lmd_long * long_nce_loss
    # total_loss = generated_seq_loss
    total_loss = generated_seq_loss + cl_loss
    # return total_loss, generated_seq_loss, loss_filter_loss
    return total_loss

def predict(model, topk, batch_size, graph, _device):
    start_time = datetime.datetime.now()  
    hit_purchase = [0, 0, 0]
    ndcg_purchase = [0, 0, 0]
    user_num = eval_data.shape[0]
    print("eval user_num: ", user_num)
    for start in range(0, user_num, batch_size):
        end = start + batch_size if start + batch_size < user_num else user_num
        if start % 5000 == 0:
            print("test progress:{}/{}".format(start, user_num))
        batch = eval_data.iloc[start:end]
        batch_size_ = end - start
        eval_inputs = get_input(batch, _device, is_train=False)
        target = eval_inputs[-1]
        seq_output, _, _  = model(eval_inputs, graph, train_flag=False)
        test_items_emb = model.item_embedding.weight[:model.item_num]
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        # scores = model.fc_score(seq_output)
        # prediction = scores.data.cpu().numpy()
        
        prediction = torch.argsort(scores)

        calculate_hit_ndcg(prediction, topk, target, hit_purchase, ndcg_purchase)
        
    for i in range(len(topk)):
        hit = hit_purchase[i] / user_num
        ndcg = ndcg_purchase[i] / user_num
        print("k | hit@k | ndcg@k")
        print("%d %.8f %.8f " % (topk[i], hit, ndcg))
    print('#############################################################')
    over_time = datetime.datetime.now()  
    total_time = (over_time - start_time).total_seconds()
    print('total times: %s' % total_time)
    return hit_purchase[0] / user_num, ndcg_purchase[0] / user_num, \
           hit_purchase[1] / user_num, ndcg_purchase[1] / user_num, \
           hit_purchase[2] / user_num, ndcg_purchase[2] / user_num,


if __name__ == '__main__':

    data_folder = args.data
    # statis data
    statis_data = pd.read_pickle(os.path.join(data_folder, 'data_statis.df'))  # includeing seq_len and item_num
    # train data
    train_data =  pd.read_pickle(os.path.join(data_folder, 'train.df'))
    # eval data
    is_test = args.is_test
    type = args.type
    if is_test == False:
        eval_data = pd.read_pickle(os.path.join(data_folder, 'val.df'))
    else:
        if type == 'all':
            eval_data = pd.read_pickle(os.path.join(data_folder, 'test.df'))
        elif type == 'clicked':
            eval_data = pd.read_pickle(os.path.join(data_folder, 'test_click.df'))
        elif type == 'unclicked':
            eval_data = pd.read_pickle(os.path.join(data_folder, 'test_unclick.df'))

    seq_len = statis_data['state_size'][0]  # the length of history to define the state
    item_num = statis_data['item_num'][0]+1  # total number of item
    user_num = statis_data['user_num'][0]+1  # total number of user
    args.item_num = item_num
    args.user_num = user_num
    print("user_num: ", user_num)
    print("item_num: ", item_num)
    topk = [5, 10, 20]

    batch_size = args.batch_size
    epoch = args.epoch
    emb_size = args.embedding_size
    dropout_rate = args.dropout_prob
    lr = args.lr
    early_stop_epoch = args.early_stop_epoch
    cuda = args.cuda
    args.max_seq_length = seq_len
    _device = torch.device('cuda' if args.cuda else 'cpu')
    args.device = _device
    generate_graph_data = pd.read_pickle('/datasets/JD/data/new_data/generate_graph/jd_train.df')
    graph_item_seq = list(generate_graph_data['item_id_list:token_seq'])
    graph_behavior_seq = list(generate_graph_data['behavior_type_list:token_seq'])
    graph_user_set = list(generate_graph_data['user_id:token'])
    graphData = {
            "item_seq": graph_item_seq,
            "behavior_seq": graph_behavior_seq,
            "user_id": graph_user_set
        }
    graph = GraphDataCollector(args=args, graphData=graphData, use_cuda=cuda).getSparseGraph()
    model = DAGraph(args=args,
                 item_num=item_num,
                 user_num=user_num,
                 seq_len=seq_len,
                 use_cuda=cuda
                 ).to(_device)

    now_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = 'model/DAGraph/new_graph/{}/alpha_{}_s_{}_l_{}_nl_{}_nh_{}_{}'.format(args.dataset, args.alpha, args.lmd_short, args.lmd_long, args.n_layers, args.n_heads, now_time)
    isExists = os.path.exists(save_dir)
    if not isExists:
        os.makedirs(save_dir)

    _loss = torch.nn.CrossEntropyLoss()
    _optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # print(model.parameters)
    # for name, parameters in model.named_parameters():
    #     print(name, ':', parameters.size())
    print("data number of click :{} , data number of purchase :{}".format(
        train_data[train_data['is_buy'] == 1].shape[0],
        train_data[train_data['is_buy'] == 2].shape[0],
    ))

    num_rows = train_data.shape[0]
    minibatch = int(num_rows / batch_size)
    total_step = 0
    best_hit_10 = -1

    # checkpoint = torch.load("/SHOCCF-baselines/MBASR/model/DAGraph/new_graph/Tmall/alpha_0.8_s_0.0_l_0.005_nl_3_nh_1_20240619_210827/epoch_5_hit@5_0.0736_ndcg@5_0.0506_hit@10_0.0973_ndcg@10_0.0583_hit@20_0.1223_ndcg@20_0.0646/DAGraph.ckpt")
    # model.load_state_dict(checkpoint)
    # _optimizer.load_state_dict(checkpoint['optimizer'])
    # start_epoch = checkpoint['epoch'] + 1
    
    for epoch_num in range(epoch):

        # TODO: Training
        model.train()
        epoch_loss = 0.0
        print("Epoch: ", epoch_num + 1)
        print("==========================================================")
        start_time = datetime.datetime.now()  

        for j in range(minibatch):
            batch = train_data.sample(n=batch_size).to_dict()
            inputs = get_input(batch, _device, is_train=True)
            target = inputs[-1]
            _optimizer.zero_grad()
            
            item_seq, user, behavior_seq, target_behavior, item_seq_len, target_item = inputs
            
            seq_representation, short, target_item_emb = model(inputs, graph, train_flag=True)
            
            loss = calculate_loss(args, model, inputs, graph, seq_representation, short, target_item_emb, target)
            epoch_loss += loss
            loss.backward()
            _optimizer.step()
            total_step += 1
            if total_step % 1000 == 0:
                print("the loss in %dth batch is: %f" % (total_step, loss.item()))

        epoch_loss /= minibatch + 1
        over_time_i = datetime.datetime.now() 
        total_time_i = (over_time_i - start_time).total_seconds()
        print('total times: %s' % total_time_i)
        print('Epoch', epoch_num + 1, 'loss: ', epoch_loss.item())

        # TODO: Evaluate
        model.eval()
        hit5, ndcg5, hit10, ndcg10, hit20, ndcg20 = predict(model, topk, batch_size, graph, _device)
        if hit10 > best_hit_10:
            best_hit_10 = hit10
            count = 0
            save_root = os.path.join(save_dir,
                                     'epoch_{}_hit@5_{:.4f}_ndcg@5_{:.4f}_hit@10_{:.4f}_ndcg@10_{:.4f}_hit@20_{:.4f}_ndcg@20_{:.4f}'.format(
                                         epoch_num, 
                                         hit5, ndcg5,
                                         hit10, ndcg10,
                                         hit20, ndcg20
                                         ))
            isExists = os.path.exists(save_root)
            if not isExists:
                os.makedirs(save_root)
            model_name = 'DAGraph.ckpt'
            save_root = os.path.join(save_root, model_name)
            
            torch.save(model.state_dict(), save_root)

        else:
            count += 1
        if count == args.early_stop_epoch:
            break



