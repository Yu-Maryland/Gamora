import argparse

import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d

from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, global_mean_pool, BatchNorm

from dataset_prep import PygNodePropPredDataset, Evaluator, EdgeListDataset
from torch_geometric.loader import NeighborSampler

from logger import Logger
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import time
import copy
from el_sage_baseline import GraphSAGE

os.environ["CUDA_VISIBLE_DEVICES"]=""
#torch.set_num_threads(80)
# num_layers = 6, hidden_channels = 80 for 7nm mapped

class SAGE_MULT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE_MULT, self).__init__()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        # two linear layer for predictions
        self.linear = torch.nn.ModuleList()
        self.linear.append(Linear(hidden_channels, hidden_channels, bias=False))
        self.linear.append(Linear(hidden_channels, out_channels, bias=False))
        self.linear.append(Linear(hidden_channels, out_channels, bias=False))
        self.linear.append(Linear(hidden_channels, out_channels, bias=False))
        
        self.bn0 = BatchNorm1d(hidden_channels)

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for lin in self.linear:
            lin.reset_parameters()

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            
        # print(x[0])
        x = self.linear[0](x)
        x = self.bn0(F.relu(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x1 = self.linear[1](x) # for xor
        x2 = self.linear[2](x) # for maj
        x3 = self.linear[3](x) # for roots
        # print(self.linear[0].weight)
        # print(x1[0])
        return x, x1.log_softmax(dim=-1), x2.log_softmax(dim=-1), x3.log_softmax(dim=-1)
    
    def forward_nosampler(self, x, adj_t, device):
        # tensor placement
        x.to(device)
        adj_t.to(device)
        
        for conv in self.convs:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        # print(x[0])
        x = self.linear[0](x)
        x = self.bn0(F.relu(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x1 = self.linear[1](x) # for xor
        x2 = self.linear[2](x) # for maj
        x3 = self.linear[3](x) # for roots
        # print(self.linear[0].weight)
        # print(x1[0])
        return x1, x2, x3

    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                x = F.relu(x)
                xs.append(x)

                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)
            #print(x_all.size())
            
        x_all = self.linear[0](x_all)
        x_all = F.relu(x_all)
        x_all = self.bn0(x_all)
        x1 = self.linear[1](x_all) # for xor
        x2 = self.linear[2](x_all) # for maj
        x3 = self.linear[3](x_all) # for roots
        pbar.close()

        return x1, x2, x3  
     
def train(model, data_r, data, train_idx, optimizer, train_loader, device):
    pbar = tqdm(total=train_idx.size(0))

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        _, out1, out2, out3 = model(data.x[n_id], adjs)
        
        ### build labels for multitask
        ### original 0: PO, 1: plain, 2: shared, 3: maj, 4: xor, 5: PI
        y1 = data.y.squeeze(1)[n_id[:batch_size]].clone().detach() # make (maj and xor) as xor
        for i in range(y1.size()[-1]):
            if y1[i] == 0 or y1[i] == 5:
                y1[i] = 1
            if y1[i] == 2: 
                y1[i] = 4
            if y1[i] > 2:
                y1[i] = y1[i] - 1 # make to 5 classes
            y1[i] = y1[i] - 1 # 3 classes: 0: plain, 1: maj, 2: xor
                
        y2 = data.y.squeeze(1)[n_id[:batch_size]].clone().detach() # make (maj and xor) as maj
        for i in range(y2.size()[-1]):
            if y2[i] > 2:
                y2[i] = y2[i] - 1 # make to 5 classes
            if y2[i] == 0 or y2[i] == 4:
                y2[i] = 1
            y2[i] = y2[i] - 1 # 3 classes: 0: plain, 1: maj, 2: xor
                
        # for root classification
        # 0: PO, 1: maj, 2: xor, 3: and, 4: PI
        # y3 = data_r.y.squeeze(1)[n_id[:batch_size]]
        y3 = data_r.y.squeeze(1)[n_id[:batch_size]].clone().detach()
        for i in range(y3.size()[-1]):
            if y3[i] == 0 or y3[i] == 4:
                y3[i] = 3
            y3[i] = y3[i] - 1 # 3 classes: 0: maj, 1: xor, 2: and+PI+PO

            
        loss =  F.nll_loss(out1, y1) + F.nll_loss(out2, y2) + 0.8 * F.nll_loss(out3, y3)
        
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out1.argmax(dim=-1).eq(y1).sum())
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / train_idx.size(0)

    return loss, approx_acc

def post_processing(out1, out2):
    pred_1 = out1.argmax(dim=-1, keepdim=True)
    pred_2 = out2.argmax(dim=-1, keepdim=True)
    pred_ecc = (out1 + out2).argmax(dim=-1, keepdim=True)
    # l =  pred_1.size()[0]
    # pred = []
    # for i in range(l):
    #     if pred_1[i] == pred_2[i]:
    #         if pred_1[i] == 0: # PO, and, PI
    #             pred.append(torch.tensor([1]))
    #         else: # maj, xor
    #             pred.append(pred_1[i] + 2) # 3 or 4
    #     else:
    #         if (pred_1[i] == 1 and pred_2[i] == 2) or (pred_1[i] == 2 and pred_2[i] == 1):
    #             pred.append(torch.tensor([2])) # maj and xor
    #         else:
    #             if pred_ecc[i] == 0: # PO, and, PI
    #                 pred.append(torch.tensor([1]))
    #             else: # maj, xor
    #                 pred.append(pred_ecc[i] + 2)
    # pred = torch.tensor(pred)
    
    pred = copy.deepcopy(pred_1)
    
    eq_idx = (torch.eq(pred_1, pred_2) == True).nonzero(as_tuple=True)[0]
    # if pred_1[i] != 0  # maj, xor
    eq_mx_idx = (pred_1[eq_idx] != 0).nonzero(as_tuple=True)[0]
    # pred_1[i] = pred_1[i] + 2  -->  3, 4
    pred[eq_idx[eq_mx_idx]] = pred_1[eq_idx[eq_mx_idx]] + 2
    # if pred_1[i] == 0 PI/PI/and --> final 1
    eq_aig_idx = (pred_1[eq_idx] == 0).nonzero(as_tuple=True)[0]
    pred[eq_idx[eq_aig_idx]] = 1

    neq_idx = (torch.eq(pred_1, pred_2) == False).nonzero(as_tuple=True)[0]
    # if pred_1[i] == 1 and pred_2[i] == 2 shared --> 2
    p1 = (pred_1[neq_idx] == 1).nonzero(as_tuple=True)[0]
    p2 = (pred_2[neq_idx] == 2).nonzero(as_tuple=True)[0]
    shared = p1[(p1.view(1, -1) == p2.view(-1, 1)).any(dim=0)]
    pred[neq_idx[shared]] = 2
    # else (error correction for discrepant predictions)
    if len(p1) != len(p2) or len(p1) != len(neq_idx):
        v, freq = torch.unique(torch.cat((p1, p2), 0), sorted=True, return_inverse=False, return_counts=True, dim=None)
        uniq = (freq == 1).nonzero(as_tuple=True)[0]
        ecc = v[uniq]
        ecc_mx = (pred_ecc[neq_idx][ecc] != 0).nonzero(as_tuple=True)[0]
        ecc_aig = (pred_ecc[neq_idx][ecc] == 0).nonzero(as_tuple=True)[0]
        pred[neq_idx[ecc[ecc_mx]]] = pred_ecc[neq_idx][ecc][ecc_mx] + 2
        pred[neq_idx[ecc[ecc_aig]]] = 1
        zz = (pred == 0).nonzero(as_tuple=True)[0]
        pred[zz] = 1

    return torch.reshape(pred, (pred.shape[0], 1))  
       
@torch.no_grad()
def test_for_elsage(model, data_r, data, subgraph_loader, device):
    model.eval()
    out1, out2, out3 = model.inference(data.x, subgraph_loader, device)
    return out1, out2, out3
    
@torch.no_grad()
def test(model, data_r, data, split_idx, evaluator, subgraph_loader, datatype, device):
    model.eval()

    start_time = time.time()
    out1, out2, out3 = model.inference(data.x, subgraph_loader, device)
    y_pred_shared = post_processing(out1, out2)
    y_pred_root = out3.argmax(dim=-1, keepdim=True)
    # print("print output stats of model.inference", out1.shape, out2.shape)
    print('The inference time is %s' % (time.time() - start_time))
    #y_shared = data.y.squeeze(1).clone().detach()
    #y_root = data_r.y.squeeze(1).clone().detach()
    y_shared = data.y.clone().detach()
    y_root = data_r.y.clone().detach()
    
    
    # for i in range(y_shared.size()[-1]): # 1: and+PI+PO, 2: shared, 3: maj, 4: xor
    #     if y_shared[i] == 0 or y_shared[i] == 5:
    #         y_shared[i] = 1
    # for i in range(y_root.size()[-1]):
    #     if y_root[i] == 0 or y_root[i] == 4:
    #         y_root[i] = 3
    #     y_root[i] = y_root[i] - 1
        
    # 1: and+PI+PO, 2: shared, 3: maj, 4: xor
    s5 = (y_shared == 5).nonzero(as_tuple=True)[0]
    s0 = (y_shared == 0).nonzero(as_tuple=True)[0]
    y_shared[s5] = 1
    y_shared[s0] = 1
    
    r0 = (y_root == 0).nonzero(as_tuple=True)[0]
    r4 = (y_root == 4).nonzero(as_tuple=True)[0]
    y_root[r0] = 3
    y_root[r4] = 3
    y_root = y_root - 1
    
    y_root = torch.reshape(y_root, (y_root.shape[0], 1))
    y_shared = torch.reshape(y_shared, (y_shared.shape[0], 1))  
    
    
    # print(y_pred_root.size())
    # print(y_pred_shared.size())
    
    # print(y_root.size())
    # print(y_shared.size())
    
    if datatype=='train':
        train_acc_r = evaluator.eval({
            'y_true': y_root,#[split_idx['train']],
            'y_pred': y_pred_root,#[split_idx['train']],
        })['acc']
        valid_acc_r = evaluator.eval({
            'y_true': y_root,#[split_idx['valid']],
            'y_pred': y_pred_root,#[split_idx['valid']],
        })['acc']
        test_acc_r = evaluator.eval({
            'y_true': y_root,#[split_idx['test']],
            'y_pred': y_pred_root,#[split_idx['test']],
        })['acc']
        train_acc_s = evaluator.eval({
            'y_true': y_shared,#[split_idx['train']],
            'y_pred': y_pred_shared,#[split_idx['train']],
        })['acc']
        valid_acc_s = evaluator.eval({
            'y_true': y_shared,#[split_idx['valid']],
            'y_pred': y_pred_shared,#[split_idx['valid']],
        })['acc']
        test_acc_s = evaluator.eval({
            'y_true': y_shared,#[split_idx['test']],
            'y_pred': y_pred_shared,#[split_idx['test']],
        })['acc']
        # print("print output label shape", data.y[split_idx['test']].shape)
        return train_acc_r, valid_acc_r, test_acc_r, train_acc_s, valid_acc_s, test_acc_s
    else:
	#we don't have labels for node-classification
	
        test_acc_r = evaluator.eval({
            'y_true': y_root,
            'y_pred': y_pred_root,
        })['acc']
        test_acc_s = evaluator.eval({
            'y_true': y_shared,
            'y_pred': y_pred_shared
        })['acc']

        return 0, 0, test_acc_r, 0, 0, test_acc_s
    
@torch.no_grad()
def test_nosampler(model, data_r, data, split_idx, evaluator, datatype, device):
    model.eval()
    
    start_time = time.time()
    out1, out2, out3 = model.forward_nosampler(data.x, data.adj_t, device)
    y_pred_shared = post_processing(out1, out2)
    y_pred_root = out3.argmax(dim=-1, keepdim=True)
    print('The inference time is %s' % (time.time() - start_time))

    # print("print output stats of model.inference", out1.shape, out2.shape)
    # tensor placement
    y_shared = data.y.squeeze(1).clone().detach().to(device)
    y_root = data_r.y.squeeze(1).clone().detach().to(device)
    
    # for i in range(y_shared.size()[-1]): # 1: and+PI+PO, 2: shared, 3: maj, 4: xor
    #     if y_shared[i] == 0 or y_shared[i] == 5:
    #         y_shared[i] = 1
    # for i in range(y_root.size()[-1]):
    #     if y_root[i] == 0 or y_root[i] == 4:
    #         y_root[i] = 3
    #     y_root[i] = y_root[i] - 1
    
    s5 = (y_shared == 5).nonzero(as_tuple=True)[0]
    s0 = (y_shared == 0).nonzero(as_tuple=True)[0]
    y_shared[s5] = 1
    y_shared[s0] = 1
    
    r0 = (y_root == 0).nonzero(as_tuple=True)[0]
    r4 = (y_root == 4).nonzero(as_tuple=True)[0]
    y_root[r0] = 3
    y_root[r4] = 3
    y_root = y_root - 1
    
    
    y_root = torch.reshape(y_root, (y_root.shape[0], 1))
    y_shared = torch.reshape(y_shared, (y_shared.shape[0], 1))  
    
    # print(y_pred_root.size())
    # print(y_pred_shared.size())
    
    # print(y_root.size())
    # print(y_shared.size())
    
    if datatype=='train':
        train_acc_r = evaluator.eval({
            'y_true': y_root[split_idx['train']],
            'y_pred': y_pred_root[split_idx['train']],
        })['acc']
        valid_acc_r = evaluator.eval({
            'y_true': y_root[split_idx['valid']],
            'y_pred': y_pred_root[split_idx['valid']],
        })['acc']
        test_acc_r = evaluator.eval({
            'y_true': y_root[split_idx['test']],
            'y_pred': y_pred_root[split_idx['test']],
        })['acc']
        train_acc_s = evaluator.eval({
            'y_true': y_shared[split_idx['train']],
            'y_pred': y_pred_shared[split_idx['train']],
        })['acc']
        valid_acc_s = evaluator.eval({
            'y_true': y_shared[split_idx['valid']],
            'y_pred': y_pred_shared[split_idx['valid']],
        })['acc']
        test_acc_s = evaluator.eval({
            'y_true': y_shared[split_idx['test']],
            'y_pred': y_pred_shared[split_idx['test']],
        })['acc']
        # print("print output label shape", data.y[split_idx['test']].shape)
        return train_acc_r, valid_acc_r, test_acc_r, train_acc_s, valid_acc_s, test_acc_s
    else:
        test_acc_r = evaluator.eval({
            'y_true': y_root,
            'y_pred': y_pred_root,
        })['acc']
        test_acc_s = evaluator.eval({
            'y_true': y_shared,
            'y_pred': y_pred_shared
        })['acc']

        return 0, 0, test_acc_r, 0, 0, test_acc_s 

def main():
    parser = argparse.ArgumentParser(description='mult16')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--model_path', type=str, default='SAGE_mult8')
    args = parser.parse_args()
    print(args)

    parser_el = argparse.ArgumentParser(description='ELGraphSAGE Training')
    parser_el.add_argument('--root', type=str, default='/home/curie/ELGraphSAGE/dataset/edgelist', help='Root directory of dataset')
    parser_el.add_argument('--highest_order', type=int, default=16, help='Highest order for the EdgeListDataset')
    parser_el.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser_el.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser_el.add_argument('--num_layers', type=int, default=4)
    parser_el.add_argument('--dropout', type=float, default=0.5)
    parser_el.add_argument('--hidden_dim', type=int, default=75, help='Hidden dimension size')
    parser_el.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser_el.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    args_el = parser_el.parse_args()
    print(args_el)
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    
    ### evaluation dataset loading
    logger_eval_r = Logger(1, args)
    logger_eval = Logger(1, args)
        

    dataset_r = EdgeListDataset(root = '/home/curie/ELGraphSAGE/dataset/edgelist', highest_order = 16)
    data_r = dataset_r[0] #Data(x=[1072, 4], edge_index=[2, 1040], y=[16], adj_t=[1072, 1072, nnz=1040])
    data_r = T.ToSparseTensor()(data_r) #Data(x=[1072, 4], y=[16], adj_t=[1072, 1072, nnz=1040])
    

    dataset = EdgeListDataset(root = '/home/curie/ELGraphSAGE/dataset/edgelist', highest_order = 16)
    data = dataset[0]
    data = T.ToSparseTensor()(data)
    subgraph_loader = NeighborSampler(data.adj_t, node_idx=None, sizes=[-1],
                                  batch_size=1024, shuffle=False,
                                  )
    
    #split_idx = dataset.get_idx_split()
    split_idx = 0
    # tensor placement
    data_r = data_r.to(device)
    data = data.to(device)
    
    gamora_model = SAGE_MULT(data.num_features, args.hidden_channels,
                     3, args.num_layers,
                     args.dropout).to(device)    
    gamora_model.load_state_dict(torch.load(args.model_path))
    

    out1, out2, out3 = test_for_elsage(gamora_model, data_r, data, subgraph_loader, device) #[1072, 3] each
    
    

if __name__ == "__main__":
    main()
