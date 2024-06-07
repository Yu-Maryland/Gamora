import torch
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor
import torch.nn.functional as F
import numpy as np
from scipy.sparse import identity, diags


def graph2adj(adj):
    #hat_adj = adj + identity(adj.shape[0])
    hat_adj = adj
    degree_vec = hat_adj.sum(axis=1)
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.squeeze(np.asarray(np.power(degree_vec, -0.5)))
    d_inv_sqrt[np.isinf(d_inv_sqrt)|np.isnan(d_inv_sqrt)] = 0
    degree_matrix  = diags(d_inv_sqrt, 0)
    DAD = degree_matrix @ (hat_adj @ degree_matrix)
    AD = hat_adj @ (degree_matrix @ degree_matrix)
    DA = degree_matrix @ (degree_matrix @ hat_adj)

    return DAD, AD, DA

def preprocess(data, args):
    print("Preprocessing node features!!!!!!")
    nnodes = data.x.shape[0]
    if not args.directed:
        data.edge_index = to_undirected(data.edge_index, nnodes)
        row, col = data.edge_index
        adj = SparseTensor(row=row, col=col, sparse_sizes=(nnodes, nnodes))
        adj = adj.to_scipy(layout='csr')
        DAD, AD, DA = graph2adj(adj)
        norm_adj = SparseTensor.from_scipy(DAD).float()
        feat_lst = []
        feat_lst.append(data.x)
        high_order_features = data.x.clone()
        for _ in range(args.num_hops):
            high_order_features = norm_adj @ high_order_features
            #data.x = torch.cat((data.x, high_order_features), dim=1)
            feat_lst.append(high_order_features)
        data.x = torch.stack(feat_lst, dim=1)
        #data.num_features *= (1+args.num_hops)
    else:
        row, col = data.edge_index
        adj = SparseTensor(row=row, col=col, sparse_sizes=(nnodes, nnodes))
        adj = adj.to_scipy(layout='csr')
        _, _, DA = graph2adj(adj)
        _, _, DA_tran = graph2adj(adj.transpose())
        norm_adj = SparseTensor.from_scipy(DA).float()
        norm_adj_tran = SparseTensor.from_scipy(DA_tran).float()
        feat_lst = []
        feat_lst.append(data.x)
        high_order_features = data.x.clone()
        high_order_features_tran = data.x.clone()
        for _ in range(args.num_hops):
            high_order_features = norm_adj @ high_order_features
            high_order_features_tran = norm_adj @ high_order_features_tran
            #data.x = torch.cat((data.x, high_order_features, high_order_features_tran), dim=1)
            feat_lst.append(high_order_features)
            feat_lst.append(high_order_features_tran)
        data.x = torch.stack(feat_lst, dim=1)
        #data.num_features *= (1+2*args.num_hops)

    return data

def all_numpy(obj):
    # Ensure everything is in numpy or int or float (no torch tensor)

    if isinstance(obj, dict):
        for key in obj.keys():
            all_numpy(obj[key])
    elif isinstance(obj, list):
        for i in range(len(obj)):
            all_numpy(obj[i])
    else:
        if not isinstance(obj, (np.ndarray, int, float)):
            return False

    return True

def train(model, train_loader, optimizer, device, args):
    model.train()
    total_loss = 0
    for i, (x, y, r_y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        r_y = r_y.to(device)

        optimizer.zero_grad()
        out1, out2, out3, attn = model(x)

        ### build labels for multitask
        ### original 0: PO, 1: plain, 2: shared, 3: maj, 4: xor, 5: PI
        y1 = y.squeeze(1).clone().detach() # make (maj and xor) as xor
        for i in range(y1.size()[-1]):
            if y1[i] == 0 or y1[i] == 5:
                y1[i] = 1
            if y1[i] == 2:
                y1[i] = 4
            if y1[i] > 2:
                y1[i] = y1[i] - 1 # make to 5 classes
            y1[i] = y1[i] - 1 # 3 classes: 0: plain, 1: maj, 2: xor

        y2 = y.squeeze(1).clone().detach() # make (maj and xor) as maj
        for i in range(y2.size()[-1]):
            if y2[i] > 2:
                y2[i] = y2[i] - 1 # make to 5 classes
            if y2[i] == 0 or y2[i] == 4:
                y2[i] = 1
            y2[i] = y2[i] - 1 # 3 classes: 0: plain, 1: maj, 2: xor

        # for root classification
        # 0: PO, 1: maj, 2: xor, 3: and, 4: PI
        # y3 = data_r.y.squeeze(1)[n_id[:batch_size]]
        y3 = r_y.squeeze(1).clone().detach()
        for i in range(y3.size()[-1]):
            if y3[i] == 0 or y3[i] == 4:
                y3[i] = 3
            y3[i] = y3[i] - 1 # 3 classes: 0: maj, 1: xor, 2: and+PI+PO


        loss = F.cross_entropy(out1, y1) + args.lda1*F.cross_entropy(out2, y2) + args.lda2*F.cross_entropy(out3, y3)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    loss = total_loss / len(train_loader)

    return loss

def post_processing(out1, out2):
    pred_1 = out1.argmax(dim=-1, keepdim=True)
    pred_2 = out2.argmax(dim=-1, keepdim=True)
    pred_ecc = (out1 + out2).argmax(dim=-1, keepdim=True)

    l =  pred_1.size()[0]
    pred = []
    for i in range(l):
        if pred_1[i] == pred_2[i]:
            if pred_1[i] == 0: # PO, and, PI
                pred.append(torch.tensor([1]))
            else: # maj, xor
                pred.append(pred_1[i] + 2) # 3 or 4
        else:
            if (pred_1[i] == 1 and pred_2[i] == 2) or (pred_1[i] == 2 and pred_2[i] == 1):
                pred.append(torch.tensor([2])) # maj and xor
            else:
                if pred_ecc[i] == 0: # PO, and, PI
                    pred.append(torch.tensor([1]))
                else: # maj, xor
                    pred.append(pred_ecc[i] + 2)
    pred = torch.tensor(pred)
    '''
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
    p1 = (pred_1[neq_idx] == 2).nonzero(as_tuple=True)[0]
    p2 = (pred_2[neq_idx] == 1).nonzero(as_tuple=True)[0]
    shared = p1[(p1.view(1, -1) == p2.view(-1, 1)).any(dim=0)]
    pred[neq_idx[shared]] = 2

    p1 = (pred_1[neq_idx] == 1).nonzero(as_tuple=True)[0]
    p2 = (pred_2[neq_idx] == 2).nonzero(as_tuple=True)[0]
    shared = p1[(p1.view(1, -1) == p2.view(-1, 1)).any(dim=0)]
    pred[neq_idx[shared]] = 2

    # else (error correction for discrepant predictions)
    if len(p1) != len(p2) or len(p1) != len(neq_idx):
        print("start error correction!!!!!!")
        v, freq = torch.unique(torch.cat((p1, p2), 0), sorted=True, return_inverse=False, return_counts=True, dim=None)
        uniq = (freq == 1).nonzero(as_tuple=True)[0]
        ecc = v[uniq]
        ecc_mx = (pred_ecc[neq_idx][ecc] != 0).nonzero(as_tuple=True)[0]
        ecc_aig = (pred_ecc[neq_idx][ecc] == 0).nonzero(as_tuple=True)[0]
        pred[neq_idx[ecc[ecc_mx]]] = pred_ecc[neq_idx][ecc][ecc_mx] + 2
        pred[neq_idx[ecc[ecc_aig]]] = 1
        zz = (pred == 0).nonzero(as_tuple=True)[0]
        pred[zz] = 1
    '''

    return torch.reshape(pred, (pred.shape[0], 1))


@torch.no_grad()
def test(model, train_loader, valid_loader, test_loader, device):
    model.eval()

    train_acc_r = 0
    train_acc_s = 0
    valid_acc_r = 0
    valid_acc_s = 0
    test_acc_r = 0
    test_acc_s = 0
    for i, (x, y, r_y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        r_y = r_y.to(device)

        out1, out2, out3, train_attn = model(x)
        y_pred_shared = post_processing(out1, out2).to(device)
        y_pred_root = out3.argmax(dim=-1, keepdim=True)

        y_shared = y.squeeze(1).clone().detach()
        y_root = r_y.squeeze(1).clone().detach()
        # 1: and+PI+PO, 2: shared, 3: maj, 4: xor
        s5 = (y_shared == 5).nonzero(as_tuple=True)[0]
        s0 = (y_shared == 0).nonzero(as_tuple=True)[0]
        y_shared[s5] = 1
        y_shared[s0] = 1
        # 0: maj, 1: xor, 2: and+PI+PO
        r0 = (y_root == 0).nonzero(as_tuple=True)[0]
        r4 = (y_root == 4).nonzero(as_tuple=True)[0]
        y_root[r0] = 3
        y_root[r4] = 3
        y_root = y_root - 1
        y_root = torch.reshape(y_root, (y_root.shape[0], 1))
        y_shared = torch.reshape(y_shared, (y_shared.shape[0], 1))

        train_acc_r += y_pred_root.eq(y_root).double().sum()
        train_acc_s += y_pred_shared.eq(y_shared).double().sum()

    train_acc_r /= len(train_loader.dataset)
    train_acc_s /= len(train_loader.dataset)

    for i, (x, y, r_y) in enumerate(valid_loader):
        x = x.to(device)
        y = y.to(device)
        r_y = r_y.to(device)

        out1, out2, out3, valid_attn = model(x)
        y_pred_shared = post_processing(out1, out2).to(device)
        y_pred_root = out3.argmax(dim=-1, keepdim=True)

        y_shared = y.squeeze(1).clone().detach()
        y_root = r_y.squeeze(1).clone().detach()
        # 1: and+PI+PO, 2: shared, 3: maj, 4: xor
        s5 = (y_shared == 5).nonzero(as_tuple=True)[0]
        s0 = (y_shared == 0).nonzero(as_tuple=True)[0]
        y_shared[s5] = 1
        y_shared[s0] = 1
        # 0: maj, 1: xor, 2: and+PI+PO
        r0 = (y_root == 0).nonzero(as_tuple=True)[0]
        r4 = (y_root == 4).nonzero(as_tuple=True)[0]
        y_root[r0] = 3
        y_root[r4] = 3
        y_root = y_root - 1
        y_root = torch.reshape(y_root, (y_root.shape[0], 1))
        y_shared = torch.reshape(y_shared, (y_shared.shape[0], 1))

        valid_acc_r += y_pred_root.eq(y_root).double().sum()
        valid_acc_s += y_pred_shared.eq(y_shared).double().sum()

    valid_acc_r /= len(valid_loader.dataset)
    valid_acc_s /= len(valid_loader.dataset)

    for i, (x, y, r_y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        r_y = r_y.to(device)

        out1, out2, out3, test_attn = model(x)
        y_pred_shared = post_processing(out1, out2).to(device)
        y_pred_root = out3.argmax(dim=-1, keepdim=True)

        y_shared = y.squeeze(1).clone().detach()
        y_root = r_y.squeeze(1).clone().detach()
        # 1: and+PI+PO, 2: shared, 3: maj, 4: xor
        s5 = (y_shared == 5).nonzero(as_tuple=True)[0]
        s0 = (y_shared == 0).nonzero(as_tuple=True)[0]
        y_shared[s5] = 1
        y_shared[s0] = 1
        # 0: maj, 1: xor, 2: and+PI+PO
        r0 = (y_root == 0).nonzero(as_tuple=True)[0]
        r4 = (y_root == 4).nonzero(as_tuple=True)[0]
        y_root[r0] = 3
        y_root[r4] = 3
        y_root = y_root - 1
        y_root = torch.reshape(y_root, (y_root.shape[0], 1))
        y_shared = torch.reshape(y_shared, (y_shared.shape[0], 1))

        test_acc_r += y_pred_root.eq(y_root).double().sum()
        test_acc_s += y_pred_shared.eq(y_shared).double().sum()

    test_acc_r /= len(test_loader.dataset)
    test_acc_s /= len(test_loader.dataset)

    return train_acc_r, valid_acc_r, test_acc_r, train_acc_s, valid_acc_s, test_acc_s

@torch.no_grad()
def test_all(model, test_loader, device, file_name=None):
    model.eval()

    test_acc_r = 0
    test_acc_s = 0
    all_y_shared = []
    all_pred_shared = []
    all_test_attn = []
    for i, (x, y, r_y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        r_y = r_y.to(device)

        out1, out2, out3, test_attn = model(x)
        y_pred_shared = post_processing(out1, out2).to(device)
        y_pred_root = out3.argmax(dim=-1, keepdim=True)

        y_shared = y.squeeze(1).clone().detach()
        y_root = r_y.squeeze(1).clone().detach()
        # 1: and+PI+PO, 2: shared, 3: maj, 4: xor
        s5 = (y_shared == 5).nonzero(as_tuple=True)[0]
        s0 = (y_shared == 0).nonzero(as_tuple=True)[0]
        y_shared[s5] = 1
        y_shared[s0] = 1
        # 0: maj, 1: xor, 2: and+PI+PO
        r0 = (y_root == 0).nonzero(as_tuple=True)[0]
        r4 = (y_root == 4).nonzero(as_tuple=True)[0]
        y_root[r0] = 3
        y_root[r4] = 3
        y_root = y_root - 1
        y_root = torch.reshape(y_root, (y_root.shape[0], 1))
        y_shared = torch.reshape(y_shared, (y_shared.shape[0], 1))

        test_acc_r += y_pred_root.eq(y_root).double().sum()
        test_acc_s += y_pred_shared.eq(y_shared).double().sum()

        all_y_shared.append(y_shared.detach().cpu().squeeze().numpy())
        all_pred_shared.append(y_pred_shared.detach().cpu().squeeze().numpy())
        all_test_attn.append(test_attn.detach().cpu().squeeze().numpy())

    test_acc_r /= len(test_loader.dataset)
    test_acc_s /= len(test_loader.dataset)

    return 0, 0, test_acc_r, 0, 0, test_acc_s

