import argparse

import torch
import torch.nn.functional as F

from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from dataset_prep import PygNodePropPredDataset, Evaluator, GenMultDataset,ABCGenDataset
from torch_geometric.loader import NeighborSampler

from logger import Logger
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x, x.log_softmax(dim=-1)

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
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


def train(model, data, train_idx, optimizer, train_loader, device):
    pbar = tqdm(total=train_idx.size(0))

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        _, out = model(data.x[n_id], adjs)
        loss = F.nll_loss(out, data.y.squeeze(1)[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(data.y.squeeze(1)[n_id[:batch_size]]).sum())
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / train_idx.size(0)

    return loss, approx_acc


@torch.no_grad()
def test(model, data, split_idx, evaluator, subgraph_loader, datatype, device):
    model.eval()

    out = model.inference(data.x, subgraph_loader, device)
    print("print output stats of model.inference", out.shape)
    y_pred = out.argmax(dim=-1, keepdim=True)
    if datatype=='train':
        train_acc = evaluator.eval({
            'y_true': data.y[split_idx['train']],
            'y_pred': y_pred[split_idx['train']],
        })['acc']
        valid_acc = evaluator.eval({
            'y_true': data.y[split_idx['valid']],
            'y_pred': y_pred[split_idx['valid']],
        })['acc']
        test_acc = evaluator.eval({
            'y_true': data.y[split_idx['test']],
            'y_pred': y_pred[split_idx['test']],
        })['acc']
        print(data.y[split_idx['test']].shape)
        return train_acc, valid_acc, test_acc
    else:
        test_acc = evaluator.eval({
            'y_true': data.y,
            'y_pred': y_pred,
        })['acc']

        return 0, 0, test_acc


def dataloader_prep(bits, datagen, root_folder, designfile, device, num_class = 5, multilabel=False):
    dataset_generator = ABCGenDataset(bits, datagen, root_folder, designfile, multilabel)
    dataset_generator.generate(dataset_path=root_folder+"/dataset",train_split=0.8, val_split=1)
    dataset = PygNodePropPredDataset(name=dataset_generator.design_name, num_class=num_class)
    print("Training on %s" % dataset_generator.design_name)
    data = dataset[0]
    data = T.ToSparseTensor()(data)
    #data.adj_t = data.adj_t.to_symmetric()

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
    loader = NeighborSampler(data.adj_t, node_idx=train_idx,
                               sizes=[5, 3, 2], batch_size=100,
                               shuffle=True)
    subgraph_loader = NeighborSampler(data.adj_t, node_idx=None, sizes=[-1],
                                  batch_size=4096, shuffle=False,
                                  )
    return data, dataset, dataset_generator, loader, subgraph_loader, train_idx, split_idx

def confusion_matrix_plot(model, data, split_idx, evaluator, subgraph_loader, device, datatype, save_file):
    model.eval()
    
    out = model.inference(data.x, subgraph_loader, device)
    y_pred = out.argmax(dim=-1, keepdim=True)
    print(y_pred.shape)
    
    if save_file == True:
        #pd.DataFrame(y_pred[split_idx[datatype]].numpy()).to_csv('pred_' + str(datatype) + '.csv')
        #pd.DataFrame(data.y[split_idx[datatype]].numpy()).to_csv('label_' + str(datatype) + '.csv')
        pd.DataFrame(y_pred.numpy()).to_csv('pred_' + str(datatype) + '.csv')
        pd.DataFrame(data.y.numpy()).to_csv('label_' + str(datatype) + '.csv')
    
    # plot confusion matrix
    conf_matrix = confusion_matrix(data.y.numpy(), y_pred.numpy())
    fig, ax = plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6), cmap=plt.cm.Greys)
    ax.set_xticklabels(['', '0', '1', '2', '3', '4'], fontsize=22)
    ax.set_yticklabels(['', '0', '1', '2', '3', '4'], fontsize=22)
    font = {'size':22}

    plt.rc('font', **font)
    plt.xlabel('Predictions', fontsize=22)
    plt.ylabel('Actuals', fontsize=22)
    #plt.title('Confusion Matrix', fontsize=25)
    plt.savefig('confusion_matrix_' + str(datatype) + '.pdf', bbox_inches = 'tight',pad_inches = 0)


def main():
    parser = argparse.ArgumentParser(description='mult16')
    parser.add_argument('--bits', type=int, default=8)
    parser.add_argument('--bits_test', type=int, default=8)
    parser.add_argument('--datagen', type=int, default=0,
		help="0=multiplier generator, 1=adder generator, 2=loading design")
    # (0)(1) require bits as inputs; (2) requires designfile as input
    parser.add_argument('--datagen_test', type=int, default=0,
		help="0=multiplier generator, 1=adder generator, 2=loading design")
    # (0)(1) require bits as inputs; (2) requires designfile as input
    parser.add_argument('--multilabel', type=bool, default=False)
    parser.add_argument('--num_class', type=int, default=5)
    parser.add_argument('--designfile', '-f', type=str, default='')
    parser.add_argument('--designfile_test', '-ft', type=str, default='')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--runs', type=int, default=2)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device('cpu') ## cpu for now only

    ## generate dataset
    root_folder = os.path.abspath((os.path.dirname(os.path.abspath("gnn_sampler.py"))))
    print(root_folder)
    # new generator functionalities: 0) multiplier 1) adder 2) read design

    ## training dataset loading
    """

    dataset_generator = ABCGenDataset(args.bits, args.datagen, root_folder, args.designfile)
    dataset_generator.generate(dataset_path=root_folder+"/dataset")
    dataset = PygNodePropPredDataset(name=dataset_generator.design_name)
    print("Training on %s" % dataset_generator.design_name)
    data = dataset[0]
    data = T.ToSparseTensor()(data)
    #data.adj_t = data.adj_t.to_symmetric()

    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
    train_loader = NeighborSampler(data.adj_t, node_idx=train_idx,
                               sizes=[5, 3, 2], batch_size=100,
                               shuffle=True)
    subgraph_loader = NeighborSampler(data.adj_t, node_idx=None, sizes=[-1],
                                  batch_size=4096, shuffle=False,
                                  )

    """
    data, dataset, dataset_generator, train_loader, subgraph_loader, train_idx, split_idx \
            = dataloader_prep(args.bits, args.datagen, root_folder, args.designfile, device, args.num_class, args.multilabel)
    evaluator = Evaluator(name=dataset_generator.design_name)

    model = SAGE(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)

    data = data.to(device)

    #evaluator_test = Evaluator(name=dataset_generator_test.design_name)
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss, _ = train(model, data, train_idx, optimizer, train_loader, device)
            # train val
            datatype='train'
            result = test(model, data, split_idx, evaluator, subgraph_loader, datatype, device)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

        logger.print_statistics(run)
    logger.print_statistics()
    # evaluation
    logger_eval = Logger(1, args)
    data, dataset, dataset_generator, train_loader, subgraph_loader, train_idx, split_idx \
            = dataloader_prep(args.bits_test, args.datagen_test, root_folder, args.designfile, device, args.num_class, args.multilabel)
    evaluator = Evaluator(name=dataset_generator.design_name)

    for run_1 in range(1):
        for epoch in range(1):
            datatype='test'
            result = test(model, data, split_idx, evaluator, subgraph_loader,datatype, device)
            logger_eval.add_result(run_1, result)
            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

        logger_eval.print_statistics(run_1)
    confusion_matrix_plot(model, data, split_idx, evaluator, subgraph_loader, device, datatype='test', save_file=True)
    logger_eval.print_statistics()



if __name__ == "__main__":
    main()
