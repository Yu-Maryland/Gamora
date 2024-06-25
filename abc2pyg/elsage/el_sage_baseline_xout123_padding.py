import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.loader import NeighborSampler
from torch_geometric.data import Batch
from torch_geometric.nn import GraphSAGE, MLP, SAGEConv, global_mean_pool, BatchNorm
from torch_geometric.transforms import ToSparseTensor
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor
from sklearn.model_selection import train_test_split
from torch.nn import Linear
from torch_geometric.loader import DataLoader
#from dataset_prep.dataloader_padding import DataLoader, Custom_Collater
from dataset_prep.dataset_el_pyg_padding import EdgeListDataset
import torch_geometric.transforms as T
import wandb
torch.manual_seed(0)
def initialize_wandb(args):
    if args.wandb:
        wandb.init(
            project="el_sage",
            config={
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "num_layers": args.num_layers,
                "hidden_dim": args.hidden_dim,
                "highest_order": args.highest_order,
                "dropout": args.dropout
                
            }
        )
    else:
        wandb.init(mode="disabled")


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, max_num_nodes, num_layers=4, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.max_num_nodes = max_num_nodes
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.mlp1 = MLP([max_num_nodes, hidden_dim, 1],
                       norm=None, dropout=0.5) #instead of global_mean_pool
        self.bn = BatchNorm(hidden_dim)
        #self.fc = Linear(hidden_dim, hidden_dim)
        self.mlp2 = MLP([hidden_dim, hidden_dim, out_dim],
                       norm=None, dropout=0.5)

    def forward(self, data, gamora_output):
        x = torch.cat((data.x, gamora_output[0], gamora_output[1], gamora_output[2]), 1)
        for conv in self.convs:
            x = conv(x, data.adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout) #torch.Size([58666, hidden_dim])
        '''
        x = global_mean_pool(x, data.batch) # torch.Size([batch, hidden_dim])
        x = self.fc(x) # torch.Size([batch, hidden_dim])
        '''
        
        x = x.reshape([-1,x.shape[1],self.max_num_nodes])
        #x = x.permute(1,0) # if every batch size is the same [75, #]
        #print(x.shape)
        x = self.mlp1(x).squeeze(2) #[32, 75]
        #print(x.shape)
        x = self.bn(x)
        x = F.relu(x)
        x = self.mlp2(x)
        return x #torch.Size([batch, 16])


def train(gamora_model, model, loader, optimizer, device, dataset):
    import time
    start_time = time.time()
    gamora_model.eval()
    model.train()
    total_loss = 0
    criterion = torch.nn.BCEWithLogitsLoss() #sigmoid + BCE
    for data in loader:
        data = data.to(device)
        out1, out2, out3 = gamora_model.forward_nosampler(data.x, data.adj_t, device)
        optimizer.zero_grad()
        out = model(data,[out1, out2, out3])
        loss = criterion(out, data.y.reshape(-1, dataset.num_classes))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print("--- Train time: %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    train_acc, train_acc_all_bits = test(gamora_model, model, loader, device, dataset)
    print("--- Test time: %s seconds ---" % (time.time() - start_time))
    return total_loss / len(loader), train_acc, train_acc_all_bits

@torch.no_grad()
def test(gamora_model, model, loader, device, dataset):
    gamora_model.eval()
    model.eval()
    correct = 0
    correct_all = 0
    total = 0
    total_all = 0
    for data in loader:
        data = data.to(device)
        out1, out2, out3 = gamora_model.forward_nosampler(data.x, data.adj_t, device)
        out = model(data, [out1, out2, out3])
        out = torch.sigmoid(out)
        pred = (out > 0.5).float()
        correct += (pred == data.y.reshape(-1, dataset.num_classes)).sum().item()
        correct_all+= torch.eq(pred, data.y.reshape(-1, dataset.num_classes)).all(dim=1).sum().item()
        total += data.y.reshape(-1, dataset.num_classes).numel()
        total_all += len(torch.eq(pred, data.y.reshape(-1, dataset.num_classes)).all(dim=1))
    return correct / total, correct_all / total_all

def main(args):
    initialize_wandb(args)
    
    dataset = EdgeListDataset(root=args.root, highest_order=args.highest_order)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Split dataset into training, validation, and test sets
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)

    # train_loader = DataLoader(dataset, train_dataset, batch_size=args.batch_size, shuffle=True)#, num_workers=4)
    # val_loader = DataLoader(dataset, val_dataset, batch_size=args.batch_size, shuffle=False)#, num_workers=4)
    # test_loader = DataLoader(dataset, test_dataset, batch_size=args.batch_size, shuffle=False)#, num_workers=4)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    max_num_nodes = dataset.find_max_num_nodes()
    
    model = GraphSAGE(in_dim=dataset[0].num_node_features, 
                 hidden_dim=args.hidden_dim, 
                 out_dim=dataset.num_classes,
                 max_num_nodes = max_num_nodes,
                 num_layers=args.num_layers,
                 dropout=args.dropout
                 ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)#, weight_decay=5e-4)
    
    for epoch in range(1, args.epochs + 1):
        loss, train_acc, train_all_bits = train(model, train_loader, optimizer, device, dataset)
        if epoch % 1 == 0:
            val_acc, val_acc_all_bits = test(model, val_loader, device, dataset)
            test_acc, test_acc_all_bits = test(model, test_loader, device, dataset)
            wandb.log({"Epoch": epoch, "Loss": loss, "Train_acc": train_acc, "Train_acc_all_bits": train_all_bits, 
                       "Val_acc":val_acc, "Test_acc": test_acc, "Val_acc_all_bits":val_acc_all_bits, "Test_acc": test_acc_all_bits})
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, Train acc all bits: {train_all_bits:.4f}, Val acc all bits: {val_acc_all_bits:.4f}, Test acc all bits: {test_acc_all_bits:.4f}')
            
    
if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser(description='ELGraphSAGE Training')
    parser.add_argument('--root', type=str, default='/home/curie/masGen/DataGen/dataset16', help='Root directory of dataset')
    parser.add_argument('--highest_order', type=int, default=16, help='Highest order for the EdgeListDataset')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden_dim', type=int, default=75, help='Hidden dimension size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    args = parser.parse_args()
    main(args)
    
