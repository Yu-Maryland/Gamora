import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.loader import NeighborSampler
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphSAGE, MLP,SAGEConv, global_mean_pool, BatchNorm
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split
from torch.nn import Linear
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from dataset_prep.dataset_el_pyg_cone import EdgeListDataset
#from loader.dataset_el_pyg import EdgeListDataset
import numpy as np
import wandb
torch.manual_seed(0)
def initialize_wandb(args):
    if args.wandb:
        wandb.init(
            project="el_sage_ensemble",
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
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers =3, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        self.bns.append(BatchNorm(hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(BatchNorm(hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.bns.append(BatchNorm(hidden_dim))

        self.fc = Linear(hidden_dim, hidden_dim)
        self.mlp = MLP([hidden_dim, hidden_dim, out_dim],
                       norm=None, dropout=0.5)

    
    def forward(self, data, gamora_output):
        #x = data.x
        x = torch.cat((data.x, gamora_output[0], gamora_output[1], gamora_output[2]), 1)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, data.adj_t)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout) #torch.Size([58666, hidden_dim])
        
        x = global_mean_pool(x, data.batch) # torch.Size([batch, hidden_dim])
        x = self.fc(x) # torch.Size([batch, hidden_dim])
        x = F.relu(x)
        
        x = self.mlp(x)
        return x #torch.Size([batch, 16])

def train_ensemble(gamora_model, model, loader, optimizer, device, dataset, bit):
    gamora_model.eval()
    model.train()
    total_loss = 0
    criterion = torch.nn.BCEWithLogitsLoss() #sigmoid + BCE
    for data in loader:
        data = data.to(device)
        out1, out2, out3 = gamora_model.forward_nosampler(data.x, data.adj_t, device)
        optimizer.zero_grad()
        out = model(data,[out1, out2, out3])
        loss = criterion(out, data.y.reshape(-1, dataset.num_classes)[:, bit].unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_acc = test(gamora_model, model, loader, device, dataset)
    
    return total_loss / len(loader), train_acc

def ensemble_predict(models, data, device):
    model_preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            data = data.to(device)
            out = model(data)
            out = torch.sigmoid(out)
            pred = (out > 0.5).float()
            model_preds.append(pred.cpu().numpy())
    model_preds = np.array(model_preds)
    final_preds = torch.Tensor(model_preds).squeeze(2).transpose(1,0)
    return final_preds

@torch.no_grad()
def test(gamora_model, model, loader, device, dataset):
    gamora_model.eval()
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out1, out2, out3 = gamora_model.forward_nosampler(data.x, data.adj_t, device)
        out = model(data, [out1, out2, out3])
        out = torch.sigmoid(out)
        pred = (out > 0.5).float()
        correct += (pred == data.y.reshape(-1, dataset.num_classes)).sum().item()
    return correct / len(loader.dataset) /len(loader.dataset[0].y)

@torch.no_grad()
def test_ensemble_bit(gamora_model, model, loader, device, dataset, bit):
    gamora_model.eval()
    model.eval()
    correct = 0
    total = 0
    for data in loader:
        data = data.to(device)
        out1, out2, out3 = gamora_model.forward_nosampler(data.x, data.adj_t, device)
        out = model(data, [out1, out2, out3])
        out = torch.sigmoid(out)
        pred = (out > 0.5).float()
        correct += (pred == data.y.reshape(-1, dataset.num_classes)[:, bit].unsqueeze(1)).sum().item()
        total += data.y.reshape(-1, dataset.num_classes)[:, bit].numel()
    return correct / total # len(loader.dataset) #/len(loader.dataset[0].y)

@torch.no_grad()
def test_ensemble_full(models, loader, device, dataset, bit):
    for model in models:
        model.eval()
    correct = 0
    total = 0
    for data in loader:
        data = data.to(device)
        out = ensemble_predict(models, data,device)
        out = torch.sigmoid(out).to(device)
        pred = (out > 0.5).float()
        correct += (pred == data.y.reshape(-1, dataset.num_classes)[:, :bit+1]).sum().item()
        total += data.y.reshape(-1, dataset.num_classes)[:, :bit+1].numel()
    return correct / total

def main(args):
    initialize_wandb(args)
    dataset = EdgeListDataset(root=args.root, highest_order=args.highest_order, PO_bit=args.PO_bit)
    #dataset = EdgeListDataset(root=args.root, highest_order=args.highest_order)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(len(dataset))
    print(dataset[0])
  
    # Split dataset into training, validation, and test sets
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42) #Does random_state ensure the same split for PO_bit=0~15?
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    #ensemble
    model = GraphSAGE(in_dim=dataset[0].num_node_features,
                    hidden_dim=args.hidden_dim, 
                    out_dim=1,
                    num_layers=args.num_layers,
                    dropout=args.dropout
                    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    
    for epoch in range(1, args.epochs + 1):
        loss, train_acc = train_ensemble(model, train_loader, optimizer, device, dataset, args.PO_bit)
        if epoch % 10 == 0:
            val_acc = test_ensemble_bit(model, val_loader, device, dataset, args.PO_bit)
            test_acc = test_ensemble_bit(model, test_loader, device, dataset, args.PO_bit)
            wandb.log({"Epoch": epoch, f"Loss_Bit_{args.PO_bit}": loss, f"Train_acc_Bit_{args.PO_bit}": train_acc, f"Val_acc_Bit_{args.PO_bit}": val_acc, f"Test_acc_Bit_{args.PO_bit}": test_acc})
            print(f'Bit: {args.PO_bit:02d}, Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
    #save model
    model_path = f'model_bit_{args.PO_bit}.pt'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')
    
    # load model and test 
    # Need to load different cone dataset for each trained model
    '''
    loaded_model = GraphSAGE(in_dim=dataset[0].num_node_features,
                             hidden_dim=args.hidden_dim, 
                             out_dim=1,
                             num_layers=args.num_layers,
                             dropout=args.dropout
                             ).to(device)
    loaded_models=[]
    for i in range(args.PO_bit):
        loaded_model.load_state_dict(torch.load('model_bit_{i}.pt'))
        loaded_model.eval()
        test_acc = test_ensemble_bit(loaded_model, test_loader, device, dataset, args.PO_bit)
        print(f'Loaded model test accuracy: {test_acc:.4f}')
        loaded_models.append(loaded_model)
    ensemble_test_acc = test_ensemble_full(loaded_models, test_loader, device, dataset, args.PO_bit)
    print("Test_acc:", ensemble_test_acc)
    wandb.log({"Ensemble Test Acc": ensemble_test_acc})
    '''
    #models.append(model)
    # Ensemble predictions
    #ensemble_test_acc = test_ensemble_full(models, test_loader, device, dataset, bit)
    #print("Test_acc:", ensemble_test_acc)
    #wandb.log({"Ensemble Test Acc": ensemble_test_acc})
        
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
    parser.add_argument('--PO_bit', type=int, default=0, help='of which bit cone dataset to train')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    args = parser.parse_args()
    main(args)
    #python el_sage.py --root /home/curie/GraphSAGE/dataset/edgelist --highest_order 16 --learning_rate 0.0001 --epochs 500 --hidden_dim 75 --batch_size 64
    #root = '/home/curie/masGen/DataGen/dataset'
    #root = '/home/curie/GraphSAGE/dataset/edgelist'