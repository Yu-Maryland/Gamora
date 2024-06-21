import os
import os.path as osp
import torch
from torch_geometric.data import Data, InMemoryDataset, Batch
from torch_geometric.utils import to_scipy_sparse_matrix
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from typing import List
from torch_geometric.loader import DataLoader
#from loader.dataloader import DataLoader

class EdgeListDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, highest_order = 16, PO_bit=0):
        self.highest_order = highest_order
        self.PO_bit = PO_bit
        super(EdgeListDataset, self).__init__(root, transform, pre_transform, highest_order)
        print(f'Load dataset {self.processed_paths[0]}')
        self.data, self.slices = torch.load(self.processed_paths[0])

    '''
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects
            - highest_order: highest_order of Mas.eqn
    '''
    @property
    def num_classes(self):
        return self.highest_order
        
    @property
    def raw_file_names(self):
        sorted(os.listdir(self.root_folders)) #processed folder

    @property
    def processed_file_names(self):
        return ['data'+str(self.PO_bit)+'.pt']
    
    def process(self):
        data_list= []

        root_folders = sorted(os.listdir(self.root))
        for data_name in root_folders:
            if 'processed' in data_name:
                pass
            else:
                print(data_name)
                folder = os.path.join(self.root, data_name)
                bprimtive_path = os.path.join(folder, 'bprimtive')
                mas_el_path = os.path.join(folder, 'Mas'+str(self.highest_order)+'_z_'+str(self.PO_bit)+'_.el')

                # Load labels
                with open(bprimtive_path, 'r') as f:
                    labels = [int(x) for x in f.read().strip().split()]
                    y = torch.zeros(self.highest_order)
                    y[torch.tensor(labels) - 1] = 1  # 1-hot encoding
                # Load graph edges and node features
                edge_index = []
                node_feat = {}
                with open(mas_el_path, 'r') as f:
                    f.readline()
                    for line in f:
                        u, v, node_type = line.strip().split()[:3] #Cone datasets do not have edge_type for AIG
                        u, v = int(u)-1, int(v)-1 # Convert to zero-based index
                        edge_index.append([u, v])
                        #edge_type = [int(bit) for bit in edge_type]
                        if node_type == 'Pi':
                            node_feat[u] = [99, 99, 99, 99]
                            node_feat[v] = [99, 99, 99, 99]
                        elif node_type == 'AIG':
                            if v not in node_feat:
                                node_feat[v] = [1, 1] * 2 #I assume 1,1 for edge_type of AIG
                        elif node_type == 'Po':
                            node_feat[v] = [-99, -99, -99, -99]

                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                # Convert x_dict to a tensor
                num_nodes = edge_index.max().item() + 1
                x = torch.full((num_nodes, 4), 50).float()  # Initialize with 50
                
                for node, features in node_feat.items():
                    x[node-1] = torch.tensor(features).float()
                
                adj_t = SparseTensor.from_edge_index(edge_index)
                data = Data(edge_index=edge_index, y=y, x=x, adj_t=adj_t)
                data = data if self.pre_transform is None else self.pre_transform(data)
                data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == '__main__':
    #Generate processed data for each cone
    for i in range(16):
        dataset = EdgeListDataset(root = '/home/curie/masGen/DataGen/dataset16', highest_order = 16, PO_bit=i) 
        
    '''
    print(dataset[0])

    dataloader = DataLoader(dataset, dataset, batch_size=32)
    
    from sklearn.model_selection import train_test_split
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(dataset, train_dataset, batch_size=32, shuffle=True)
    val_loader= DataLoader(dataset, val_dataset, batch_size=32, shuffle=False)
    test_loader= DataLoader(dataset, test_dataset, batch_size=32, shuffle=False)
    for batch in train_loader:
        print(batch)
    '''
