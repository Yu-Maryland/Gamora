from collections.abc import Mapping
from typing import Any, List, Optional, Sequence, Union

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.data.on_disk_dataset import OnDiskDataset
from torch_geometric.typing import TensorFrame, torch_frame
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from dataset_el_pyg import EdgeListDataset

class Collater:
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Any]) -> Any:
        elem = batch[0]
        if isinstance(elem, BaseData):
            return Batch.from_data_list(
                batch,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
            )
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, TensorFrame):
            return torch_frame.cat(batch, along='row')
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")

    def collate_fn(self, batch: List[Any]) -> Any:
        if isinstance(self.dataset, OnDiskDataset):
            return self(self.dataset.multi_get(batch))
        return self(batch)

###Custom Collater to make the number of nodes in each batch the same!

class Custom_Collater(object):
    def __init__(self, total_dataset, dataset):
        self.max_num_nodes = self.find_max_num_nodes(total_dataset)
        
    def find_max_num_nodes(self, total_dataset):
        max_num_nodes = 0
        for data in total_dataset :
            if data.num_nodes > max_num_nodes:
                max_num_nodes = data.num_nodes
        return max_num_nodes
    
    def pad_graphs(self, batch):
        padded_batch = []
        for data in batch:
            num_nodes = data.num_nodes
            pad_size = self.max_num_nodes - num_nodes
            
            # Pad node features
            x_padded = torch.cat([data.x, torch.full((pad_size, data.x.size(1)), 50.0)], dim=0)
            
            # Pad edge indices
            edge_index_padded = data.edge_index
            edge_index_padded = torch.cat([edge_index_padded, torch.tensor([[num_nodes + i, num_nodes + i] for i in range(pad_size)], dtype=torch.long).t().contiguous()], dim=1)
            
            # Pad adjacency matrix
            adj_t_padded = data.adj_t.to_dense()
            adj_t_padded = torch.nn.functional.pad(adj_t_padded, (0, pad_size, 0, pad_size), value=0)
            adj_t_padded = SparseTensor.from_dense(adj_t_padded)
            
            # Create a new Data object with padded data
            padded_data = Data(x=x_padded, edge_index=edge_index_padded, y=data.y, adj_t=adj_t_padded)
            padded_batch.append(padded_data)

        return Batch.from_data_list(padded_batch)
    
    def collate_fn(self, batch):
        return self.pad_graphs(batch)
    
    
class DataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        total_dataset (Dataset): To define max_num_nodes, we need the the whole dataset.
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        total_dataset,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        # curie: 
        # kwargs.pop('collate_fn', None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        #self.collator = Collater(dataset, follow_batch, exclude_keys)
        self.collator = Custom_Collater(total_dataset, dataset)
        
        if isinstance(dataset, OnDiskDataset):
            dataset = range(len(dataset))

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=self.collator.collate_fn, # curie
            **kwargs,
        )

def main():
    dataset = EdgeListDataset(root = '/home/curie/masGen/DataGen/dataset16', highest_order = 16)
    data_loader = DataLoader(dataset, dataset, batch_size=32, shuffle=True)
    print(dataset[0])
    print(dataset[1])
    for batch in data_loader:
        print(batch)
        #DataBatch(x=[63808, 4], edge_index=[2, 62784], y=[512], adj_t=[63808, 63808, nnz=56624], batch=[63808], ptr=[33])
if __name__ == "__main__":
    main()
