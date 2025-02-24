import os
import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import torch
import math
import pickle

from torch_geometric.data import (
    Data,
    InMemoryDataset
)
from torch_geometric.io import read_txt_array
from torch_geometric.utils import coalesce, cumsum

class FakeNewsNetDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        name: str,
        feature: str,
        split: str = "train",
        ratio: float = 1.0,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        assert name in ['politifact', 'gossipcop']
        assert split in ['train', 'val', 'test']
        assert 0.0 <= ratio <= 1.0

        self.root = root
        self.name = name
        self.feature = feature
        self.ratio = ratio
        self.split = split

        super().__init__(root, transform, pre_transform, pre_filter,
                         force_reload=force_reload)

        path = self.processed_paths[['train', 'val', 'test'].index(split)]
        self.load(path)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        # if self.split == 'test' and self.ratio != 1.0:
        #     return osp.join(self.root, self.name, 'processed', self.feature, f'ratio_{self.ratio}')
        # return osp.join(self.root, self.name, 'processed', self.feature)
        return osp.join(self.root, self.name, 'processed', self.feature, self.split, f'ratio_{self.ratio}')

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'node_graph_id.npy', 'graph_labels.npy', 'A.txt', 'train_idx.npy',
            'val_idx.npy', 'test_idx.npy', f'new_{self.feature}_feature.npz'
        ]

    @property
    def processed_file_names(self) -> List[str]:
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self) -> None:
        if not osp.exists(self.raw_dir):
            raise RuntimeError(
                f'Raw data not found. Please put your data in {self.raw_dir}'
            )

    def _prune_nodes(self, data: Data) -> Data:
        if self.split != 'test' or self.ratio == 1.0:
            return data
        
        adj_list = [[] for _ in range(data.x.size(0))]
        for i in range(data.edge_index.size(1)):
            src, dst = data.edge_index[:, i]
            adj_list[src.item()].append(dst.item())

        in_degrees = [0] * len(adj_list)
        for edges in adj_list:
            for dst in edges:
                in_degrees[dst] += 1

        root = in_degrees.index(0)

        levels = {root: 0}
        queue = [(root, 0)]
        while queue:
            node, level = queue.pop(0)
            for neighbor in adj_list[node]:
                if neighbor not in levels:
                    levels[neighbor] = level + 1
                    queue.append((neighbor, level + 1))

        nodes_by_level = {}
        for node, level in levels.items():
            nodes_by_level.setdefault(level, []).append(node)
        
        total_nodes = data.x.size(0)
        nodes_to_keep = math.ceil(total_nodes * self.ratio)
        
        keep_mask = [True] * total_nodes
        current_nodes = total_nodes
        
        max_level = max(nodes_by_level.keys())
        for level in range(max_level, -1, -1):
            if current_nodes <= nodes_to_keep:
                break
            
            level_nodes = sorted(nodes_by_level.get(level, []))
            for node in level_nodes:
                if current_nodes <= nodes_to_keep:
                    break
                if keep_mask[node]:
                    keep_mask[node] = False
                    current_nodes -= 1

        keep_indices = torch.tensor([i for i, k in enumerate(keep_mask) if k])
        new_x = data.x[keep_indices]
        
        old_to_new = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(keep_indices)}
        
        new_edges = []
        for i in range(data.edge_index.size(1)):
            src, dst = data.edge_index[:, i].tolist()
            if src in old_to_new and dst in old_to_new:
                new_edges.append([old_to_new[src], old_to_new[dst]])
        
        new_edge_index = torch.tensor(new_edges).t() if new_edges else torch.empty((2, 0), dtype=torch.long)
        
        return Data(x=new_x, edge_index=new_edge_index, y=data.y)
    
    def _prune_sorted_nodes(self, root, data: Data) -> Data:
        ranking_path = osp.join(self.raw_dir, f'{self.name}_user_similarity_rankings.pkl')
        with open(ranking_path, 'rb') as f:
            rankings = pickle.load(f)
        
        # Get sorted nodes starting with root
        sorted_nodes = [0] + rankings.get(root, [])
        
        # Extend unvisited nodes to the end of sorted nodes
        all_nodes = set(range(data.x.size(0)))
        sorted_nodes_set = set(sorted_nodes)
        remaining_nodes = list(all_nodes - sorted_nodes_set)
        sorted_nodes.extend(remaining_nodes)
        
        # Calculate how many nodes to keep
        nodes_to_keep = math.ceil(data.x.size(0) * self.ratio)
        keep_indices = sorted_nodes[:nodes_to_keep]
        
        # Convert to set for O(1) lookup
        keep_indices_set = set(keep_indices)
        
        # Filter edges first
        edge_list = data.edge_index.t().tolist()
        valid_edges = []
        for src, dst in edge_list:
            if src in keep_indices_set and dst in keep_indices_set:
                valid_edges.append([src, dst])
        
        # Create mapping after filtering edges
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}
        
        # Create new edge index with mapped indices
        new_edges = [[old_to_new[src], old_to_new[dst]] for src, dst in valid_edges]
        new_edge_index = torch.tensor(new_edges, dtype=torch.long).t() if new_edges else torch.empty((2, 0), dtype=torch.long)
        
        # Select nodes
        new_x = data.x[keep_indices]
        
        # Verify data consistency
        if new_edge_index.numel() > 0:  # Only check if we have edges
            assert new_edge_index.max() < new_x.size(0), f"Edge index {new_edge_index.max()} out of bounds for {new_x.size(0)} nodes"
            assert new_edge_index.min() >= 0, f"Negative edge index found: {new_edge_index.min()}"
        
        return Data(x=new_x, edge_index=new_edge_index, y=data.y)

    def process(self) -> None:
        import scipy.sparse as sp

        x = sp.load_npz(
            osp.join(self.raw_dir, f'new_{self.feature}_feature.npz'))
        x = torch.from_numpy(x.todense()).to(torch.float)

        edge_index = read_txt_array(osp.join(self.raw_dir, 'A.txt'), sep=',',
                                    dtype=torch.long).t()
        edge_index = coalesce(edge_index, num_nodes=x.size(0))

        y = np.load(osp.join(self.raw_dir, 'graph_labels.npy'))
        y = torch.from_numpy(y).to(torch.long)
        _, y = y.unique(sorted=True, return_inverse=True)

        batch = np.load(osp.join(self.raw_dir, 'node_graph_id.npy'))
        batch = torch.from_numpy(batch).to(torch.long)

        node_slice = cumsum(batch.bincount())
        edge_slice = cumsum(batch[edge_index[0]].bincount())
        graph_slice = torch.arange(y.size(0) + 1)
        self.slices = {
            'x': node_slice,
            'edge_index': edge_slice,
            'y': graph_slice
        }

        edge_index -= node_slice[batch[edge_index[0]]].view(1, -1)
        self.data = Data(x=x, edge_index=edge_index, y=y)

        for path, split in zip(self.processed_paths, ['train', 'val', 'test']):
            idx = np.load(osp.join(self.raw_dir, f'{split}_idx.npy')).tolist()
            data_list = [self.get(i) for i in idx]
            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]
            data_list = [self._prune_sorted_nodes(i, d) for i, d in zip(idx, data_list)]
            self.save(data_list, path)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({len(self)}, name={self.name}, '
                f'feature={self.feature})')
        
if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')
    for split in ['train', 'val', 'test']:
        test_dataset_1 = FakeNewsNetDataset(path, 'politifact', 'bert', split=split, ratio=1.0)
        test_dataset_2 = FakeNewsNetDataset(path, 'politifact', 'bert', split=split, ratio=0.8)
        test_dataset_3 = FakeNewsNetDataset(path, 'politifact', 'bert', split=split, ratio=0.5)
        test_dataset_4 = FakeNewsNetDataset(path, 'politifact', 'bert', split=split, ratio=0.2)
        
        ls1 = []
        for batch in test_dataset_1:
            ls1.append(batch.x.shape)
        ls2 = []
        for batch in test_dataset_2:
            ls2.append(batch.x.shape)
        ls3 = []
        for batch in test_dataset_3:
            ls3.append(batch.x.shape)
        ls4 = []
        for batch in test_dataset_4:
            ls4.append(batch.x.shape)
        print(len(ls1), len(ls2), len(ls3), len(ls4))
        