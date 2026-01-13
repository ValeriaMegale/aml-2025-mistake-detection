"""
Task Graph Dataset for GNN Classification (Substep 4).

Carica grafi preparati da prepare_graph_data.py in formato PyTorch Geometric.
"""

import json
import os
from pathlib import Path
import torch
from torch_geometric.data import Data, Dataset


class TaskGraphDataset(Dataset):
    """
    Dataset per grafi di task graphs matched.
    
    Carica grafi salvati come .pt files nella directory specificata.
    Supporta leave-one-out split tramite metadata.json.
    """
    
    def __init__(self, root, split='train', transform=None, pre_transform=None):
        """
        Args:
            root: str o Path - directory root contenente train/, test/, metadata.json
            split: 'train' o 'test' - quale split caricare
            transform: optional transform function
            pre_transform: optional pre-transform function
        """
        self.split = split
        self.root = Path(root)
        
        metadata_file = self.root / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {'graphs': {}}
            print(f"Warning: metadata.json not found in {self.root}")
        
        super().__init__(root, transform, pre_transform)
        
        self.data_list = self.load_graphs()
        
    def load_graphs(self):
        data_list = []
        split_dir = self.root / self.split
        
        if not split_dir.exists():
            print(f"Warning: {split_dir} does not exist")
            return data_list
        
        graph_files = list(split_dir.glob('*.pt'))
        
        for graph_file in sorted(graph_files):
            video_id = graph_file.stem
            if self.metadata.get('graphs', {}).get(video_id, {}).get('split') == self.split:
                data_list.append(str(graph_file))
            elif not self.metadata.get('graphs'):  # If no metadata, include all
                data_list.append(str(graph_file))
        
        print(f"Loaded {len(data_list)} graphs for {self.split} split")
        return data_list
    
    def len(self):
        """Restituisce il numero di grafi nel dataset."""
        return len(self.data_list)
    
    def get(self, idx):
        """
        Carica e restituisce il grafo all'indice idx.
        
        Args:
            idx: int - indice del grafo
        
        Returns:
            Data object PyTorch Geometric
        """
        graph_path = self.data_list[idx]
        data = torch.load(graph_path, weights_only=False)
        
        if self.transform is not None:
            data = self.transform(data)
        
        return data


def create_leave_one_out_datasets(root, test_recipe_id):
    """
    Crea dataset train/test per leave-one-out split.
    
    Args:
        root: str o Path - directory root con metadata.json
        test_recipe_id: str - ID ricetta da usare per test
    
    Returns:
        train_dataset, test_dataset: TaskGraphDataset instances
    """
    root = Path(root)
    metadata_file = root / 'metadata.json'
    
    if not metadata_file.exists():
        raise FileNotFoundError(f"metadata.json not found in {root}")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    all_video_ids = list(metadata['graphs'].keys())
    
    train_ids = [vid for vid in all_video_ids 
                 if not vid.startswith(f"{test_recipe_id}_")]
    test_ids = [vid for vid in all_video_ids 
                if vid.startswith(f"{test_recipe_id}_")]
    
    train_metadata = metadata.copy()
    train_metadata['graphs'] = {vid: metadata['graphs'][vid] for vid in train_ids}
    for vid in train_ids:
        train_metadata['graphs'][vid]['split'] = 'train'
    
    test_metadata = metadata.copy()
    test_metadata['graphs'] = {vid: metadata['graphs'][vid] for vid in test_ids}
    for vid in test_ids:
        test_metadata['graphs'][vid]['split'] = 'test'
    
    train_metadata_file = root / 'metadata_train_temp.json'
    test_metadata_file = root / 'metadata_test_temp.json'
    
    with open(train_metadata_file, 'w') as f:
        json.dump(train_metadata, f, indent=2)
    with open(test_metadata_file, 'w') as f:
        json.dump(test_metadata, f, indent=2)
    
    train_dataset = TaskGraphDataset(root, split='train')
    test_dataset = TaskGraphDataset(root, split='test')
    
    train_dataset.metadata = train_metadata
    test_dataset.metadata = test_metadata
    
    # Filter data_list based on actual split
    train_dataset.data_list = [
        str(root / 'train' / f"{vid}.pt") for vid in train_ids
        if (root / 'train' / f"{vid}.pt").exists() or (root / 'test' / f"{vid}.pt").exists()
    ]
    test_dataset.data_list = [
        str(root / 'test' / f"{vid}.pt") for vid in test_ids
        if (root / 'test' / f"{vid}.pt").exists()
    ]
    
    # If files are in different locations, try to find them
    all_pt_files = list((root / 'train').glob('*.pt')) + list((root / 'test').glob('*.pt'))
    pt_dict = {f.stem: f for f in all_pt_files}
    
    train_dataset.data_list = [str(pt_dict[vid]) for vid in train_ids if vid in pt_dict]
    test_dataset.data_list = [str(pt_dict[vid]) for vid in test_ids if vid in pt_dict]
    
    return train_dataset, test_dataset
