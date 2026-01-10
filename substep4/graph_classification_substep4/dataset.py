
import torch
from torch_geometric.data import Data, Dataset
import os
import pandas as pd

class TaskGraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data_list = self.load_graphs()

    def load_graphs(self):
        # Carica i grafi dal disco (adatta questa funzione ai tuoi dati)
        # Esempio: ogni grafo è salvato come file .pt o .csv
        data_list = []
        # ...
        return data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
