
import torch
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool, DAGNNConv





# DAGNN-based classifier
class DAGNNClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, K=10):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.dagnn = DAGNNConv(hidden_channels, K)
        self.lin_out = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.dagnn(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.lin_out(x)
        return x
