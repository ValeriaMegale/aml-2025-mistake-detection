import torch
from torch_geometric.loader import DataLoader

from graph_classification_substep4.dataset import TaskGraphDataset
from graph_classification_substep4.model import DAGNNClassifier
from graph_classification_substep4.utils import train, test

# Parametri
root = './substep4/graph_classification_substep4/data'  # Path aggiornato ai dati
epochs = 50
batch_size = 64
hidden_channels = 64
num_classes = 2  # Corretto/Non corretto

# Dataset
train_dataset = TaskGraphDataset(root + '/train')
test_dataset = TaskGraphDataset(root + '/test')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)



# Modello DAGNN per il substep 4
sample = train_dataset.get(0)
model = DAGNNClassifier(sample.num_node_features, hidden_channels, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(1, epochs+1):
    loss = train(model, train_loader, optimizer)
    acc = test(model, test_loader)
    print(f'Epoch: {epoch}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')
