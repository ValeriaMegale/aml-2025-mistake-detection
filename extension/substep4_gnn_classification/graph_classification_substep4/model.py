"""
DAGNN-based Classifier for Task Graph Classification (Substep 4).

Secondo la traccia, il modello deve:
1. Prendere node features che sono combinazione di text + visual features matched
2. Applicare proiezione learnable per fondere text + visual
3. Usare DAGNN layer per processare il DAG
4. Classificare se il task graph (recipe) è corretto o errato
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GraphConv

# DAGNNConv implementation (simplified version for DAGs)
class DAGNNConvWrapper(torch.nn.Module):
    """
    Simplified DAGNN layer wrapper using GraphConv.
    GraphConv works well with directed graphs including DAGs.
    """
    def __init__(self, in_channels, K=10):
        super().__init__()
        self.K = K
        # Use GraphConv which works with directed graphs
        self.conv = GraphConv(in_channels, in_channels)
    
    def forward(self, x, edge_index):
        # Apply K propagation steps
        for _ in range(self.K):
            x = self.conv(x, edge_index)
            x = F.relu(x)
        return x


class DAGNNClassifier(torch.nn.Module):
    """
    DAGNN-based classifier per graph-level classification.
    
    Architettura:
    - Input projection: proietta node features (text + visual concatenate) a hidden_dim
    - DAGNN layers: processa il DAG con DAGNNConv
    - Global pooling: aggrega node embeddings a graph embedding
    - Classifier: binary classification head
    """
    
    def __init__(self, in_channels, hidden_channels, num_classes=2, K=10, dropout=0.3):
        """
        Args:
            in_channels: dimensione input node features (text_dim + visual_dim, tipicamente 512 + 768 = 1280)
            hidden_channels: dimensione hidden/embedding (dopo proiezione)
            num_classes: numero classi (2 per binary classification)
            K: numero di propagazioni per DAGNN
            dropout: dropout rate
        """
        super().__init__()
        
        # Input projection: combina text + visual features in uno spazio comune
        # Secondo traccia: "learnable projection of the node features and the visual features"
        self.input_proj = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        
        # Second layer (opzionale, per maggior espressività)
        self.hidden_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.LayerNorm(hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        )
        
        # DAGNN layer: specificamente progettato per Directed Acyclic Graphs
        self.dagnn = DAGNNConvWrapper(hidden_channels, K)
        
        # Classification head (binary classification, output 1 dim)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels // 2, 1)  # Binary classification: 1 output
        )
        
    def forward(self, x, edge_index, batch):
        """
        Forward pass.
        
        Args:
            x: [N_nodes, in_channels] - node features (text + visual concatenate)
            edge_index: [2, N_edges] - edge connectivity (DAG)
            batch: [N_nodes] - batch assignment vector
        
        Returns:
            logits: [batch_size, num_classes] - classification logits
        """
        # Project input features
        x = self.input_proj(x)  # [N_nodes, hidden_channels]
        x = self.hidden_proj(x)  # [N_nodes, hidden_channels]
        
        # DAGNN layer: propaga informazioni attraverso il DAG
        x = self.dagnn(x, edge_index)  # [N_nodes, hidden_channels]
        
        # Global mean pooling: aggrega node embeddings a graph embedding
        graph_embedding = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        
        # Classification
        logits = self.classifier(graph_embedding)  # [batch_size, num_classes]
        
        return logits


def get_model_input_dim(text_dim=512, visual_dim=768):
    """
    Calcola la dimensione input del modello basata su text e visual dimensions.
    
    Args:
        text_dim: dimensione text embeddings (CLIP: 512, sentence-transformers: 384)
        visual_dim: dimensione visual embeddings (perception: 768, omnivore: 1024)
    
    Returns:
        in_channels: text_dim + visual_dim
    """
    return text_dim + visual_dim
