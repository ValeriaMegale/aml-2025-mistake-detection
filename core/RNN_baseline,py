import torch
import torch.nn as nn

class RNNBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1):
        super(RNNBaseline, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM Layer
        # batch_first=True si aspetta input (Batch, Seq_Len, Features)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Classification Head
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        
        # LSTM output shape: [batch_size, seq_len, hidden_dim]
        # hn shape: [num_layers, batch_size, hidden_dim]
        # cn shape: [num_layers, batch_size, hidden_dim]
        out, (hn, cn) = self.lstm(x)
        
        # Usiamo l'hidden state dell'ultimo step temporale per la classificazione
        # hn[-1] prende l'ultimo layer
        out = self.fc(hn[-1])
        
        return out