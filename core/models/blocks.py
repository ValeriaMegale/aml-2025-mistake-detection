import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import dropout

from constants import Constants as const

# define the transformer backbone here
EncoderLayer = nn.TransformerEncoderLayer
Encoder = nn.TransformerEncoder


def fetch_input_dim(config, decoder=False):
    if config.backbone == const.OMNIVORE:
        return 1024
    elif config.backbone == const.SLOWFAST:
        return 400
    elif config.backbone == const.X3D:
        return 400
    elif config.backbone == const.RESNET3D:
        return 400
    elif config.backbone == const.IMAGEBIND:
        if decoder is True:
            return 1024
        k = len(config.modality)
        return 1024 * k



class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x
import torch
import torch.nn as nn

class RNNBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.5):
        super(RNNBaseline, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM Layer con dropout
        # batch_first=True si aspetta input (Batch, Seq_Len, Features)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
        # Classification Head
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout_layer = nn.Dropout(dropout)
    def forward(self, x):
        # x shape potrebbe essere [batch_size, seq_len, input_dim] o [seq_len, input_dim]
        # Se manca la dimensione batch, aggiungila
        original_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [seq_len, input_dim] -> [1, seq_len, input_dim]
            was_2d = True
        else:
            was_2d = False
        
        # x shape: [batch_size, seq_len, input_dim]
        
        # LSTM output shape: [batch_size, seq_len, hidden_dim]
        # hn shape: [num_layers, batch_size, hidden_dim]
        # cn shape: [num_layers, batch_size, hidden_dim]
        lstm_out, (hn, cn) = self.lstm(x)
        
        # Applica dropout dopo LSTM
        lstm_out = self.dropout(lstm_out)
        
        # Applica il fully connected layer a ogni timestep dell'output LSTM
        # lstm_out shape: [batch_size, seq_len, hidden_dim]
        # Reshape per applicare fc: [batch_size * seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = lstm_out.shape
        lstm_out_reshaped = lstm_out.reshape(batch_size * seq_len, hidden_dim)
        
        # Applica il fully connected layer
        out = self.fc(lstm_out_reshaped)  # [batch_size * seq_len, output_dim]
        
        # Reshape di nuovo: [batch_size, seq_len, output_dim]
        out = out.reshape(batch_size, seq_len, -1)
        
        # Se era 2D in input, rimuovi la dimensione batch per coerenza con altri modelli
        if was_2d:
            out = out.squeeze(0)  # [seq_len, output_dim]
        
        return out

class MLP1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP1, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size * 8)
        self.layer2 = nn.Linear(hidden_size * 8, hidden_size * 2)
        self.layer3 = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class CNN(nn.Module):
    def __init__(self, in_channels, final_width, final_height, num_classes):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * final_width * final_height, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor, indices=None) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        if indices is None:
            x = x + self.pe[:, :x.size(1)]
        else:
            pos = torch.cat([self.pe[:, index] for index in indices])
            x = x + pos
        return self.dropout(x)


class PositionalEncodingLearn(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.embed = nn.Embedding(max_len, d_model)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.embed.weight)

    def forward(self, x, indices=None):
        # x: b, l, d
        r = torch.arange(x.shape[1], device=x.device)
        embed = self.embed(r)  # seq_len, embedding_dim
        return x + embed.repeat(x.shape[0], 1, 1)
