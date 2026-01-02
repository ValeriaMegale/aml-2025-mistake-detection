import torch
import torch.nn as nn


class TaskVerifier(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, n_heads=4, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, src_key_padding_mask):
        b, s, _ = x.shape
        x = self.input_proj(x)

        # Add Positional Encoding
        if s <= self.pos_embedding.size(1):
            x = x + self.pos_embedding[:, :s, :]
        else:
            x = x + self.pos_embedding[:, :self.pos_embedding.size(1), :]

        feat = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Mean Pooling (masked)
        valid_mask = (~src_key_padding_mask).unsqueeze(-1).float()
        sum_feat = (feat * valid_mask).sum(dim=1)
        valid_lengths = valid_mask.sum(dim=1).clamp(min=1.0)
        mean_feat = sum_feat / valid_lengths

        return self.classifier(mean_feat)