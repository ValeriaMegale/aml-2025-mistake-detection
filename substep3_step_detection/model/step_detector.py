import torch
import torch.nn as nn


class StepMistakeDetector(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, n_heads=4, n_layers=2):
        super().__init__()

        # Proiezione Input
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional Encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, hidden_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classificatore (Dense-Prediction Head)
        # Viene applicato a OGNI step della sequenza indipendentemente
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # Output logit per classificazione binaria
            # Nota: Non mettiamo Sigmoid qui se usiamo BCEWithLogitsLoss (più stabile)
        )

    def forward(self, x, src_key_padding_mask):
        # x: [Batch, Seq_Len, 1024]
        b, s, _ = x.shape
        x = self.input_proj(x)

        # Positional Encoding
        if s <= self.pos_embedding.size(1):
            x = x + self.pos_embedding[:, :s, :]
        else:
            x = x + self.pos_embedding[:, :self.pos_embedding.size(1), :]

        # Transformer Pass
        # Output: [Batch, Seq_Len, Hidden_Dim]
        feat = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Classificazione Token-Level
        # Output: [Batch, Seq_Len, 1]
        logits = self.classifier(feat)

        return logits