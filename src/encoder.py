# src/encoder.py

import torch
import torch.nn as nn

class ProteinEncoder(nn.Module):
    def __init__(self, input_dim=3, latent_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, x):
        return self.encoder(x)
