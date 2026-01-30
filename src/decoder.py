# src/decoder.py

import torch.nn as nn

class ProteinDecoder(nn.Module):
    def __init__(self, latent_dim=128, output_dim=3):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.decoder(x)
