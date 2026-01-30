# bonus/consistency_check.py

import torch
import numpy as np
from src.pdb_parser import parse_pdb
from src.feature_extractor import extract_features
from src.encoder import ProteinEncoder
from src.decoder import ProteinDecoder

def reconstruction_error(pdb_path):
    atoms = parse_pdb(pdb_path)
    coords, _ = extract_features(atoms)

    coords_tensor = torch.tensor(coords, dtype=torch.float32)

    encoder = ProteinEncoder()
    decoder = ProteinDecoder()

    latent = encoder(coords_tensor)
    reconstructed = decoder(latent)

    error = torch.mean((coords_tensor - reconstructed) ** 2).item()
    return error

if __name__ == "__main__":
    mse = reconstruction_error("data/7rfw.pdb")
    print(f"Reconstruction MSE (untrained): {mse:.4f}")
