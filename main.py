from src.pdb_parser import parse_pdb
from src.feature_extractor import extract_features
from src.encoder import ProteinEncoder
from src.decoder import ProteinDecoder
import torch

atoms = parse_pdb("data/7rfw.pdb")
coords, _ = extract_features(atoms)

coords_tensor = torch.tensor(coords, dtype=torch.float32)

encoder = ProteinEncoder()
decoder = ProteinDecoder()

latent = encoder(coords_tensor)
reconstructed = decoder(latent)

print("Original:", coords_tensor[0])
print("Reconstructed:", reconstructed[0])
