# bonus/visualize.py

import matplotlib.pyplot as plt
from src.pdb_parser import parse_pdb
from src.feature_extractor import extract_features

def visualize_structure(pdb_path):
    atoms = parse_pdb(pdb_path)
    coords, _ = extract_features(atoms)

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, s=1)

    ax.set_title("Protein Atom-Level Spatial Structure")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()

if __name__ == "__main__":
    visualize_structure("data/7rfw.pdb")
