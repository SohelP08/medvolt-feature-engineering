# src/feature_extractor.py

import numpy as np

def extract_features(atoms):
    """
    Converts atom data into numerical features.
    Returns:
    - coordinates matrix
    - distance matrix (pairwise distances)
    """

    # Extract coordinates
    coordinates = np.array([atom["coordinates"] for atom in atoms])

    # Compute pairwise distance matrix
    num_atoms = coordinates.shape[0]
    distance_matrix = np.zeros((num_atoms, num_atoms))

    for i in range(num_atoms):
        for j in range(num_atoms):
            distance_matrix[i, j] = np.linalg.norm(
                coordinates[i] - coordinates[j]
            )

    return coordinates, distance_matrix
