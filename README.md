# Medvolt – Feature Engineering Assignment

## About the Assignment
This assignment focuses on building a basic but clean feature extraction and
encoding–decoding pipeline for a protein structure provided in PDB format.

The goal was to design a representation that can later be used in
structure-based generative learning workflows.
Model training or performance evaluation was not required.

---

## Input Data
- Protein structure file: `7rfw.pdb`
- Format: PDB (Protein Data Bank)

---

## Project Structure

medvolt_assignment/
data/
     7rfw.pdb
 
src/
    pdb_parser.py
    feature_extractor.py
    encoder.py
    decoder.py

bonus/
    visualize.py
    consistency_check.py


main.py
requirements.txt
README.md

---

## Feature Extraction Approach

I used an **atom-level representation** for this assignment.

From the PDB file, the following information is extracted:
- Atom name
- Residue name
- Chain ID
- 3D coordinates (x, y, z)

In addition, a pairwise Euclidean distance matrix is computed using atom
coordinates to capture spatial relationships.

This representation was chosen because atom-level features preserve detailed
structural information and can later be aggregated into residue-level or
graph-based representations if needed.

---

## Feature Encoding

The encoder converts atom coordinates into a numerical latent representation.

- Input shape: `(number_of_atoms, 3)`
- Output shape: `(number_of_atoms, latent_dimension)`

A simple neural network is used to demonstrate how raw structural features can
be transformed into embeddings suitable for downstream machine learning or
generative models.

---

## Feature Decoding

The decoder maps latent embeddings back to an approximate coordinate space.

Exact reconstruction is not the objective here.
The purpose of the decoder is to show:
- The encoding is reversible in principle
- The latent representation retains structural information

---

## Bonus Work

### 1. Structure Visualization
A 3D scatter plot of atom coordinates is generated to visually verify that the
extracted features correctly represent the spatial structure of the protein.

This was mainly used as a sanity check for parsing and feature extraction.

### 2. Encoding–Decoding Consistency Check
A reconstruction mean squared error (MSE) is computed between original and
reconstructed coordinates.

Since the encoder–decoder is not trained, a high error is expected.
This step is included to validate the overall pipeline rather than accuracy.

---

## Environment Setup

A Python virtual environment was used for dependency management.

### Setup Instructions
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

Run the Main Pipeline
python main.py

Run Bonus Scripts
python -m bonus.visualize
python -m bonus.consistency_check
