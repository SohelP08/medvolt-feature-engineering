# src/pdb_parser.py

from Bio.PDB import PDBParser

def parse_pdb(pdb_file_path):
    """
    This function reads a PDB file and extracts atom-level information.
    Returns a list of atoms with their coordinates.
    """

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file_path)

    atoms_data = []

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_info = {
                        "atom_name": atom.get_name(),
                        "residue_name": residue.get_resname(),
                        "chain_id": chain.get_id(),
                        "coordinates": atom.get_coord()
                    }
                    atoms_data.append(atom_info)

    return atoms_data
