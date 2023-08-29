from typing import List

import numpy as np

from protstruc.alphabet import one2three


def to_pdb(
    filename: str,
    coords: np.array,
    sequences: List[str],
    chain_ids: List[str],
    atoms: List[str] = ["N", "CA", "C", "O", "CB"],
):
    """Save coordinates to a PDB file.

    Args:
        filename (str): Path to the output PDB file.
        coords (np.array):
            Coordinates of shape (5, L, 3), where the first dimension denotes the atom type.
        atoms (List, optional): Defaults to ["N", "CA", "C", "O", "CB"].
    """
    with open(filename, "w") as outFile:
        coord_idx, line_idx = 0, 1

        for seq, chain_id in zip(sequences, chain_ids):
            residue_idx = 1
            for aa1 in seq:
                for atom_idx, atom in enumerate(atoms):
                    if atom == "CB" and aa1 == "G":
                        continue

                    aa3 = one2three[aa1]
                    x, y, z = coords[atom_idx, coord_idx, :]

                    line = f"ATOM  {line_idx:5d}  {atom:4s}{aa3} {chain_id}{residue_idx:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"  # noqa: E501

                    outFile.write(line)
                    line_idx += 1

                coord_idx += 1
                residue_idx += 1
