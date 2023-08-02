from typing import List, Tuple
import torch
import numpy as np

from biopandas.pdb import PandasPdb
from protstruc.alphabet import one2three
from protstruc.constants import MAX_N_ATOMS_PER_RESIDUE
from protstruc.general import (
    restype_to_heavyatom_names,
    non_standard_residue_substitutions,
    AA,
)


def count_residues(pdb_df):
    return pdb_df.groupby("chain_id").agg({"residue_number": "max"})["residue_number"].sum()


def pdb_to_xyz(
    filename: str,
) -> Tuple[torch.Tensor, torch.BoolTensor, torch.LongTensor, List[str]]:
    """Parse a PDB file and return a tensor containing 3D coordinates of atoms.

    Args:
        filename: Path to a PDB file.

    Returns:
        atom_xyz: A xyz coordinate tensor.
            Shape (n_residues, MAX_N_ATOMS_PER_RESIDUE, 3).
        atom_mask: A mask tensor. 1 if the corresponding atom exists, 0 otherwise.
            Shape (n_residues, MAX_N_ATOMS_PER_RESIDUE)
        chain_idx: A LongTensor containing chain indices per residue.
            Shape (n_residues,)
        chain_ids: A list of unique chain IDs in the order of integers appearing in the
            `chain_idx` tensor.

    Note:
        `MAX_N_ATOMS_PER_RESIDUE` is set to **15** by default.
    """
    pdb_df = PandasPdb().read_pdb(filename).df["ATOM"]
    pdb_df["atom_name"] = pdb_df["atom_name"].replace(non_standard_residue_substitutions)

    n_residues = count_residues(pdb_df)
    atom_xyz = torch.ones(n_residues, MAX_N_ATOMS_PER_RESIDUE, 3) * torch.nan
    chain_idx = torch.ones(n_residues) * torch.nan

    curr_chain, curr_chain_start, curr_residue_idx = None, 0, None
    curr_chain_idx = 0.0
    for r in pdb_df.to_records():
        residue_name = r.residue_name
        heavyatom_names = restype_to_heavyatom_names[AA[residue_name]]

        if curr_chain is None:
            curr_chain = r.chain_id
            curr_residue_idx = r.residue_number - 1
        elif curr_chain == r.chain_id:
            curr_residue_idx = curr_chain_start + r.residue_number - 1
        else:
            curr_chain = r.chain_id
            curr_chain_start = curr_residue_idx + 1
            curr_chain_idx += 1.0
            curr_residue_idx = curr_chain_start + r.residue_number - 1

        atom_idx = heavyatom_names.index(r.atom_name)
        atom_xyz[curr_residue_idx, atom_idx] = torch.tensor([r.x_coord, r.y_coord, r.z_coord])

        chain_idx[curr_residue_idx] = curr_chain_idx

    atom_mask = ~(torch.isnan(atom_xyz).any(dim=-1))

    _, chain_ids = pdb_df.chain_id.factorize()
    chain_ids = chain_ids.tolist()

    return atom_xyz, atom_mask, chain_idx, chain_ids


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
