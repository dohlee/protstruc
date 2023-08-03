from typing import List, Tuple
import torch
import numpy as np
import pandas as pd

from biopandas.pdb import PandasPdb
from protstruc.alphabet import one2three
from protstruc.constants import MAX_N_ATOMS_PER_RESIDUE
from protstruc.general import (
    restype_to_heavyatom_names,
    non_standard_residue_substitutions,
    AA,
)


def _precompute_internal_index_map(pdb_df):
    ret = {}  # (chain_id, residue_number, insertion) -> internal_index

    pdb_df = pdb_df.drop_duplicates(subset=["chain_id", "residue_number", "insertion"])

    chain, idx, prev_residue_number = None, 0, None
    for r in pdb_df.to_records():
        if chain is None:
            chain = r.chain_id
        elif chain == r.chain_id:
            # missing residue exists
            if r.residue_number > prev_residue_number + 1:
                n_missing_residues = r.residue_number - prev_residue_number - 1
                idx += n_missing_residues + 1
            else:
                idx += 1
        else:  # chain transition
            chain = r.chain_id
            idx += 1

        ret[(r.chain_id, r.residue_number, r.insertion)] = idx
        prev_residue_number = r.residue_number

    return ret


def tidy_pdb(pdb_df: pd.DataFrame) -> pd.DataFrame:
    pdb_df["atom_name"] = pdb_df["atom_name"].replace(non_standard_residue_substitutions)
    return pdb_df


def pdb_df_to_xyz(
    pdb_df: pd.DataFrame,
) -> Tuple[torch.Tensor, torch.BoolTensor, torch.LongTensor, List[str]]:
    pdb_df = tidy_pdb(pdb_df)

    index_map = _precompute_internal_index_map(pdb_df)
    n_residues = max(index_map.values()) + 1

    atom_xyz = torch.ones(n_residues, MAX_N_ATOMS_PER_RESIDUE, 3) * torch.nan
    chain_idx = torch.ones(n_residues) * torch.nan

    curr_chain, curr_chain_idx = None, 0.0
    for r in pdb_df.to_records():
        residue_name = r.residue_name
        heavyatom_names = restype_to_heavyatom_names[AA[residue_name]]

        if curr_chain is None:
            curr_chain = r.chain_id
        if curr_chain != r.chain_id:
            curr_chain = r.chain_id
            curr_chain_idx += 1.0

        internal_idx = index_map[(r.chain_id, r.residue_number, r.insertion)]

        atom_idx = heavyatom_names.index(r.atom_name)
        atom_xyz[internal_idx, atom_idx] = torch.tensor([r.x_coord, r.y_coord, r.z_coord])
        chain_idx[internal_idx] = curr_chain_idx

    atom_mask = ~(torch.isnan(atom_xyz).any(dim=-1))

    _, chain_ids = pdb_df.chain_id.factorize()
    chain_ids = chain_ids.tolist()

    return atom_xyz, atom_mask, chain_idx, chain_ids


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

    atom_xyz, atom_mask, chain_idx, chain_ids = pdb_df_to_xyz(pdb_df)
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
