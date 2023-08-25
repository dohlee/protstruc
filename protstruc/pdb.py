import torch
import numpy as np
import pandas as pd

from biopandas.pdb import PandasPdb
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from typing import List, Tuple, Union
from collections import defaultdict
from protstruc.constants import MAX_N_ATOMS_PER_RESIDUE
from protstruc.general import (
    restype_to_heavyatom_names,
    non_standard_residue_substitutions,
    standard_aa_names,
    standard_heavy_atom_names,
    AA,
)


def _always_list(x):
    if isinstance(x, list):
        return x
    else:
        return [x]


def tidy_structure(structure: struc.AtomArray) -> struc.AtomArray:
    # convert non-standard residues to standard residues
    standardize = non_standard_residue_substitutions
    standardized = [standardize.get(r, r) for r in structure.res_name]
    structure.res_name = standardized

    # retain only standard residues names
    # hopefully this will discard non-peptide chains, too
    mask = struc.filter_canonical_amino_acids(structure)
    structure = structure[mask]

    # retain only heavy atoms with standard names
    # hopefully this will discard all the hydrogen atoms
    mask = np.isin(structure.atom_name, standard_heavy_atom_names)
    structure = structure[mask]

    return structure


def tidy_pdb(pdb_df: pd.DataFrame) -> pd.DataFrame:
    # convert non-standard residues to standard residues
    pdb_df["residue_name"].replace(non_standard_residue_substitutions, inplace=True)

    # try discard non-standard residues
    # hopefully this will discard non-peptide chains, too
    _mask = pdb_df["residue_name"].isin(standard_aa_names)
    pdb_df = pdb_df[_mask].reset_index(drop=True)

    return pdb_df


class ChothiaAntibodyPDB:
    fv_heavy_range = (1, 113)
    fv_light_range = (1, 106)
    h1_range = (26, 32)
    h2_range = (52, 56)
    h3_range = (95, 102)
    l1_range = (24, 34)
    l2_range = (50, 56)
    l3_range = (89, 97)

    def __init__(
        self,
        structure: struc.AtomArray,
        heavy_chain_id,
        light_chain_id,
        antigen_chain_ids: List[str] = None,
        keep_fv_only: bool = False,
    ):
        self.structure = structure

        self.heavy_chain_id = heavy_chain_id
        self.light_chain_id = light_chain_id
        self.antigen_chain_ids = antigen_chain_ids
        self.keep_fv_only = keep_fv_only

        self._retain_only_relevant_chains()
        if self.keep_fv_only:
            self._retain_only_fv()

        self._initialize_lookup()
        self.n_residues = len(self._lookup)

        self._compute_atom_xyz()

    @classmethod
    def read_pdb(
        cls,
        fp: str,
        heavy_chain_id: str,
        light_chain_id: str,
        antigen_chain_ids: Union[str, List[str]] = None,
        keep_fv_only: bool = False,
    ) -> "ChothiaAntibodyPDB":
        structure = PDBFile.read(fp).get_structure(model=1)  # take the first model
        structure = tidy_structure(structure)

        antigen_chain_ids = _always_list(antigen_chain_ids)
        return cls(
            structure, heavy_chain_id, light_chain_id, antigen_chain_ids, keep_fv_only
        )

    def _retain_only_relevant_chains(self):
        target_chains = [self.heavy_chain_id, self.light_chain_id]

        if self.antigen_chain_ids is not None:
            target_chains += self.antigen_chain_ids

        mask = np.isin(self.structure.chain_id, target_chains)
        self.structure = self.structure[mask]

    def _retain_only_fv(self):
        hmin, hmax = self.fv_heavy_range
        lmin, lmax = self.fv_light_range

        mask_heavy = self.structure.chain_id == self.heavy_chain_id
        mask_light = self.structure.chain_id == self.light_chain_id
        mask_vh = (hmin <= self.structure.res_id) & (self.structure.res_id <= hmax)
        mask_vl = (lmin <= self.structure.res_id) & (self.structure.res_id <= lmax)

        mask = (mask_heavy & mask_vh) | (mask_light & mask_vl)

        if self.antigen_chain_ids is not None:
            mask_ag = np.isin(
                self.structure.chain_id, self.antigen_chain_ids
            )  # antigen
            mask |= mask_ag

        self.structure = self.structure[mask]

    def _fill_lookup(
        self, chain_id, residue_number, insertion, threeletter, oneletter, idx
    ):
        """Fill the lookup table with a new entry."""
        self._lookup["internal_idx"].append(idx)
        self._lookup["chain_id"].append(chain_id)
        self._lookup["residue_number"].append(residue_number)
        self._lookup["insertion"].append(insertion)
        self._lookup["threeletter"].append(threeletter)
        self._lookup["oneletter"].append(oneletter)

    def _initialize_lookup(self):
        """Initialize a lookup table mapping (chain_id, residue_number, insertion) to
        internal index.
        """
        self._lookup = defaultdict(list)

        idx = 0
        curr_chain_id, curr_residue_number = None, None
        for r in struc.residue_iter(self.structure):
            chain_id = r.chain_id[0]
            residue_number = r.res_id[1]
            insertion = r.ins_code[0]
            threeletter = r.res_name[0]
            oneletter = AA[threeletter].oneletter()

            if curr_chain_id is None or curr_chain_id != chain_id:
                curr_chain_id = chain_id
                curr_residue_number = residue_number

            # fill the missing in-between residues with a dummy residue internally
            while curr_residue_number + 1 < residue_number:
                self._fill_lookup(
                    curr_chain_id,
                    curr_residue_number + 1,
                    insertion,
                    "UNK",
                    AA["UNK"].oneletter(),
                    idx,
                )
                curr_residue_number += 1
                idx += 1

            self._fill_lookup(
                chain_id, residue_number, insertion, threeletter, oneletter, idx
            )

            curr_chain_id = chain_id
            curr_residue_number = residue_number
            idx += 1

        self._lookup = pd.DataFrame(self._lookup)
        self._lookup["chain_idx"] = pd.Categorical(self._lookup.chain_id).codes

        # compute internal index mapping
        self.cri2idx = {}
        for r in self._lookup.to_records():
            self.cri2idx[(r.chain_id, r.residue_number, r.insertion)] = r.internal_idx

    def _compute_atom_xyz(self):
        self.atom_xyz = (
            torch.ones(self.n_residues, MAX_N_ATOMS_PER_RESIDUE, 3) * torch.nan
        )
        self.atom_xyz_mask = torch.zeros(
            self.n_residues, MAX_N_ATOMS_PER_RESIDUE, dtype=torch.bool
        )

        for r in struc.residue_iter(self.structure):
            cri = (r.chain_id[0], r.res_id[1], r.ins_code[0])
            internal_idx = self.cri2idx[cri]

            residue_name = r.res_name[0]
            heavyatom_names_for_residue = restype_to_heavyatom_names[AA[residue_name]]

            for atom in r:
                atom_idx = heavyatom_names_for_residue.index(atom.atom_name)

                self.atom_xyz[internal_idx, atom_idx] = torch.tensor(atom.coord)
                self.atom_xyz_mask[internal_idx, atom_idx] = True

    def get_heavy_chain_structure(self):
        mask = self.structure.chain_id == self.heavy_chain_id
        return self.structure[mask]

    def get_light_chain_structure(self):
        mask = self.structure.chain_id == self.light_chain_id
        return self.structure[mask]

    def get_antigen_chains_structure(self):
        if self.antigen_chain_ids is None:
            return None
        else:
            mask = np.isin(self.structure.chain_id, self.antigen_chain_ids)
            return self.structure[mask]

    def get_atom_xyz(self) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        return self.atom_xyz, self.atom_xyz_mask

    def get_chain_idx(self) -> torch.LongTensor:
        return torch.tensor(self._lookup.chain_idx.values).long()

    def get_chain_ids(self) -> List[str]:
        return list(self._lookup.chain_id.unique())

    def get_residue_idx(self) -> torch.LongTensor:
        return torch.tensor(self._lookup.internal_idx.values).long()

    def get_seq_idx(self) -> torch.LongTensor:
        residue_names = self._lookup.residue_name.values
        return torch.tensor([AA[c].value for c in residue_names]).long()

    def get_seq(self) -> str:
        return "".join(self._lookup.oneletter.values)

    def get_seq_dict(self):
        seq_dict = {}
        for chain_id in self.get_chain_ids():
            selected = self._lookup[self._lookup.chain_id == chain_id]
            seq = "".join(selected.oneletter.values)
            seq_dict[chain_id] = seq

        return seq_dict

    def get_heavy_chain_mask(self) -> torch.BoolTensor:
        return torch.Tensor(self._lookup.chain_id == self.heavy_chain_id).bool()

    def get_light_chain_mask(self) -> torch.BoolTensor:
        return torch.Tensor(self._lookup.chain_id == self.light_chain_id).bool()

    def get_antigen_mask(self) -> torch.BoolTensor:
        return torch.Tensor(self._lookup.chain_id.isin(self.antigen_chain_ids)).bool()

    def get_fv_mask(self) -> torch.BoolTensor:
        mask_heavy = self.get_heavy_chain_mask()
        mask_light = self.get_light_chain_mask()
        mask_vh = self._lookup.residue_number.between(*self.fv_heavy_range)
        mask_vl = self._lookup.residue_number.between(*self.fv_light_range)
        return torch.tensor((mask_heavy & mask_vh) | (mask_light & mask_vl)).bool()

    def get_cdr_mask(self, subset: Union[str, List[str]] = None) -> torch.BoolTensor:
        subset = _always_list(subset)
        subset = subset if subset is None else [x.upper() for x in subset]

        chain_masks = {
            "H": self.get_heavy_chain_mask(),
            "L": self.get_light_chain_mask(),
        }
        masks = {
            "H1": torch.tensor(
                self._lookup.residue_number.between(*self.h1_range)
            ).bool(),
            "H2": torch.tensor(
                self._lookup.residue_number.between(*self.h2_range)
            ).bool(),
            "H3": torch.tensor(
                self._lookup.residue_number.between(*self.h3_range)
            ).bool(),
            "L1": torch.tensor(
                self._lookup.residue_number.between(*self.l1_range)
            ).bool(),
            "L2": torch.tensor(
                self._lookup.residue_number.between(*self.l2_range)
            ).bool(),
            "L3": torch.tensor(
                self._lookup.residue_number.between(*self.l3_range)
            ).bool(),
        }
        mask = torch.zeros(self.n_residues).bool()

        if subset is None:
            for s in masks:
                mask |= chain_masks[s[0]] & masks[s]
        else:
            for s in subset:
                mask |= chain_masks[s[0]] & masks[s]

        return mask
