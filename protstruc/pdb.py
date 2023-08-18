import torch
import pandas as pd

from biopandas.pdb import PandasPdb
from typing import List, Tuple, Union
from collections import defaultdict
from protstruc.constants import MAX_N_ATOMS_PER_RESIDUE
from protstruc.general import (
    restype_to_heavyatom_names,
    resindex_to_oneletter,
    non_standard_residue_substitutions,
    standard_aa_names,
    AA,
)


def tidy_pdb(pdb_df: pd.DataFrame) -> pd.DataFrame:
    # convert non-standard residues to standard residues
    pdb_df["residue_name"].replace(non_standard_residue_substitutions, inplace=True)

    # try discard non-standard residues
    # hopefully this will discard non-peptide chains, too
    _mask = pdb_df["residue_name"].isin(standard_aa_names)
    pdb_df = pdb_df[_mask].reset_index(drop=True)

    return pdb_df


def _always_list(x):
    if isinstance(x, list):
        return x
    else:
        return [x]


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
        pdb_df,
        heavy_chain_id,
        light_chain_id,
        antigen_chain_ids: List[str] = None,
        keep_fv_only: bool = False,
    ):
        self.pdb_df = pdb_df

        self.heavy_chain_id = heavy_chain_id
        self.light_chain_id = light_chain_id
        self.antigen_chain_ids = antigen_chain_ids
        self.keep_fv_only = keep_fv_only

        self._retain_only_relevant_chains()
        if self.keep_fv_only:
            self._retain_only_fv()

        self._compute_lookup()
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
        pdb_df = PandasPdb().read_pdb(fp).df["ATOM"]
        pdb_df = tidy_pdb(pdb_df)
        antigen_chain_ids = _always_list(antigen_chain_ids)
        return cls(
            pdb_df, heavy_chain_id, light_chain_id, antigen_chain_ids, keep_fv_only
        )

    def _retain_only_relevant_chains(self):
        if self.antigen_chain_ids is None:
            mask = self.pdb_df.chain_id.isin([self.heavy_chain_id, self.light_chain_id])
        else:
            mask = self.pdb_df.chain_id.isin(
                [self.heavy_chain_id, self.light_chain_id] + self.antigen_chain_ids
            )

        self.pdb_df = self.pdb_df[mask]

    def _retain_only_fv(self):
        mask_heavy = self.pdb_df.chain_id == self.heavy_chain_id
        mask_light = self.pdb_df.chain_id == self.light_chain_id

        mask_vh = self.pdb_df.residue_number.between(*self.fv_heavy_range)
        mask_vl = self.pdb_df.residue_number.between(*self.fv_light_range)

        if self.antigen_chain_ids is None:
            mask = (mask_heavy & mask_vh) | (mask_light & mask_vl)
        else:
            mask_ag = self.pdb_df.chain_id.isin(self.antigen_chain_ids)  # antigen
            mask = (mask_heavy & mask_vh) | (mask_light & mask_vl) | mask_ag

        self.pdb_df = self.pdb_df[mask]

    def _compute_lookup(self):
        dedup_subset = ["chain_id", "residue_number", "insertion"]
        pdb_df_dedup = self.pdb_df.drop_duplicates(subset=dedup_subset)

        self._lookup = defaultdict(list)
        idx = 0

        curr_chain_id, curr_residue_number, curr_insertion = None, None, None
        for r in pdb_df_dedup.to_records():
            if curr_chain_id is None or curr_chain_id != r.chain_id:
                curr_chain_id = r.chain_id
                curr_residue_number = r.residue_number
                curr_insertion = r.insertion

            while curr_residue_number + 1 < r.residue_number:
                self._lookup["internal_idx"].append(idx)
                self._lookup["chain_id"].append(curr_chain_id)
                self._lookup["residue_number"].append(curr_residue_number + 1)
                self._lookup["insertion"].append(curr_insertion)
                self._lookup["residue_name"].append("UNK")
                self._lookup["oneletter"].append(AA["UNK"].oneletter())
                curr_residue_number += 1
                idx += 1

            self._lookup["internal_idx"].append(idx)
            self._lookup["chain_id"].append(r.chain_id)
            self._lookup["residue_number"].append(r.residue_number)
            self._lookup["insertion"].append(r.insertion)
            self._lookup["residue_name"].append(r.residue_name)
            self._lookup["oneletter"].append(AA[r.residue_name].oneletter())

            curr_chain_id = r.chain_id
            curr_residue_number = r.residue_number
            curr_insertion = r.insertion

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

        for r in self.pdb_df.to_records():
            cri = (r.chain_id, r.residue_number, r.insertion)
            internal_idx = self.cri2idx[cri]

            heavyatom_names_for_residue = restype_to_heavyatom_names[AA[r.residue_name]]
            atom_idx = heavyatom_names_for_residue.index(r.atom_name)

            self.atom_xyz[internal_idx, atom_idx] = torch.tensor(
                [r.x_coord, r.y_coord, r.z_coord]
            )
            self.atom_xyz_mask[internal_idx, atom_idx] = True

    def get_heavy_chain_pdb_df(self):
        return self.pdb_df.query(f"chain_id == '{self.heavy_chain_id}'")

    def get_light_chain_pdb_df(self):
        return self.pdb_df.query(f"chain_id == '{self.light_chain_id}'")

    def get_antigen_chains_pdb_df(self):
        if self.antigen_chain_ids is None:
            return None
        else:
            return self.pdb_df.query(f"chain_id in {self.antigen_chain_ids}")

    def get_atom_xyz(self) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        return self.atom_xyz, self.atom_xyz_mask

    def get_chain_idx(self, return_chain_ids=True) -> torch.LongTensor:
        return torch.tensor(self._lookup.chain_idx.values).long()

    def get_chain_ids(self) -> List[str]:
        return list(self._lookup.chain_id.unique())

    def get_residue_idx(self) -> torch.LongTensor:
        return torch.tensor(self._lookup.internal_idx.values).long()

    def get_seq_idx(self) -> torch.LongTensor:
        return torch.tensor(
            [AA[c].value for c in self._lookup.residue_name.values]
        ).long()

    def get_seq(self) -> str:
        return "".join(self._lookup.oneletter.values)

    def get_seq_dict(self):
        seq_dict = {}

        for chain_id in self.get_chain_ids():
            seq = "".join(
                self._lookup[self._lookup.chain_id == chain_id].oneletter.values
            )
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
