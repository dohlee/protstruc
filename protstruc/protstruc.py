import numpy as np
import torch
import torch.nn.functional as F

from typing import List, Union
from biopandas.pdb import PandasPdb
from scipy.spatial.distance import cdist
from collections import defaultdict

import protstruc.geometry as geom
from protstruc.constants import ideal
from protstruc.alphabet import three2one
from protstruc.constants import MAX_N_ATOMS_PER_RESIDUE
from protstruc.io import pdb_to_xyz

CC_BOND_LENGTH = 1.522
CB_CA_N_ANGLE = 1.927
CB_DIHEDRAL = -2.143

N_IDX, CA_IDX, C_IDX = 0, 1, 2


class StructureBatch:
    """A batch of protein structures.

    This class provides an interface to initialize from and represent a batch of protein structures
    with various types of representations:

    StructureBatch object can be initialized with:
        - A single PDB file or a list of PDB files `StructureBatch.from_pdb`
        - Backbone or full atom 3D coordinates `StructureBatch.from_xyz`
        - Dihedral angles `StructureBatch.from_dihedrals` (TODO)
    """

    def __init__(
        self,
        xyz: torch.Tensor,
        atom_mask: torch.BoolTensor = None,
        chain_idx: torch.Tensor = None,
        chain_ids: List[str] = None,
    ):
        if (chain_idx is not None and chain_ids is None) or (
            chain_idx is None and chain_ids is not None
        ):
            raise ValueError("Both `chain_idx` and `chain_ids` should be provided or None.")

        self.xyz = xyz
        self.atom_mask = atom_mask
        self.batch_size, self.n_residues, self.max_n_atoms_per_residue = self.xyz.shape[:3]

        if atom_mask is not None:
            self.residue_mask = atom_mask.any(dim=-1)
        else:
            self.residue_mask = torch.ones(self.batch_size, self.n_residues, dtype=torch.bool)

        if chain_idx is not None:
            for i, chidx in enumerate(chain_idx):
                msk = ~torch.isnan(chidx)
                assert (
                    chidx[msk].min() == 0
                ), f"Protein {i}: Chain index should start from zero"

            self.chain_idx = chain_idx
        else:
            bsz, n_max_res = self.xyz.shape[:2]
            self.chain_idx = np.zeros((bsz, n_max_res))

        self.chain_ids = chain_ids

    @classmethod
    def from_xyz(
        cls,
        xyz: Union[np.ndarray, torch.Tensor],
        atom_mask: Union[np.ndarray, torch.Tensor] = None,
        chain_idx: Union[np.ndarray, torch.Tensor] = None,
        chain_ids: List[List[str]] = None,
    ) -> "StructureBatch":
        """Initialize a StructureBatch from a 3D coordinate array.

        Examples:
            Initialize a `StructureBatch` object from a numpy array of atom 3D coordinates.
            >>> batch_size, n_max_res, n_max_atoms = 2, 10, 25
            >>> xyz = np.random.randn(batch_size, n_max_res, n_max_atoms, 3)
            >>> sb = StructureBatch.from_xyz(xyz)

        Args:
            xyz: Shape: (batch_size, num_residues, num_atoms, 3)
            atom_mask: Shape: (batch_size, num_residues, num_atoms, 3)
            chain_idx: Chain indices for each residue.
                Should be starting from zero. Defaults to None.
                Shape: (batch_size, num_residues)
            chain_ids: A list of unique chain IDs for each protein.

        Returns:
            StructureBatch: A StructureBatch object.
        """
        xyz = torch.from_numpy(xyz) if isinstance(xyz, np.ndarray) else xyz
        atom_mask = (
            torch.from_numpy(atom_mask) if isinstance(atom_mask, np.ndarray) else atom_mask
        )
        chain_idx = (
            torch.from_numpy(chain_idx) if isinstance(chain_idx, np.ndarray) else chain_idx
        )
        self = cls(xyz, atom_mask, chain_idx, chain_ids)

        return self

    @classmethod
    def from_pdb(cls, pdb_path: Union[str, List[str]]) -> "StructureBatch":
        """Initialize a StructureBatch from a PDB file or a list of PDB files.

        Examples:
            Initialize a `StructureBatch` object from a single PDB file,
            >>> pdb_path = '1a0a.pdb'
            >>> sb = StructureBatch.from_pdb(pdb_path)

            or with a list of PDB files.
            >>> pdb_paths = ['1a0a.pdb', '1a0b.pdb']
            >>> sb = StructureBatch.from_pdb(pdb_paths)

        Args:
            pdb_path: Path to a PDB file or a list of paths to PDB files.

        Returns:
            StructureBatch: A StructureBatch object.
        """
        # parse pdb file and get xyz coordinates
        pdb_path = pdb_path if isinstance(pdb_path, list) else [pdb_path]

        bsz = len(pdb_path)
        tmp_atom_xyz, tmp_atom_mask, tmp_chain_idx = [], [], []
        chain_ids = []
        for f in pdb_path:
            _atom_xyz, _atom_mask, _chain_idx, _chain_ids = pdb_to_xyz(f)
            tmp_atom_xyz.append(_atom_xyz)
            tmp_atom_mask.append(_atom_mask)
            tmp_chain_idx.append(_chain_idx)
            chain_ids.append(_chain_ids)

        max_n_residues = max([len(xyz) for xyz in tmp_atom_xyz])

        atom_xyz = torch.zeros(bsz, max_n_residues, MAX_N_ATOMS_PER_RESIDUE, 3)
        atom_mask = torch.zeros(bsz, max_n_residues, MAX_N_ATOMS_PER_RESIDUE)
        chain_idx = torch.zeros(bsz, max_n_residues)

        for i in range(bsz):
            _atom_xyz = tmp_atom_xyz[i]
            _atom_mask = tmp_atom_mask[i]
            _chain_idx = tmp_chain_idx[i]

            atom_xyz[i, : len(_atom_xyz)] = _atom_xyz
            atom_mask[i, : len(_atom_mask)] = _atom_mask
            chain_idx[i, : len(_chain_idx)] = _chain_idx

        self = cls(atom_xyz, atom_mask, chain_idx, chain_ids)
        return self

    @classmethod
    def from_dihedrals(
        cls,
        dihedrals: Union[np.ndarray, torch.Tensor],
        chain_idx: Union[np.ndarray, torch.Tensor] = None,
        chain_ids: List[List[str]] = None,
    ) -> "StructureBatch":
        """Initialize a StructureBatch from a dihedral angle array.

        Args:
            dihedrals: Shape: (batch_size, num_residues, num_dihedrals)
            chain_idx: Chain identifiers for each residue.
                Should be starting from zero. Defaults to None.
                Shape: (batch_size, num_residues)
            chain_ids: A list of unique chain IDs for each protein.
        """
        # TODO: Implement this
        pass

    def _parse_pdb(self, f: str) -> torch.Tensor:
        """Parse a PDB file and return the all-atom 3D coordinates."""
        pass

    def get_xyz(self):
        return self.xyz

    def get_chain_ids(self):
        return self.chain_ids

    def get_max_n_atoms_per_residue(self):
        return self.max_n_atoms_per_residue

    def _infill_chain_idx(self, chain_idx):
        """Infill the chain index tensor to fill in the gaps."""
        chain_idx = chain_idx.clone()
        for i in range(chain_idx.shape[0]):
            chain_idx[i] = self._infill_chain_idx_single(chain_idx[i])
        return chain_idx

    def _infill_chain_idx_single(self, chain_idx_single):
        """Infill the chain index tensor to fill in the gaps."""
        chain_idx_single = chain_idx_single.clone()
        for i in range(chain_idx_single.shape[0]):
            if torch.isnan(chain_idx_single[i]):
                chain_idx_single[i] = chain_idx_single[i - 1]
        return chain_idx_single

    def get_n_terminal_mask(self) -> torch.BoolTensor:
        """Return a boolean mask for the N-terminal residues.

        Returns:
            A boolean tensor denoting N-terminal residues. True if N-terminal.
                Shape: (batch_size, num_residues)
        """
        chain_idx_infilled = self._infill_chain_idx(self.chain_idx).float()

        padded = F.pad(chain_idx_infilled, (1, 0, 0, 0), mode="constant", value=torch.nan)
        return (padded[:, :-1] != padded[:, 1:]).bool() * self.residue_mask

    def get_c_terminal_mask(self) -> torch.BoolTensor:
        """Return a boolean mask for the C-terminal residues.

        Returns:
            A boolean tensor denoting C-terminal residues. True if C-terminal.
                Shape: (batch_size, num_residues)
        """
        chain_idx_infilled = self._infill_chain_idx(self.chain_idx).float()

        padded = F.pad(chain_idx_infilled, (0, 1, 0, 0), mode="constant", value=torch.nan)
        return (padded[:, :-1] != padded[:, 1:]).bool() * self.residue_mask

    def get_backbone_dihedrals(self) -> torch.FloatTensor:
        """Return the backbone dihedral angles.

        Returns:
            A tensor containing `phi`, `psi` and `omega` dihedral angles for each residue.
                Shape: (batch_size, num_residues, 3)
        """
        n_coords = self.xyz[:, :, N_IDX]
        ca_coords = self.xyz[:, :, CA_IDX]
        c_coords = self.xyz[:, :, C_IDX]

        nterm, cterm = self.get_n_terminal_mask(), self.get_c_terminal_mask()

        phi = geom.dihedral(
            c_coords[:, :-1], n_coords[:, 1:], ca_coords[:, 1:], c_coords[:, 1:]
        )
        phi = np.pad(phi, ((0, 0), (1, 0)), mode="constant", constant_values=0.0)
        phi[nterm] = 0.0

        psi = geom.dihedral(
            n_coords[:, :-1], ca_coords[:, :-1], c_coords[:, :-1], n_coords[:, 1:]
        )
        psi = np.pad(psi, ((0, 0), (0, 1)), mode="constant", constant_values=0.0)
        psi[cterm] = 0.0

        omega = geom.dihedral(
            ca_coords[:, :-1], c_coords[:, :-1], n_coords[:, 1:], ca_coords[:, 1:]
        )
        omega = np.pad(omega, ((0, 0), (0, 1)), mode="constant", constant_values=0.0)
        omega[cterm] = 0.0

        dihedrals = np.stack([phi, psi, omega], axis=-1)
        dihedral_mask = np.stack([nterm, cterm, cterm], axis=-1)

        return dihedrals, dihedral_mask

    def pairwise_distance_matrix(self) -> np.ndarray:
        """Return the pairwise distance matrix between residues.

        Returns:
            Shape: (batch_size, num_residues, num_residues, max_n_atoms_per_residue, max_n_atoms_per_residue)
        """
        return np.linalg.norm(
            self.xyz[:, :, None, :, None] - self.xyz[:, None, :, None, :], axis=-1
        )

    def inter_residue_dihedrals(self, use_cb=False):
        """Return the inter-residue dihedral angles.

        Args:
            use_cb (bool, optional): Use CB atom instead of CA. Defaults to False.

        Returns:
            Shape: (batch_size, num_residues, num_residues, 2)
        """
        n_coords = self.xyz[:, :, N_IDX]
        ca_coords = self.xyz[:, :, CA_IDX]
        c_coords = self.xyz[:, :, C_IDX]  # bsz, n_res, 3

        ret = {}

        phi = None
        psi = None


class AntibodyFvStructure:
    def __init__(
        self,
        pdb_path,
        impute_missing_atoms=True,
        heavy_chain_id="H",
        light_chain_id="L",
    ):
        self.df = PandasPdb().read_pdb(pdb_path).df["ATOM"]

        chain_id = self.df["chain_id"]
        _max_char_len = self.df["residue_number"].astype(str).map(len).max()
        residue_number = (
            self.df["residue_number"]
            .astype(str)
            .str.pad(_max_char_len, side="left", fillchar="0")
        )
        insertion = self.df["insertion"]
        self.df["residue_id"] = chain_id + residue_number + insertion

        self.coord = self.df[["x_coord", "y_coord", "z_coord"]].values

        atoms = ["N", "CA", "C", "O", "CB"]
        mask = self.df.atom_name.isin(atoms)
        df_piv = (
            self.df[mask]
            .pivot(
                columns="atom_name",
                index="residue_id",
                values=["x_coord", "y_coord", "z_coord"],
            )
            .swaplevel(0, 1, axis=1)
        )

        residue_ids = df_piv.index.values
        residues = (
            self.df.drop_duplicates("residue_id").set_index("residue_id").loc[residue_ids]
        )
        self.chain_ids = residues.chain_id.unique()
        self.sequences = defaultdict(list)
        for r in residues.to_records():
            self.sequences[r.chain_id].append(three2one[r.residue_name])
        self.sequences = {k: "".join(v) for k, v in self.sequences.items()}

        self.coord_per_atom = {}
        for atom in atoms:
            self.coord_per_atom[atom] = df_piv[atom].values

        if impute_missing_atoms:
            self.impute_cb_coord()

        self.heavy_chain_id = heavy_chain_id
        self.light_chain_id = light_chain_id

    def get_sequences(self):
        return [self.sequences[chain] for chain in self.chain_ids]

    def get_chain_ids(self):
        return self.chain_ids

    def get_heavy_chain_length(self):
        return self.df[self.df.chain_id == self.heavy_chain_id].residue_number.nunique()

    def get_light_chain_length(self):
        return self.df[self.df.chain_id == self.light_chain_id].residue_number.nunique()

    def valid_coord_mask(self, atom):
        return np.isfinite(self.coord_per_atom[atom]).all(axis=-1)

    def pdist(self, atom1="CA", atom2="CA"):
        c1 = self.coord_per_atom[atom1]
        c2 = self.coord_per_atom[atom2]

        return cdist(c1, c2)

    def get_seq(self, chain=None):
        if chain is None:
            residues = self.df.drop_duplicates("residue_id").residue_name.values
        else:
            tmp = self.df[self.df.chain_id == chain]
            residues = tmp.drop_duplicates("residue_id").residue_name.values

        return "".join(three2one[aa] for aa in residues)

    def impute_cb_coord(self):
        c = self.coord_per_atom["C"]
        n = self.coord_per_atom["N"]
        ca = self.coord_per_atom["CA"]

        cb_coords = geom.place_fourth_atom(c, n, ca, ideal.AB, ideal.NAB, ideal.BANC)

        to_fill = ~self.valid_coord_mask("CB")
        self.coord_per_atom["CB"][to_fill] = cb_coords[to_fill]

    def inter_residue_geometry(self, to_degree=False):
        """
        https://github.com/RosettaCommons/RoseTTAFold/blob/main/network/kinematics.py
        """
        ret = {}

        # Ca-Ca distance (Symmetric)
        ret["d_ca"] = self.pdist("CA", "CA")
        # Cb-Cb distance (Symmetric)
        ret["d_cb"] = self.pdist("CB", "CB")
        # N-O distance (Non-symmetric)
        ret["d_no"] = self.pdist("N", "O")
        # Ca-Cb-Cb'-Ca' dihedral (Symmetric)
        ret["omega"] = geom.dihedral(
            self.coord_per_atom["CA"][:, np.newaxis, :],
            self.coord_per_atom["CB"][:, np.newaxis, :],
            self.coord_per_atom["CB"][np.newaxis, :, :],
            self.coord_per_atom["CA"][np.newaxis, :, :],
            to_degree=to_degree,
        )
        # N-Ca-Cb-Cb' dihedral (Non-symmetric)
        ret["theta"] = geom.dihedral(
            self.coord_per_atom["N"][:, np.newaxis, :],
            self.coord_per_atom["CA"][:, np.newaxis, :],
            self.coord_per_atom["CB"][:, np.newaxis, :],
            self.coord_per_atom["CB"][np.newaxis, :, :],
            to_degree=to_degree,
        )
        # Ca-Cb-Cb' planar angle (Non-symmetric)
        ret["phi"] = geom.angle(
            self.coord_per_atom["CA"][:, np.newaxis, :],
            self.coord_per_atom["CB"][:, np.newaxis, :],
            self.coord_per_atom["CB"][np.newaxis, :, :],
            to_degree=to_degree,
        )

        return ret

    def totensor(self):
        pass
