import numpy as np
import torch
import torch.nn.functional as F

from typing import List, Dict, Union, Tuple
from biopandas.pdb import PandasPdb
from scipy.spatial.distance import cdist
from collections import defaultdict
from einops import rearrange, repeat

import protstruc.geometry as geom
from protstruc.constants import ideal
from protstruc.alphabet import three2one
from protstruc.constants import MAX_N_ATOMS_PER_RESIDUE
from protstruc.io import pdb_to_xyz, pdb_df_to_xyz

CC_BOND_LENGTH = 1.522
CB_CA_N_ANGLE = 1.927
CB_DIHEDRAL = -2.143

N_IDX, CA_IDX, C_IDX, O_IDX, CB_IDX = 0, 1, 2, 3, 4
atom2idx = {"N": N_IDX, "CA": CA_IDX, "C": C_IDX, "O": O_IDX, "CB": CB_IDX}


def _always_tensor(x):
    return torch.from_numpy(x) if isinstance(x, np.ndarray) else x


def _always_list(x):
    return x if isinstance(x, list) else [x]


class StructureBatch:
    """A batch of protein structures.

    This class provides an interface to initialize from and represent a batch of protein structures
    with various types of representations:

    StructureBatch object can be initialized with:
        - A single PDB file or a list of PDB files `StructureBatch.from_pdb`
        - A pdb identifier or a list of PDB identifiers `StructureBatch.from_pdb_id`
        - Backbone or full atom 3D coordinates `StructureBatch.from_xyz`
        - Backbone orientation and translations `StructureBatch.from_backbone_orientations_translations`
        - Dihedral angles `StructureBatch.from_dihedrals` (TODO)
    """

    def __init__(
        self,
        xyz: torch.Tensor,
        atom_mask: torch.BoolTensor = None,
        chain_idx: torch.Tensor = None,
        chain_ids: List[str] = None,
        seq: List[Dict[str, str]] = None,
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
            self.chain_idx = torch.zeros(bsz, n_max_res)

        self.chain_ids = chain_ids
        self.seq = seq

        # assumes that the input is not standardized yet
        self._standardized = False

    @classmethod
    def from_xyz(
        cls,
        xyz: Union[np.ndarray, torch.Tensor],
        atom_mask: Union[np.ndarray, torch.Tensor] = None,
        chain_idx: Union[np.ndarray, torch.Tensor] = None,
        chain_ids: List[List[str]] = None,
        seq: List[Dict[str, str]] = None,
    ) -> "StructureBatch":
        """Initialize a `StructureBatch` from a 3D atom coordinate array.

        Examples:
            Initialize a `StructureBatch` object from a numpy array of 3D atom coordinates.
            >>> batch_size, n_max_res, n_max_atoms = 2, 10, 25
            >>> xyz = np.random.randn(batch_size, n_max_res, n_max_atoms, 3)
            >>> sb = StructureBatch.from_xyz(xyz)

        Args:
            xyz: Shape: (batch_size, num_residues, num_atoms, 3)
            atom_mask: Shape: (batch_size, num_residues, num_atoms)
            chain_idx: Chain indices for each residue.
                Should be starting from zero. Defaults to None.
                Shape: (batch_size, num_residues)
            chain_ids: A list of unique chain IDs for each protein.
            seq: A list of dictionaries containing sequence information for each chain.

        Returns:
            StructureBatch: A StructureBatch object.
        """
        xyz = _always_tensor(xyz)
        atom_mask = _always_tensor(atom_mask)
        chain_idx = _always_tensor(chain_idx)

        self = cls(xyz, atom_mask, chain_idx, chain_ids, seq)
        return self

    @classmethod
    def from_pdb(cls, pdb_path: Union[str, List[str]]) -> "StructureBatch":
        """Initialize a `StructureBatch` from a PDB file or a list of PDB files.

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
        pdb_path = _always_list(pdb_path)
        bsz = len(pdb_path)

        tmp_atom_xyz, tmp_atom_mask, tmp_chain_idx, seq = [], [], [], []
        chain_ids = []
        for f in pdb_path:
            _atom_xyz, _atom_mask, _chain_idx, _chain_ids, _seq_dict = pdb_to_xyz(f)
            tmp_atom_xyz.append(_atom_xyz)
            tmp_atom_mask.append(_atom_mask)
            tmp_chain_idx.append(_chain_idx)
            chain_ids.append(_chain_ids)
            seq.append(_seq_dict)

        max_n_residues = max([len(xyz) for xyz in tmp_atom_xyz])

        atom_xyz = torch.zeros(bsz, max_n_residues, MAX_N_ATOMS_PER_RESIDUE, 3)
        atom_mask = torch.zeros(bsz, max_n_residues, MAX_N_ATOMS_PER_RESIDUE)
        chain_idx = torch.ones(bsz, max_n_residues) * torch.nan

        for i in range(bsz):
            _atom_xyz = tmp_atom_xyz[i]
            _atom_mask = tmp_atom_mask[i]
            _chain_idx = tmp_chain_idx[i]

            atom_xyz[i, : len(_atom_xyz)] = _atom_xyz
            atom_mask[i, : len(_atom_mask)] = _atom_mask
            chain_idx[i, : len(_chain_idx)] = _chain_idx

        self = cls(atom_xyz, atom_mask, chain_idx, chain_ids, seq)
        return self

    @classmethod
    def from_pdb_id(
        cls,
        pdb_id: Union[str, List[str]],
    ) -> "StructureBatch":
        """Initialize a `StructureBatch` from a PDB ID or a list of PDB IDs.

        Examples:
            >>> pdb_id = "2ZIL"  # Human lysozyme
            >>> sb = StructureBatch.from_pdb_id(pdb_id)
            >>> xyz = sb.get_xyz()
            >>> xyz.shape
            torch.Size([1, 130, 15, 3])
            >>> dihedrals, dihedral_mask = sb.backbone_dihedrals()
            >>> dihedrals.shape
            torch.Size([1, 130, 3])
            >>> dihedral_mask.shape
            torch.Size([1, 130, 3])
            >>> dihedral_mask.sum()
            tensor(3)

        Args:
            pdb_id: A PDB identifier or a list of PDB identifiers.

        Returns:
            StructureBatch: A StructureBatch object.
        """
        pdb_id = _always_list(pdb_id)
        bsz = len(pdb_id)

        tmp_atom_xyz, tmp_atom_mask, tmp_chain_idx, seq = [], [], [], []
        chain_ids = []
        for id in pdb_id:
            pdb_df = PandasPdb().fetch_pdb(id).df["ATOM"]
            _atom_xyz, _atom_mask, _chain_idx, _chain_ids, _seq_dict = pdb_df_to_xyz(pdb_df)

            tmp_atom_xyz.append(_atom_xyz)
            tmp_atom_mask.append(_atom_mask)
            tmp_chain_idx.append(_chain_idx)
            chain_ids.append(_chain_ids)
            seq.append(_seq_dict)

        max_n_residues = max([len(xyz) for xyz in tmp_atom_xyz])

        atom_xyz = torch.zeros(bsz, max_n_residues, MAX_N_ATOMS_PER_RESIDUE, 3)
        atom_mask = torch.zeros(bsz, max_n_residues, MAX_N_ATOMS_PER_RESIDUE)
        chain_idx = torch.ones(bsz, max_n_residues) * torch.nan

        for i in range(bsz):
            _atom_xyz = tmp_atom_xyz[i]
            _atom_mask = tmp_atom_mask[i]
            _chain_idx = tmp_chain_idx[i]

            atom_xyz[i, : len(_atom_xyz)] = _atom_xyz
            atom_mask[i, : len(_atom_mask)] = _atom_mask
            chain_idx[i, : len(_chain_idx)] = _chain_idx

        self = cls(atom_xyz, atom_mask, chain_idx, chain_ids, seq)
        return self

    @classmethod
    def from_backbone_orientations_translations(
        cls,
        orientations: Union[np.ndarray, torch.Tensor],
        translations: Union[np.ndarray, torch.Tensor],
        chain_idx: Union[np.ndarray, torch.Tensor] = None,
        chain_ids: List[List[str]] = None,
        seq: List[Dict[str, str]] = None,
        include_cb: bool = False,
    ) -> "StructureBatch":
        """Initialize a StructureBatch from an array of backbone orientations and translations.

        Args:
            orientations: Shape: (batch_size, num_residues, 3, 3)
            translations: Shape: (batch_size, num_residues, 3)
            chain_idx: Chain identifiers for each residue. Should be starting from zero.
                Defaults to None. Shape: (batch_size, num_residues)
            chain_ids: A list of unique chain IDs for each protein.
            seq: A list of dictionaries containing sequence information for each chain.
            include_cb: Whether to include CB atoms when initializing. Defaults to False.

        Returns:
            StructureBatch: A StructureBatch object.
        """
        batch_size, n_residues = orientations.shape[:2]

        # determine ideal backbone coordinates (b n a 3) or (b n a 4)
        ideal_backbone = geom.ideal_backbone_coordinates((batch_size, n_residues), include_cb)
        n_atoms = ideal_backbone.shape[2]

        # rotate and translate ideal backbone coordinates by orientations
        orientations = repeat(orientations, "b n i j -> b n a i j", a=n_atoms)

        atom_xyz = torch.einsum("bnaij,bnaj->bnai", orientations, ideal_backbone)
        atom_xyz = atom_xyz + rearrange(translations, "b n i -> b n () i")

        atom_mask = torch.ones_like(atom_xyz[..., 0])

        self = cls(atom_xyz, atom_mask, chain_idx, chain_ids, seq)
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

    def get_xyz(self):
        return self.xyz

    def get_local_xyz(self) -> torch.Tensor:
        """Return the coordinates of each atom in the local frame of each residue.

        Returns:
            local_xyz: Shape: (batch_size, num_residues, num_atoms_per_residue, 3)
        """
        n_atoms = self.max_n_atoms_per_residue

        orientation = self.backbone_orientations()  # b n 3 3
        orientation = repeat(orientation, "b n i j -> b n a i j", a=n_atoms)

        xyz = self.xyz  # b n a 3

        local_xyz = torch.einsum("bnaji,bnaj->bnai", orientation, xyz)
        local_xyz = local_xyz - xyz[:, :, atom2idx["CA"]].unsqueeze(-2)
        return local_xyz

    def get_atom_mask(self):
        return self.atom_mask

    def get_chain_idx(self):
        return self.chain_idx

    def get_chain_ids(self):
        return self.chain_ids

    def get_seq(self) -> List[Dict[str, str]]:
        """Return the amino acid sequence of proteins.

        Returns:
            seq_dict: A list of dictionaries containing sequence information for each chain.
        """
        return self.seq

    def get_total_lengths(self) -> torch.LongTensor:
        """Return the total sum of chain lengths for each protein.

        Note:
            This **counts the number of missing residues in the middle of a chain**,
            but **does not count the missing residues at the beginning and end of a chain**.

        Returns:
            total_lengths: A tensor containing the total length of each protein.
                Shape: (batch_size,)
        """
        return self.residue_mask.cumsum(dim=1).argmax(dim=1) + 1

    def get_max_n_atoms_per_residue(self):
        return self.max_n_atoms_per_residue

    def get_n_terminal_mask(self) -> torch.BoolTensor:
        """Return a boolean mask for the N-terminal residues.

        Returns:
            A boolean tensor denoting N-terminal residues. `True` if N-terminal.
                Shape: (batch_size, num_residues)
        """
        padded = F.pad(self.chain_idx, (1, 0), mode="constant", value=torch.nan)
        return (padded[:, :-1] != padded[:, 1:]).bool() * self.residue_mask

    def get_c_terminal_mask(self) -> torch.BoolTensor:
        """Return a boolean mask for the C-terminal residues.

        Returns:
            A boolean tensor denoting C-terminal residues. `True` if C-terminal.
                Shape: (batch_size, num_residues)
        """
        padded = F.pad(self.chain_idx, (0, 1), mode="constant", value=torch.nan)
        return (padded[:, :-1] != padded[:, 1:]).bool() * self.residue_mask

    def pairwise_distance_matrix(self) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
        """Return the all-atom pairwise pairwise distance matrix between residues.

        Info:
            Distances are measured in **Angstroms**.

        Examples:
            `dist[:, :, :, 1, 1]` will give pairwise alpha-carbon distance matrix between residues,
            as the index `1` corresponds to the alpha-carbon atom.
            ```python
            >>> structure_batch = StructureBatch.from_pdb("1a8o.pdb")
            >>> dist = structure_batch.pairwise_distance_matrix()
            >>> ca_dist = dist[:, :, :, 1, 1]  # 1 = CA_IDX
            ```
        Returns:
            dist: A tensor containing an all-atom pairwise distance matrix for each pair of residues.
                A distance between atom `a` of residue `i` and atom `b` of residue `j` of protein
                at index `batch_idx` is given by `dist[batch_idx, i, j, a, b]`.
                Shape: (batch_size, num_residues, num_residues, max_n_atoms_per_residue, max_n_atoms_per_residue)
            dist_mask: A boolean tensor denoting which distances are valid.
                Shape: (batch_size, num_residues, num_residues, max_n_atoms_per_residue, max_n_atoms_per_residue)
        """
        dist = torch.norm(
            self.xyz[:, :, None, :, None] - self.xyz[:, None, :, None, :], dim=-1
        )

        dist_mask = self.atom_mask[:, :, None, :, None] * self.atom_mask[:, None, :, None, :]
        return dist, dist_mask

    def backbone_dihedrals(self) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Return the backbone dihedral angles phi, psi and omega for each residue.

        Info:
            Dihedral angles are measured in **radians** and are in the range `[-pi, pi]`.

            For a quick reminder of the definition of the dihedral angles, refer to the following image:
            ![Dihedral](https://i.imgur.com/fZ0Sx3V.png)

            Source: [Fabian Fuchs](https://fabianfuchsml.github.io/alphafold2/)

        Note:
            `phi` angles are not defined for the first residue (it needs a predecessor)
            and `psi` and `omega` angles are not defined for the last residue (they need successors).
            Those invalid angles can be filtered using the `dihedral_mask` tensor returned from the method.

        Warning:
            Dihedral angles involving the residues at the chain breaks are not handled correctly for now.

        Returns:
            dihedrals: A tensor containing `phi`, `psi` and `omega` dihedral angles for each residue.
                Shape: (batch_size, num_residues, 3)
            dihedral_mask: A tensor containing a boolean mask for the dihedral angles.
                `True` if the corresponding dihedral angle is defined, `False` otherwise.
                Shape: (batch_size, num_residues, 3)
        """
        n_coords = self.xyz[:, :, N_IDX]
        ca_coords = self.xyz[:, :, CA_IDX]
        c_coords = self.xyz[:, :, C_IDX]

        nterm, cterm = self.get_n_terminal_mask(), self.get_c_terminal_mask()

        phi = geom.dihedral(
            c_coords[:, :-1], n_coords[:, 1:], ca_coords[:, 1:], c_coords[:, 1:]
        )
        phi = F.pad(phi, (1, 0, 0, 0), mode="constant", value=0.0)
        phi[nterm] = 0.0

        psi = geom.dihedral(
            n_coords[:, :-1], ca_coords[:, :-1], c_coords[:, :-1], n_coords[:, 1:]
        )
        psi = F.pad(psi, (0, 1, 0, 0), mode="constant", value=0.0)
        psi[cterm] = 0.0

        omega = geom.dihedral(
            ca_coords[:, :-1], c_coords[:, :-1], n_coords[:, 1:], ca_coords[:, 1:]
        )
        omega = F.pad(omega, (0, 1, 0, 0), mode="constant", value=0.0)
        omega[cterm] = 0.0

        dihedrals = torch.stack([phi, psi, omega], axis=-1)

        dihedral_mask = ~torch.stack([nterm, cterm, cterm], axis=-1)
        dihedral_mask *= self.residue_mask[:, :, None]

        return dihedrals, dihedral_mask

    def backbone_orientations(
        self, a1: str = "N", a2: str = "CA", a3: str = "C"
    ) -> torch.FloatTensor:
        """Return the orientation of the backbone for each residue.

        Args:
            a1: First atom used to determine backbone orientation.
                Defaults to 'N'.
            a2: Second atom used to determine backbone orientation.
                Defaults to 'CA'.
            a3: Third atom used to determine backbone orientation.
                Defaults to 'C'.

        Note:
            The backbone orientations are determined by using Gram-Schmidt
            orthogonalization on the vectors `a3 - a2` and `a1 - a2`.
            Note that `a3 - a2` forms the first basis, and `a1 - a2` - proj_{a3 - a2}(a1 - a2)
            forms the second basis. The third basis is formed by taking the cross product of the
            first and second basis vectors.

        Returns:
            bb_orientations: A tensor containing the local reference backbone
                orientation for each residue.
        """
        a1_coords = self.xyz[:, :, atom2idx[a1]]
        a2_coords = self.xyz[:, :, atom2idx[a2]]
        a3_coords = self.xyz[:, :, atom2idx[a3]]

        return geom.gram_schmidt(a1_coords, a2_coords, a3_coords)

    def backbone_translations(self, atom: str = "CA") -> torch.FloatTensor:
        """Return the coordinate (translation) of a given backbone atom for each residue.

        Note:
            Reference atom is set to the **alpha-carbon (CA)** by default.

        Args:
            atom: Type of atom used to determine backbone translation.
                Defaults to 'CA'.

        Returns:
            bb_translations: xyz coordinates (translations) of a specified backbone atoms.
                Shape: (batch_size, num_residues, 3)
        """
        return self.xyz[:, :, atom2idx[atom]]

    def pairwise_dihedrals(self, atoms_i: List[str], atoms_j: List[str]) -> torch.FloatTensor:
        """Return a matrix representing a pairwise dihedral angle between residues defined by
        two sets of atoms, one for each side of the residue.

        Args:
            atoms_i: List of atoms to be used for the first residue.
            atoms_j: List of atoms to be used for the second residue.

        Returns:
            pairwise_dihedrals: A tensor containing pairwise dihedral angles between residues.
                Shape: (batch_size, num_residues, num_residues)
        """
        # take uppercase of atom names
        atoms_i = [atom.upper() for atom in atoms_i]
        atoms_j = [atom.upper() for atom in atoms_j]

        for atom in atoms_i:
            if atom not in atom2idx:
                raise ValueError(f"Atom {atom} is not valid.")
        for atom in atoms_j:
            if atom not in atom2idx:
                raise ValueError(f"Atom {atom} is not valid.")

        atoms_i = [atom2idx[atom] for atom in atoms_i]
        atoms_j = [atom2idx[atom] for atom in atoms_j]

        # get coordinates of specified atoms for residue i and j
        n = self.n_residues
        coords_i = self.xyz[:, :, atoms_i].repeat_interleave(n, dim=1)
        coords_j = self.xyz[:, :, atoms_j].repeat(1, n, 1, 1)

        # and construct all-pairwise four-atom coordinates
        coords = torch.cat([coords_i, coords_j], dim=-2)  # bsz, n_res^2, 4, 3

        dih = geom.dihedral(coords[:, :, 0], coords[:, :, 1], coords[:, :, 2], coords[:, :, 3])
        dih = dih.reshape(-1, n, n)
        return dih

    def translate(self, translation: torch.Tensor, atomwise: bool = False):
        """Translate the structures by a given tensor of shape (batch_size, num_residues, 3)
        Translation is performed residue-wise by default, but atomwise translation
        can be performed when `atomwise=True`. In that case, the translation tensor
        should have a shape of (batch_size, num_residues, num_atom, 3).

        Args:
            translation: Translation vector.
                Shape: (batch_size, num_residues, 3) if `atomwise=False`,
                (batch_size, num_residues, num_atom, 3) otherwise.
        """

        if not atomwise:
            translation = rearrange(translation, "b n c -> b n () c")

        # update xyz coordinates
        self.xyz += translation

    def standardize(self, atom_mask: bool = None, residue_mask: bool = None):
        """Standardize the coordinates of the structures to have zero mean and unit standard deviation.

        Args:
            atom_mask: Mask for atoms used for standardization. If None, all atoms are used.
                `atom_mask` and `residue_mask` cannot be specified at the same time.
                Shape: (batch_size, num_residues, num_atoms)
            residue_mask: Mask for residues used for standardization. If None, all residues are used.
                `atom_mask` and `residue_mask` cannot be specified at the same time.
                Shape: (batch_size, num_residues)
        """
        if atom_mask is not None and residue_mask is not None:
            raise ValueError("Only one of atom_mask and residue_mask can be specified.")

        if self._standardized:
            raise ValueError("Coordinates are already standardized.")

        if atom_mask:
            atom_mask = atom_mask * self.atom_mask
        elif residue_mask:
            atom_mask = residue_mask.unsqueeze(-1) * self.atom_mask
        else:
            atom_mask = self.atom_mask

        total_atom_counts = rearrange(atom_mask, "b n a -> b (n a)").sum(axis=1, keepdims=True)
        # compute coordinate mean
        xyz_masked = self.xyz * atom_mask.unsqueeze(-1)
        xyz_masked = rearrange(xyz_masked, "b n a c -> b (n a) c")
        self.mu = xyz_masked.nan_to_num(0.0).sum(axis=1) / total_atom_counts
        # compute coordinate standard deviation
        xyz_centered = self.xyz.nan_to_num(0.0) - rearrange(self.mu, "b c -> b () () c")
        xyz_centered = xyz_centered**2 * atom_mask.unsqueeze(-1)
        xyz_centered = rearrange(xyz_centered, "b n a c -> b (n a) c")
        self.std = torch.sqrt(xyz_centered.sum(axis=1) / total_atom_counts)

        self.xyz = (self.xyz - self.mu) / self.std
        self._standardized = True

    def unstandardize(self):
        """Recover the coordinates at original scale from teh standardized coordinates."""
        if not self._standardized:
            raise ValueError("Cannot unstandardize structures that are not standardized.")

        self.xyz = self.xyz * self.std + self.mu
        self._standardized = False


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
