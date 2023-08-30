from collections import defaultdict
from typing import Dict, List, Literal, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from biotite.database.rcsb import fetch
from einops import rearrange, repeat

import protstruc.geometry as geom
from protstruc.constants import MAX_N_ATOMS_PER_RESIDUE
from protstruc.general import AA, ATOM, CDR_NAMES, ressymb_to_resindex
from protstruc.pdb import PDB, ChothiaAntibodyPDB


def isnull(x):
    if isinstance(x, list):
        return any([pd.isnull(_x) for _x in x])
    else:
        return pd.isnull(x)


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
        residue_idx: torch.LongTensor = None,
    ):
        if (chain_idx is not None and chain_ids is None) or (
            chain_idx is None and chain_ids is not None
        ):
            raise ValueError(
                "Both `chain_idx` and `chain_ids` should be provided or None."
            )

        self.xyz = xyz
        self.atom_mask = atom_mask
        self.batch_size, self.n_residues, self.max_n_atoms_per_residue = self.xyz.shape[
            :3
        ]

        if atom_mask is not None:
            self.residue_mask = atom_mask.any(dim=-1)
        else:
            self.residue_mask = torch.ones(
                self.batch_size, self.n_residues, dtype=torch.bool
            )

        if chain_idx is not None:
            for i, chidx in enumerate(chain_idx):
                msk = ~torch.isnan(chidx)
                assert (
                    chidx[msk].min() == 0
                ), f"Protein {i}: Chain index should start from zero"

            self.chain_idx = chain_idx
        else:
            self.chain_idx = torch.zeros(self.batch_size, self.n_residues)

        self.chain_ids = chain_ids
        self.seq = seq
        self.residue_idx = residue_idx

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
        **kwargs,
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

        self = cls(xyz, atom_mask, chain_idx, chain_ids, seq, **kwargs)
        return self

    @classmethod
    def from_pdb(cls, pdb_path: Union[str, List[str]], **kwargs) -> "StructureBatch":
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
        tmp_residue_idx = []
        chain_ids = []
        for f in pdb_path:
            pdb = PDB.read_pdb(f)

            _atom_xyz, _atom_mask = pdb.get_atom_xyz()
            _chain_idx = pdb.get_chain_idx()
            _residue_idx = pdb.get_residue_idx()
            _chain_ids = pdb.get_chain_ids()
            _seq_dict = pdb.get_seq_dict()

            tmp_atom_xyz.append(_atom_xyz)
            tmp_atom_mask.append(_atom_mask)
            tmp_chain_idx.append(_chain_idx)
            tmp_residue_idx.append(_residue_idx)
            chain_ids.append(_chain_ids)
            seq.append(_seq_dict)

        max_n_residues = max([len(xyz) for xyz in tmp_atom_xyz])

        atom_xyz = torch.zeros(bsz, max_n_residues, MAX_N_ATOMS_PER_RESIDUE, 3)
        atom_mask = torch.zeros(bsz, max_n_residues, MAX_N_ATOMS_PER_RESIDUE).bool()
        chain_idx = torch.ones(bsz, max_n_residues) * torch.nan
        residue_idx = torch.ones(bsz, max_n_residues) * torch.nan

        for i in range(bsz):
            _atom_xyz = tmp_atom_xyz[i]
            _atom_mask = tmp_atom_mask[i]
            _chain_idx = tmp_chain_idx[i]
            _residue_idx = tmp_residue_idx[i]

            atom_xyz[i, : len(_atom_xyz)] = _atom_xyz
            atom_mask[i, : len(_atom_mask)] = _atom_mask
            chain_idx[i, : len(_chain_idx)] = _chain_idx
            residue_idx[i, : len(_residue_idx)] = _residue_idx

        self = cls(
            atom_xyz, atom_mask, chain_idx, chain_ids, seq, residue_idx, **kwargs
        )
        return self

    @classmethod
    def from_pdb_id(cls, pdb_id: Union[str, List[str]], **kwargs) -> "StructureBatch":
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
        tmp_residue_idx = []
        chain_ids = []
        for id in pdb_id:
            pdb = PDB.read_pdb(fetch(id, "pdb"))

            _atom_xyz, _atom_mask = pdb.get_atom_xyz()
            _chain_idx = pdb.get_chain_idx()
            _residue_idx = pdb.get_residue_idx()
            _chain_ids = pdb.get_chain_ids()
            _seq_dict = pdb.get_seq_dict()

            tmp_atom_xyz.append(_atom_xyz)
            tmp_atom_mask.append(_atom_mask)
            tmp_chain_idx.append(_chain_idx)
            tmp_residue_idx.append(_residue_idx)
            chain_ids.append(_chain_ids)
            seq.append(_seq_dict)

        max_n_residues = max([len(xyz) for xyz in tmp_atom_xyz])

        atom_xyz = torch.zeros(bsz, max_n_residues, MAX_N_ATOMS_PER_RESIDUE, 3)
        atom_mask = torch.zeros(bsz, max_n_residues, MAX_N_ATOMS_PER_RESIDUE)
        chain_idx = torch.ones(bsz, max_n_residues) * torch.nan
        residue_idx = torch.ones(bsz, max_n_residues) * torch.nan

        for i in range(bsz):
            _atom_xyz = tmp_atom_xyz[i]
            _atom_mask = tmp_atom_mask[i]
            _chain_idx = tmp_chain_idx[i]
            _residue_idx = tmp_residue_idx[i]

            atom_xyz[i, : len(_atom_xyz)] = _atom_xyz
            atom_mask[i, : len(_atom_mask)] = _atom_mask
            chain_idx[i, : len(_chain_idx)] = _chain_idx
            residue_idx[i, : len(_residue_idx)] = _residue_idx

        self = cls(
            atom_xyz, atom_mask, chain_idx, chain_ids, seq, residue_idx, **kwargs
        )
        return self

    @classmethod
    def from_backbone_orientations_translations(
        cls,
        orientations: Union[np.ndarray, torch.Tensor],
        translations: Union[np.ndarray, torch.Tensor],
        chain_idx: Union[np.ndarray, torch.Tensor] = None,
        chain_ids: List[List[str]] = None,
        seq: List[Dict[str, str]] = None,
        residue_idx: Union[np.ndarray, torch.Tensor] = None,
        include_cb: bool = False,
        **kwargs,
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
        ideal_backbone = geom.ideal_backbone_coordinates(
            (batch_size, n_residues), include_cb
        )
        n_atoms = ideal_backbone.shape[2]

        # rotate and translate ideal backbone coordinates by orientations
        orientations = repeat(orientations, "b n i j -> b n a i j", a=n_atoms)

        atom_xyz = torch.einsum("bnaij,bnaj->bnai", orientations, ideal_backbone)
        atom_xyz = atom_xyz + rearrange(translations, "b n i -> b n () i")

        atom_mask = torch.ones_like(atom_xyz[..., 0])

        # pad to xyz and mask MAX_N_ATOMS_PER_RESIDUE
        dummy_xyz = torch.zeros(
            batch_size, n_residues, MAX_N_ATOMS_PER_RESIDUE - n_atoms, 3
        )
        atom_xyz = torch.cat([atom_xyz, dummy_xyz], axis=-2)

        dummy_mask = torch.zeros(
            batch_size, n_residues, MAX_N_ATOMS_PER_RESIDUE - n_atoms
        )
        atom_mask = torch.cat([atom_mask, dummy_mask], axis=-1)

        self = cls(
            atom_xyz, atom_mask, chain_idx, chain_ids, seq, residue_idx, **kwargs
        )
        return self

    @classmethod
    def from_dihedrals(
        cls,
        dihedrals: Union[np.ndarray, torch.Tensor],
        chain_idx: Union[np.ndarray, torch.Tensor] = None,
        chain_ids: List[List[str]] = None,
        **kwargs,
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

    def get_batch_size(self) -> int:
        return self.batch_size

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
        local_xyz = local_xyz - xyz[:, :, ATOM.CA].unsqueeze(-2)
        return local_xyz

    def get_atom_mask(self) -> torch.BoolTensor:
        """Return a boolean mask for valid atoms.

        Returns:
            atom_mask: Shape (batch_size, num_residues, num_atoms)
        """
        return self.atom_mask

    def get_residue_mask(self) -> torch.BoolTensor:
        """Return a boolean mask for valid residues.

        Returns:
            residue_mask: Shape (batch_size, num_residues)
        """
        return self.atom_mask[:, :, ATOM.CA].bool()

    def get_chain_idx(self) -> torch.LongTensor:
        return self.chain_idx.long()

    def get_chain_ids(self):
        return self.chain_ids

    def get_seq(self) -> List[Dict[str, str]]:
        """Return the amino acid sequence of proteins.

        Returns:
            seq_dict: A list of dictionaries containing sequence information for each chain.
        """
        return self.seq

    def get_seq_idx(self) -> torch.LongTensor:
        """Return a tensor containing the integer representation of amino acid sequence of proteins.

        Returns:
            seq_idx: A tensor containing the integer representation of amino acid sequence of proteins.
        """
        seq_idx = (
            torch.ones(self.batch_size, self.n_residues, dtype=torch.long) * AA.UNK
        )
        for i, (seqdict, chain_ids) in enumerate(zip(self.seq, self.chain_ids)):
            seq_concat = "".join([seqdict[chain_id] for chain_id in chain_ids])
            seq_idx[i, : len(seq_concat)] = torch.tensor(
                [ressymb_to_resindex[res] for res in seq_concat]
            ).long()

        return seq_idx

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

    def get_max_n_residues(self) -> int:
        """Return the number of residues in the longest protein in the batch.

        Returns:
            max_n_residues: The number of residues in the longest protein in the batch.
        """
        return self.n_residues

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

        dist_mask = (
            self.atom_mask[:, :, None, :, None] * self.atom_mask[:, None, :, None, :]
        )
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
        n_coords = self.xyz[:, :, ATOM.N]
        ca_coords = self.xyz[:, :, ATOM.CA]
        c_coords = self.xyz[:, :, ATOM.C]

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
        a1_coords = self.xyz[:, :, ATOM[a1]]
        a2_coords = self.xyz[:, :, ATOM[a2]]
        a3_coords = self.xyz[:, :, ATOM[a3]]

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
        return self.xyz[:, :, ATOM[atom]]

    def _pairwise_xyz(
        self, atoms_i: List[str], atoms_j: List[str]
    ) -> torch.FloatTensor:
        """Return a matrix representing a pairwise distance between residues defined by
        two sets of atoms, one for each side of the residue.

        Args:
            atoms_i: List of atoms to be used for the first residue.
            atoms_j: List of atoms to be used for the second residue.

        Returns:
            pairwise_xyz: A tensor containing pairwise distances between residues.
                Shape: (batch_size, num_residues^2, len(atoms_i) + len(atoms_j), 3)
        """
        for atom in atoms_i + atoms_j:
            if not ATOM.is_valid(atom):
                raise ValueError(f"Atom {atom} is not valid.")

        atoms_i = [ATOM[a] for a in atoms_i]
        atoms_j = [ATOM[a] for a in atoms_j]

        # get coordinates of specified atoms for residue i and j
        n = self.n_residues
        coords_i = self.xyz[:, :, atoms_i].repeat_interleave(n, dim=1)
        coords_j = self.xyz[:, :, atoms_j].repeat(1, n, 1, 1)

        # and construct all-pairwise four-atom coordinates
        coords = torch.cat([coords_i, coords_j], dim=-2)  # bsz, n_res^2, n_atoms, 3

        return coords

    def pairwise_dihedrals(
        self, atoms_i: List[str], atoms_j: List[str]
    ) -> torch.FloatTensor:
        """Return a matrix representing a pairwise dihedral angle between residues defined by
        two sets of atoms, one for each side of the residue.

        Args:
            atoms_i: List of atoms to be used for the first residue.
            atoms_j: List of atoms to be used for the second residue.

        Returns:
            pairwise_dihedrals: A tensor containing pairwise dihedral angles between residues.
                Shape: (batch_size, num_residues, num_residues)
        """
        coords = self._pairwise_xyz(atoms_i, atoms_j)  # bsz, n_res^2, 4, 3

        dih = geom.dihedral(
            coords[:, :, 0], coords[:, :, 1], coords[:, :, 2], coords[:, :, 3]
        )
        dih = dih.reshape(-1, self.n_residues, self.n_residues)
        return dih

    def pairwise_planar_angles(
        self, atoms_i: List[str], atoms_j: List[str]
    ) -> torch.FloatTensor:
        """Return a matrix representing a pairwise planar angles between residues defined by
        two sets of atoms, one for each side of the residue.

        Args:
            atoms_i: List of atoms to be used for the first residue.
            atoms_j: List of atoms to be used for the second residue.

        Returns:
            pairwise_planar_angles: A tensor containing pairwise planar angles between residues.
                Shape: (batch_size, num_residues, num_residues)
        """
        coords = self._pairwise_xyz(atoms_i, atoms_j)  # bsz, n_res^2, 3, 3

        planar_angle = geom.angle(coords[:, :, 0], coords[:, :, 1], coords[:, :, 2])
        planar_angle = planar_angle.reshape(-1, self.n_residues, self.n_residues)
        return planar_angle

    def translate(self, translation: torch.Tensor, atomwise: bool = False):
        """Translate the structures by a given tensor of shape (batch_size, num_residues, 3)
        or (batch_size, 1, 3). Translation is performed residue-wise by default,
        but atomwise translation can be performed when `atomwise=True`.
        In that case, the translation tensor should have a
        shape of (batch_size, num_residues, num_atom, 3).

        Args:
            translation: Translation vector.
                Shape: (batch_size, num_residues, 3) if `atomwise=False`,
                (batch_size, num_residues, num_atom, 3) otherwise.
        """

        if not atomwise:
            translation = rearrange(translation, "b n c -> b n () c")

        # update xyz coordinates
        self.xyz += translation

    def rotate(self, rotation: torch.Tensor):
        """Rotate the structures by a given rotation matrix of shape (batch_size, 3, 3).

        Args:
            rotation: Rotation matrix. Shape: (batch_size, 3, 3) if rotations is applied structure-by-structure,
                (3, 3) if the same rotation is to be applied to all structures.
        """
        if rotation.ndim == 2:
            rotation = rearrange(rotation, "i j -> () () () i j")
        else:
            rotation = rearrange(rotation, "b i j -> b () () i j")

        # update xyz coordinates
        self.xyz = torch.einsum("bnaij,bnaj->bnai", rotation, self.xyz)

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

        total_atom_counts = rearrange(atom_mask, "b n a -> b (n a)").sum(
            axis=1, keepdims=True
        )
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
        """Recover the coordinates at original scale from the standardized coordinates."""
        if not self._standardized:
            raise ValueError(
                "Cannot unstandardize structures that are not standardized."
            )

        self.xyz = self.xyz * self.std + self.mu
        self._standardized = False

    def center_of_mass(self) -> torch.Tensor:
        """Compute the center of mass of the structures.

        Warning:
            Only Ca atoms are considered when computing the coordinates of center of mass.

        Returns:
            center_of_mass: A tensor containing the center of mass of the structures.
                Shape: (batch_size, 3)
        """
        xyz_ca = self.xyz[:, :, ATOM.CA]
        return xyz_ca.nanmean(axis=1)

    def center_at(self, center: torch.Tensor = None):
        """Translate the whole structure so that the center of Ca atom coordinates is at the given
        3D coordinates. If `center` is not specified, the structures (considering only Ca coordinates)
        are centered at the origin.

        Args:
            center: Coordinates of the center.
                Shape: (batch_size, 3) or (3,)
        """
        if center is None:
            center = torch.zeros(1, 3)

        if center.ndim > 2 or center.shape[-1] != 3:
            raise ValueError(
                f"`center` must have a shape of (batch_size, 3) or (3,), got {center.shape}."
            )

        if center.ndim == 2 and center.shape[0] != self.batch_size:
            raise ValueError(
                f"`center` must have a shape of (batch_size, 3) or (3,), got {center.shape}."
            )

        if center.ndim == 1:
            center = center.unsqueeze(0)  # (1 3)

        # compute translation vector and translate
        translation = center - self.center_of_mass()
        translation = rearrange(translation, "b c -> b () () c")

        self.xyz += translation

    def inter_residue_geometry(self) -> Dict[str, torch.Tensor]:
        """Return a dictionary of inter-residue geometry, which is used for representing
        protein structure for trRoseTTA.

        Returns:
            inter_residue_geometry: A dictionary containing inter-residue geometry tensors.
        """
        ret = {}
        dist, dist_mask = self.pairwise_distance_matrix()  # b n n a a

        # Ca-Ca distance (Symmetric)
        ret["d_ca"] = dist[:, :, :, ATOM.CA, ATOM.CA]
        ret["d_ca_mask"] = dist_mask[:, :, :, ATOM.CA, ATOM.CA]
        # Cb-Cb distance (Symmetric)
        ret["d_cb"] = dist[:, :, :, ATOM.CB, ATOM.CB]
        ret["d_cb_mask"] = dist_mask[:, :, :, ATOM.CB, ATOM.CB]
        # N-O distance (Non-symmetric)
        ret["d_no"] = dist[:, :, :, ATOM.N, ATOM.O]
        ret["d_no_mask"] = dist_mask[:, :, :, ATOM.N, ATOM.O]

        # Ca-Cb-Cb'-Ca' dihedral (Symmetric)
        ret["omega"] = self.pairwise_dihedrals(["CA", "CB"], ["CA", "CB"])
        # N-Ca-Cb-Cb' dihedral (Non-symmetric)
        ret["theta"] = self.pairwise_dihedrals(["N", "CA", "CB"], ["CB"])
        # Ca-Cb-Cb' planar angle (Non-symmetric)
        ret["phi"] = self.pairwise_planar_angles(["CA", "CB"], ["CB"])

        return ret

    def get_topk_nearest_residue_mask(
        self, query_xyz: torch.FloatTensor, k: int = 128, mask: torch.BoolTensor = None
    ) -> torch.BoolTensor:
        """Return a boolean mask for the top-k nearest residues with respect to the
        given 3D coordinates of query points. Distances are measured using the alpha-carbon atoms.

        Warning:
            This operation is defined only for a single-size StructureBatch.

        Args:
            query_xyz: Coordinates of the query points. Shape: (num_query_points, 3)
            k: Maximum mumber of neighboring residues to consider.
                Defaults to min(128, n_residues).
            mask: If given, only consider residues denoted by this mask. Defaults to None.

        Returns:
            torch.BoolTensor: Shape: (1, num_residues)
        """

        if self.batch_size > 1:
            raise ValueError(
                "get_topk_nearest_residue_mask method is not defined "
                "for a StructureBatch with batch size > 1."
            )

        xyz = self.get_xyz()[0, :, ATOM.CA]
        dist = torch.norm(
            xyz[:, None] - query_xyz, dim=-1
        )  # n_residues, n_query_points

        dist, _ = dist.min(dim=-1)  # n_residues

        _mask = self.residue_mask[0]
        if mask is not None:
            _mask = _mask & mask

        dist[~_mask] = 1e9

        # adjust k if necessary
        k = min(k, _mask.sum())

        _, idx = dist.topk(k, largest=False)  # k
        ret = torch.zeros(self.n_residues, dtype=torch.bool).scatter(0, idx, True)
        return ret.unsqueeze(0)

    def diffuse_xyz(self, beta: torch.FloatTensor):
        """Add Gaussian noises of given variance `beta` to the atom 3D coordinates of the structures.
        This may be conceived as a single diffusion step of the Langevin dynamics
        from timestep t-1 to t.

        Note:
            This operation modifies the xyz coordinate in an in-place manner.

        Args:
            beta: Variance of the Gaussian noise. Shape: (batch_size,)
        """
        beta = rearrange(beta, "b -> b () () ()")
        noise = torch.randn_like(self.xyz) * beta.sqrt()

        self.xyz = (1 - beta).sqrt() * self.xyz + noise

    def align(
        self, target: "StructureBatch", atom_mask: torch.BoolTensor = None
    ) -> torch.Tensor:
        """Align the structures to a given structure.

        Args:
            target: StructureBatch to align to.
            atom_mask: Mask for atoms used for alignment. If None, all atoms are used.

        Returns:
            rotation: Rotation matrices used for the alignment.
                Shape: (batch_size, 3, 3)
        """
        if target.get_batch_size() != 1 and self.batch_size != target.get_batch_size():
            raise ValueError("Batch size of the two structures must be the same.")

        if atom_mask is None:
            atom_mask = self.atom_mask * target.get_atom_mask()

        source_xyz = rearrange(self.get_xyz(), "b n a c -> b (n a) c")
        target_xyz = rearrange(target.get_xyz(), "b n a c -> b (n a) c")

        atom_mask = rearrange(atom_mask, "b n a -> b (n a)").bool()

        rotations, translations = [], []
        for source, target, mask in zip(source_xyz, target_xyz, atom_mask):
            source = source[mask]
            target = target[mask]
            # compute rotation matrix and translation vectors
            r, t = geom.kabsch(source, target)

            rotations.append(r)
            translations.append(t)

        rotations, translations = torch.stack(rotations), torch.stack(translations)

        # apply rotation and translation
        self.rotate(rotations)
        self.translate(translations.unsqueeze(1))

    def residue_masked_select(self, mask: torch.BoolTensor) -> "StructureBatch":
        """Return a StructureBatch containing only the residues denoted by the given mask.

        Args:
            mask: Mask for residues to select. Shape: (batch_size, num_residues)

        Note:
            `chain_ids` and `seq` attributes are not updated accordingly. i.e., they will
            still contain the information for all residues.

        Returns:
            StructureBatch: A StructureBatch containing only the residues denoted by the given mask.
        """
        if self.batch_size > 1:
            raise ValueError(
                "residue_masked_select method is not defined "
                "for a StructureBatch with batch size > 1."
            )

        if mask.shape != self.residue_mask.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match residue mask shape {self.residue_mask.shape}."
            )

        if mask.dtype != torch.bool:
            raise ValueError("Mask must be a boolean tensor.")

        if mask.ndim == 1:
            mask = mask.unsqueeze(0)

        xyz = self.xyz[mask].unsqueeze(0)
        atom_mask = self.atom_mask[mask].unsqueeze(0)
        chain_idx = self.chain_idx[mask].unsqueeze(0)
        chain_ids = self.chain_ids
        seq = self.seq

        return StructureBatch(xyz, atom_mask, chain_idx, chain_ids, seq)


class AntibodyStructureBatch(StructureBatch):
    def __init__(
        self,
        xyz: torch.Tensor,
        atom_mask: torch.BoolTensor = None,
        chain_idx: torch.Tensor = None,
        chain_ids: List[str] = None,
        seq: List[Dict[str, str]] = None,
        residue_idx: torch.BoolTensor = None,
        residue_masks: Dict[str, torch.BoolTensor] = None,
        heavy_chain_id: List[str] = None,
        light_chain_id: List[str] = None,
        antigen_chain_ids: List[List[str]] = None,
        numbering_scheme: Literal["kabat", "chothia", "imgt"] = "chothia",
        keep_fv_only: bool = False,
    ):
        super().__init__(xyz, atom_mask, chain_idx, chain_ids, seq, residue_idx)

        self.numbering_scheme = numbering_scheme
        self.residue_masks = residue_masks

        self.heavy_chain_id = heavy_chain_id
        self.light_chain_id = light_chain_id
        self.antigen_chain_ids = antigen_chain_ids
        self.keep_fv_only = keep_fv_only

    def get_heavy_chain_mask(self) -> torch.BoolTensor:
        return self.residue_masks["heavy_chain"]

    def get_light_chain_mask(self) -> torch.BoolTensor:
        return self.residue_masks["light_chain"]

    def get_antigen_mask(self) -> torch.BoolTensor:
        return self.residue_masks["antigen"]

    def get_heavy_chain_id(self) -> List[str]:
        return self.heavy_chain_id

    def get_light_chain_id(self) -> List[str]:
        return self.light_chain_id

    def get_antigen_chain_ids(self) -> List[List[str]]:
        return self.antigen_chain_ids

    def is_fv_only(self) -> bool:
        return self.keep_fv_only

    def get_cdr_mask(self, subset: Union[str, List[str]] = None) -> torch.BoolTensor:
        subset = subset or CDR_NAMES
        subset = _always_list(subset)

        _masks = torch.stack([self.residue_masks[cdr] for cdr in subset], axis=0)
        return _masks.any(axis=0)

    def get_cdr_anchor_mask(
        self,
        subset: Union[
            Literal["H1", "H2", "H3", "L1", "L2", "L3"],
            List[Literal["H1", "H2", "H3", "L1", "L2", "L3"]],
        ] = None,
    ) -> torch.BoolTensor:
        """Return a boolean mask for the CDR anchor residues.

        Note:
            CDR anchor residues are defined as the residues that are not in the CDRs,
            but are adjacent to the CDRs. i.e., both a residue right before a CDR loop and
            a residue right after a CDR loop.

        Args:
            subset: A CDR name or a list of CDR names.
            Defaults to None, which means all CDRs are considered.

        Returns:
            torch.BoolTensor: A boolean mask tensor denoting CDR anchor residues.
        """
        subset = subset or CDR_NAMES
        subset = _always_list(subset)

        for cdr in subset:
            if cdr not in CDR_NAMES:
                raise ValueError(f"CDR {cdr} is not valid.")

        cdr_mask = self.get_cdr_mask(subset)
        cdr_mask_next = F.pad(cdr_mask, (0, 1), mode="constant", value=False)[:, 1:]
        cdr_mask_prev = F.pad(cdr_mask, (1, 0), mode="constant", value=False)[:, :-1]

        front_anchor_mask = ~cdr_mask & cdr_mask_next
        back_anchor_mask = ~cdr_mask & cdr_mask_prev

        return front_anchor_mask | back_anchor_mask

    def get_residue_idx(self) -> torch.BoolTensor:
        return self.residue_idx

    @classmethod
    def from_pdb(
        cls,
        pdb_path: Union[str, List[str]],
        heavy_chain_id: List[str] = None,
        light_chain_id: List[str] = None,
        antigen_chain_ids: List[List[str]] = None,
        numbering_scheme: Literal["kabat", "chothia", "imgt"] = "chothia",
        keep_fv_only: bool = False,
        **kwargs,
    ) -> "AntibodyStructureBatch":
        """Initialize an `AntibodyStructureBatch` from a PDB file or a list of PDB files.

        Examples:
            Initialize a `StructureBatch` object from a single PDB file,
            >>> pdb_path = '1a0a.pdb'
            >>> sb = StructureBatch.from_pdb(pdb_path)

            or with a list of PDB files.
            >>> pdb_paths = ['1a0a.pdb', '1a0b.pdb']
            >>> sb = StructureBatch.from_pdb(pdb_paths)

        Args:
            pdb_path: Path to a PDB file or a list of paths to PDB files.
            heavy_chain_id: Chain ID of the heavy chain.
            light_chain_id: Chain ID of the light chain.
            antigen_chain_ids: Chain IDs of the antigen chains. Defaults to None

        Returns:
            AntibodyStructureBatch: An AntibodyStructureBatch object.
        """
        if numbering_scheme not in ["kabat", "chothia", "imgt", None]:
            raise ValueError(
                'Antibody numbering scheme must be one of "kabat", "chothia", "imgt".'
            )

        pdb_path = _always_list(pdb_path)
        bsz = len(pdb_path)

        heavy_chain_id = _always_list(heavy_chain_id)
        light_chain_id = _always_list(light_chain_id)
        antigen_chain_ids = _always_list(antigen_chain_ids)

        heavy_chain_id = [None if isnull(x) else x for x in heavy_chain_id]
        light_chain_id = [None if isnull(x) else x for x in light_chain_id]
        antigen_chain_ids = [None if isnull(x) else x for x in antigen_chain_ids]

        tmp_atom_xyz, tmp_atom_mask, tmp_chain_idx, seq = [], [], [], []
        tmp_residue_idx = []
        tmp_residue_masks = defaultdict(list)

        chain_ids = []
        for f, hid, lid, aids in zip(
            pdb_path, heavy_chain_id, light_chain_id, antigen_chain_ids
        ):
            pdb = ChothiaAntibodyPDB.read_pdb(f, hid, lid, aids, keep_fv_only)

            _atom_xyz, _atom_mask = pdb.get_atom_xyz()
            _chain_idx = pdb.get_chain_idx()
            _residue_idx = pdb.get_residue_idx()
            _chain_ids = pdb.get_chain_ids()
            _seq_dict = pdb.get_seq_dict()

            tmp_atom_xyz.append(_atom_xyz)
            tmp_atom_mask.append(_atom_mask)
            tmp_chain_idx.append(_chain_idx)
            tmp_residue_idx.append(_residue_idx)
            chain_ids.append(_chain_ids)
            seq.append(_seq_dict)

            tmp_residue_masks["heavy_chain"].append(pdb.get_heavy_chain_mask())
            tmp_residue_masks["light_chain"].append(pdb.get_light_chain_mask())
            tmp_residue_masks["antigen"].append(pdb.get_antigen_mask())
            for cdr in CDR_NAMES:
                tmp_residue_masks[cdr].append(pdb.get_cdr_mask(cdr))

        max_n_residues = max([len(xyz) for xyz in tmp_atom_xyz])

        atom_xyz = torch.zeros(bsz, max_n_residues, MAX_N_ATOMS_PER_RESIDUE, 3)
        atom_mask = torch.zeros(bsz, max_n_residues, MAX_N_ATOMS_PER_RESIDUE)
        chain_idx = torch.ones(bsz, max_n_residues) * torch.nan
        residue_idx = torch.ones(bsz, max_n_residues) * torch.nan

        residue_mask_keys = ["heavy_chain", "light_chain", "antigen"]
        residue_mask_keys += CDR_NAMES

        residue_masks = {}
        for key in residue_mask_keys:
            residue_masks[key] = torch.zeros(bsz, max_n_residues).bool()

        for i in range(bsz):
            _atom_xyz = tmp_atom_xyz[i]
            _atom_mask = tmp_atom_mask[i]
            _chain_idx = tmp_chain_idx[i]
            _residue_idx = tmp_residue_idx[i]

            atom_xyz[i, : len(_atom_xyz)] = _atom_xyz
            atom_mask[i, : len(_atom_mask)] = _atom_mask
            chain_idx[i, : len(_chain_idx)] = _chain_idx
            residue_idx[i, : len(_residue_idx)] = _residue_idx

            for key in residue_mask_keys:
                len_residues = len(tmp_residue_masks[key][i])
                residue_masks[key][i, :len_residues] = tmp_residue_masks[key][i]

        self = cls(
            atom_xyz,
            atom_mask,
            chain_idx,
            chain_ids,
            seq,
            residue_idx,
            residue_masks,
            heavy_chain_id,
            light_chain_id,
            antigen_chain_ids,
            numbering_scheme,
            keep_fv_only,
            **kwargs,
        )
        return self
