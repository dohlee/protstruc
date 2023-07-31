import numpy as np

from typing import List, Union
from biopandas.pdb import PandasPdb
from scipy.spatial.distance import cdist
from collections import defaultdict

import protstruc.geometry as geom
from protstruc.constants import ideal
from protstruc.alphabet import three2one

CC_BOND_LENGTH = 1.522
CB_CA_N_ANGLE = 1.927
CB_DIHEDRAL = -2.143

N_IDX, CA_IDX, C_IDX = 0, 1, 2


class StructureBatch:
    """A batch of protein structures.

    This class provides an interface to initialize from and represent a batch of protein structures
    with various types of representations:

    StructureBatch object can be initialized with:
        - Backbone atom 3D coordinates `StructureBatch.from_xyz`
        - Full-atom 3D coordinates `StructureBatch.from_xyz`
        - Dihedral angles `StructureBatch.from_dihedrals` (TODO)
        - A set of PDB files `StructureBatch.from_pdb` (TODO)
    """

    def __init__(self, xyz: np.ndarray, chain_ids: np.ndarray = None):
        self.xyz = xyz
        self.max_n_atoms_per_residue = self.xyz.shape[2]

        if chain_ids is not None:
            self.chain_ids = chain_ids.astype(float)
            assert self.chain_ids.min() == 0.0, "Chain ids should start from zero"
        else:
            bsz, n_max_res = self.xyz.shape[:2]
            self.chain_ids = np.zeros((bsz, n_max_res))

    @classmethod
    def from_xyz(cls, xyz: np.ndarray, chain_ids: np.ndarray = None):
        """Initialize a StructureBatch from a 3D coordinate array.

        Examples:
            >>> bsz, n_max_res, n_max_atoms = 2, 10, 25
            >>> xyz = np.random.randn(bsz, n_max_res, n_max_atoms, 3)
            >>> sb = StructureBatch.from_xyz(xyz)

        Args:
            xyz (np.ndarray): Shape: (batch_size, num_residues, num_atoms, 3)
            chain_ids (np.ndarray, optional): Chain identifiers for each residue.
                Should be starting from zero. Defaults to None.
                Shape: (batch_size, num_residues)

        Returns:
            StructureBatch: A StructureBatch object.
        """
        self = cls(xyz, chain_ids)
        return self

    @classmethod
    def from_dihedrals(cls, dihedrals: np.ndarray, chain_ids: np.ndarray = None):
        """Initialize a StructureBatch from a dihedral angle array.

        Args:
            dihedrals (np.ndarray): Shape: (batch_size, num_residues, num_dihedrals)
            chain_ids (np.ndarray, optional): Chain identifiers for each residue.
                Should be starting from zero. Defaults to None.
                Shape: (batch_size, num_residues)
        """
        # TODO: Implement this
        pass

    @classmethod
    def from_pdb(cls, pdb_path: Union[List[str], str]):
        """Initialize a StructureBatch from a PDB file.

        Args:
            pdb_path: Path to a PDB file or a list of paths to PDB files.
        """
        # TODO: Implement this

    def get_xyz(self):
        return self.xyz

    def get_max_n_atoms_per_residue(self):
        return self.max_n_atoms_per_residue

    def get_n_terminal_mask(self) -> np.ndarray:
        """Return a boolean mask for the N-terminal residues.

        Returns:
            np.ndarray: Shape: (batch_size, num_residues)
        """
        padded = np.pad(
            self.chain_ids, ((0, 0), (1, 0)), mode="constant", constant_values=np.nan
        )
        return (padded[:, :-1] != padded[:, 1:]).astype(bool)

    def get_c_terminal_mask(self) -> np.ndarray:
        """Return a boolean mask for the C-terminal residues.

        Returns:
            np.ndarray: Shape: (batch_size, num_residues)
        """
        padded = np.pad(
            self.chain_ids, ((0, 0), (0, 1)), mode="constant", constant_values=np.nan
        )
        return (padded[:, :-1] != padded[:, 1:]).astype(bool)

    def get_backbone_dihedrals(self) -> np.ndarray:
        """Return the backbone dihedral angles.

        Returns:
            np.ndarray: Shape: (batch_size, num_residues, 3)
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

        # for i in range(1):
        #     print(
        #         np.linalg.norm(self.coord_per_atom["C"][i] - self.coord_per_atom["N"][i + 1])
        #     )

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
