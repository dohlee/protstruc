import numpy as np

from biopandas.pdb import PandasPdb
from scipy.spatial.distance import cdist

import protstruc.geometry as geom

CC_BOND_LENGTH = 1.522
CB_CA_N_ANGLE = 1.927
CB_DIHEDRAL = -2.143


class AntibodyFvStructure(object):
    def __init__(
        self,
        pdb_path,
        impute_missing_atoms=True,
        heavy_chain_id="H",
        light_chain_id="L",
    ):
        self.df = PandasPdb().read_pdb(pdb_path).df["ATOM"]
        self.df["residue_id"] = (
            self.df["chain_id"] + self.df["insertion"] + self.df["residue_number"].astype(str)
        )

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

        self.coord_per_atom = {}
        for atom in atoms:
            self.coord_per_atom[atom] = df_piv[atom].values

        if impute_missing_atoms:
            self.impute_cb_coord()

    def valid_coord_mask(self, atom):
        return np.isfinite(self.coord_per_atom[atom]).all(axis=-1)

    def filter_atom(self, atom):
        return self.df["atom_name"] == atom

    def pdist(self, atom1="CA", atom2="CA"):
        c1 = self.coord[self.filter_atom(atom1)]
        c2 = self.coord[self.filter_atom(atom2)]

        return cdist(c1, c2)

    def impute_cb_coord(self):
        c = self.coord_per_atom["C"]
        n = self.coord_per_atom["N"]
        ca = self.coord_per_atom["CA"]

        cb_coords = geom.place_fourth_atom(
            c, n, ca, CC_BOND_LENGTH, CB_CA_N_ANGLE, CB_DIHEDRAL
        )

        to_fill = ~self.valid_coord_mask("CB")
        self.coord_per_atom["CB"][to_fill] = cb_coords[to_fill]

    def inter_residue_geometry(self, to_degree=True):
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
