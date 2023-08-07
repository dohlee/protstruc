import protstruc.geometry as geom
import numpy as np

from protstruc import AntibodyFvStructure


def test_inter_residue_geometry():
    struc = AntibodyFvStructure("tests/15c8_HL.pdb")

    g = struc.inter_residue_geometry()

    L = 229
    d_cb, omega, theta, phi = g["d_cb"], g["omega"], g["theta"], g["phi"]
    assert d_cb.shape == (L, L)
    assert omega.shape == (L, L)
    assert theta.shape == (L, L)
    assert phi.shape == (L, L)

    pdist = geom.reconstruct_backbone_distmat_from_interresidue_geometry(
        d_cb, omega, theta, phi
    )
    assert pdist.shape == (3, 3, L, L)