import pytest
import protstruc
import protstruc.geometry as geom
from protstruc.io import to_pdb

import numpy as np


def test_angle():
    a = np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
        ]
    )
    b = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    c = np.array(
        [
            [0, 1, 0],
            [0.5, np.sqrt(3) / 2, 0],
        ]
    )

    angle = geom.angle(a, b, c, to_degree=True).flatten()
    assert np.allclose(angle, np.array([90.0, 60.0]))


def test_dihedral():
    a = np.array(
        [
            [1, 0, 0],
        ]
    )
    b = np.array(
        [
            [0, 0, 0],
        ]
    )
    c = np.array(
        [
            [0, 1, 0],
        ]
    )
    d = np.array(
        [
            [0, 1, 1],
        ]
    )

    dihedral = geom.dihedral(a, b, c, d, to_degree=True).flatten()
    assert np.allclose(dihedral, np.array([-90.0]))


def test_reconstruct_backbone_distmat_from_interresidue_geometry_dummy():
    L = 10
    d_cb = np.random.uniform(size=(L, L))
    omega = np.random.uniform(size=(L, L))
    theta = np.random.uniform(size=(L, L))
    phi = np.random.uniform(size=(L, L))

    distmat = geom.reconstruct_backbone_distmat_from_interresidue_geometry(
        d_cb, omega, theta, phi
    )

    assert distmat.shape == (3, 3, L, L)


def test_initialize_backbone_with_mds():
    struc = protstruc.AntibodyFvStructure("tests/15c8_HL.pdb")

    g = struc.inter_residue_geometry()

    L = 229
    d_cb, omega, theta, phi = g["d_cb"], g["omega"], g["theta"], g["phi"]
    assert d_cb.shape == (L, L)
    assert omega.shape == (L, L)
    assert theta.shape == (L, L)
    assert phi.shape == (L, L)

    distmat = geom.reconstruct_backbone_distmat_from_interresidue_geometry(
        d_cb, omega, theta, phi, chain_breaks=[struc.get_heavy_chain_length() - 1]
    )
    assert distmat.shape == (3, 3, L, L)

    coords = geom.initialize_backbone_with_mds(distmat, max_iter=2000)
    assert coords.shape == (5, L, 3)

    sequences = struc.get_sequences()
    chain_ids = struc.get_chain_ids()

    to_pdb("tests/15c8_HL_reconstructed.pdb", coords, sequences, chain_ids)
