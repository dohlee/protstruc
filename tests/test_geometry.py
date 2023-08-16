import torch
import pytest
import protstruc
import protstruc.geometry as geom
from protstruc.io import to_pdb

import numpy as np


def test_dot_tensor():
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    assert geom.dot(a, b) == 32


def test_dot_numpy():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    assert geom.dot(a, b) == 32


def test_norm_tensor():
    a = torch.tensor([[1, 2, 3], [4, 5, 6]]).float()
    assert geom.norm(a).shape == (2, 1)
    assert torch.isclose(geom.norm(a), torch.tensor([[14**0.5], [77**0.5]])).all()


def test_norm_numpy():
    a = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
    assert isinstance(geom.norm(a), np.ndarray)
    assert geom.norm(a).shape == (2, 1)
    assert np.allclose(geom.norm(a), np.array([[14**0.5], [77**0.5]]))


def test_angle_tensor():
    a = torch.tensor(
        [
            [1, 0, 0],
            [1, 0, 0],
        ],
    ).float()
    b = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 0],
        ],
    ).float()
    c = torch.tensor(
        [
            [0, 1, 0],
            [0.5, np.sqrt(3) / 2, 0],
        ],
    ).float()

    angle = geom.angle(a, b, c, to_degree=True).flatten()

    assert isinstance(angle, torch.Tensor)
    assert angle.shape == (2,)
    assert torch.isclose(angle, torch.tensor([90.0, 60.0])).all()


def test_angle_numpy():
    a = np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=np.float32,
    )
    b = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.float32,
    )
    c = np.array(
        [
            [0, 1, 0],
            [0.5, np.sqrt(3) / 2, 0],
        ],
        dtype=np.float32,
    )

    angle = geom.angle(a, b, c, to_degree=True).flatten()

    assert isinstance(angle, np.ndarray)
    assert angle.shape == (2,)
    assert np.allclose(angle, np.array([90.0, 60.0]))


def test_dihedral_tensor():
    a = torch.tensor(
        [
            [1, 0, 0],
        ]
    ).float()
    b = torch.tensor(
        [
            [0, 0, 0],
        ]
    ).float()
    c = torch.tensor(
        [
            [0, 1, 0],
        ]
    ).float()
    d = torch.tensor(
        [
            [0, 1, 1],
        ]
    ).float()

    dihedral = geom.dihedral(a, b, c, d, to_degree=True)

    assert isinstance(dihedral, torch.Tensor)
    assert dihedral.shape == (1,)
    assert torch.isclose(dihedral, torch.tensor([-90.0])).all()


def test_dihedral_numpy():
    a = np.array(
        [
            [1, 0, 0],
        ],
        dtype=np.float32,
    )
    b = np.array(
        [
            [0, 0, 0],
        ],
        dtype=np.float32,
    )
    c = np.array(
        [
            [0, 1, 0],
        ],
        dtype=np.float32,
    )
    d = np.array(
        [
            [0, 1, 1],
        ],
        dtype=np.float32,
    )

    dihedral = geom.dihedral(a, b, c, d, to_degree=True)

    assert isinstance(dihedral, np.ndarray)
    assert dihedral.shape == (1,)
    assert np.allclose(dihedral, np.array([-90.0]))


def test_dihedral_for_higher_dimension():
    a = np.array(
        [
            [
                [1, 0, 0],
            ]
        ],
        dtype=np.float32,
    )
    b = np.array(
        [
            [
                [0, 0, 0],
            ]
        ],
        dtype=np.float32,
    )
    c = np.array(
        [
            [
                [0, 1, 0],
            ]
        ],
        dtype=np.float32,
    )
    d = np.array(
        [
            [
                [0, 1, 1],
            ]
        ],
        dtype=np.float32,
    )

    dihedral = geom.dihedral(a, b, c, d, to_degree=True)
    assert dihedral.shape == (1, 1)
    assert np.allclose(dihedral, np.array([-90.0]))


@pytest.mark.skip(reason="Temporarily skipping")
def test_reconstruct_backbone_distmat_from_interresidue_geometry_dummy():
    L = 10
    d_cb = torch.rand(L, L)
    omega = torch.rand(L, L)
    theta = torch.rand(L, L)
    phi = torch.rand(L, L)

    distmat = geom.reconstruct_backbone_distmat_from_interresidue_geometry(
        d_cb, omega, theta, phi
    )

    assert distmat.shape == (3, 3, L, L)


@pytest.mark.skip(reason="Temporarily skipping")
def test_initialize_backbone_with_mds():
    struc = protstruc.AntibodyFvStructureBatch.from_pdb("tests/15c8_HL.pdb")

    g = struc.inter_residue_geometry()

    L = 229
    d_cb, omega, theta, phi = g["d_cb"], g["omega"], g["theta"], g["phi"]
    assert d_cb.shape == (1, L, L)
    assert omega.shape == (1, L, L)
    assert theta.shape == (1, L, L)
    assert phi.shape == (1, L, L)

    distmat = geom.reconstruct_backbone_distmat_from_interresidue_geometry(
        d_cb, omega, theta, phi, chain_breaks=[struc.get_heavy_chain_lengths()[0] - 1]
    )
    assert distmat.shape == (3, 3, L, L)

    coords = geom.initialize_backbone_with_mds(distmat, max_iter=3)  # cheap test
    assert coords.shape == (5, L, 3)

    sequences = struc.get_sequences()
    chain_ids = struc.get_chain_ids()

    to_pdb("tests/15c8_HL_reconstructed.pdb", coords, sequences, chain_ids)


def test_gram_schmidt():
    bsz, n_res = 16, 30

    a = torch.randn(bsz, n_res, 3)
    b = torch.randn(bsz, n_res, 3)
    c = torch.randn(bsz, n_res, 3)

    frame = geom.gram_schmidt(a, b, c)
    assert frame.shape == (bsz, n_res, 3, 3)


def test_ideal_backbone_coordinates():
    bsz, n_res = 16, 30

    xyz = geom.ideal_backbone_coordinates(size=(bsz, n_res))
    assert xyz.shape == (bsz, n_res, 3, 3)

    xyz_cb = geom.ideal_backbone_coordinates(size=(bsz, n_res), include_cb=True)
    assert xyz_cb.shape == (bsz, n_res, 4, 3)

    n = xyz[:, :, 0]
    ca = xyz[:, :, 1]
    c = xyz[:, :, 2]
    frame = geom.gram_schmidt(n, ca, c)
    assert frame.shape == (bsz, n_res, 3, 3)

    # check that the ideal coordinates results in identity frame
    assert (frame == torch.eye(3).expand(bsz, n_res, -1, -1)).all()
