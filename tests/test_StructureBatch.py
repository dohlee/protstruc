from protstruc import StructureBatch
from protstruc.general import ATOM

import torch
import protstruc as ps
import numpy as np
import pytest


def test_StructureBatch_from_xyz():
    n_proteins, max_n_residues, max_n_atoms = 16, 100, 25
    xyz = np.random.rand(n_proteins, max_n_residues, max_n_atoms, 3)
    sb = StructureBatch.from_xyz(xyz)


def test_max_n_atoms_per_residue():
    n_proteins, max_n_residues, max_n_atoms = 16, 100, 25
    xyz = np.random.rand(n_proteins, max_n_residues, max_n_atoms, 3)
    sb = StructureBatch.from_xyz(xyz)

    assert sb.get_max_n_atoms_per_residue() == 25


def test_StructureBatch_from_xyz_with_chain_ids():
    n_proteins, max_n_residues, max_n_atoms = 16, 100, 25
    xyz = np.random.rand(n_proteins, max_n_residues, max_n_atoms, 3)

    chain_idx = np.zeros((n_proteins, max_n_residues))
    chain_idx[:, 20:60] = 1.0
    chain_idx[:, 60:] = 2.0

    chain_ids = [["A", "B", "C"] for _ in range(n_proteins)]

    sb = StructureBatch.from_xyz(xyz, chain_idx=chain_idx, chain_ids=chain_ids)

    assert sb.get_n_terminal_mask().shape == (n_proteins, max_n_residues)
    assert sb.get_c_terminal_mask().shape == (n_proteins, max_n_residues)

    assert (sb.get_n_terminal_mask().sum(axis=1) == 3).all()
    assert (sb.get_c_terminal_mask().sum(axis=1) == 3).all()


def test_StructureBatch_from_pdb_single():
    # pdb_path = "tests/15c8_HL.pdb"
    pdb_path = "tests/1ad0_DC.pdb"
    sb = StructureBatch.from_pdb(pdb_path)

    xyz = sb.get_xyz()
    assert len(xyz) == 1

    # two chains
    assert (sb.get_n_terminal_mask().sum(axis=1) == 2).all()
    assert (sb.get_c_terminal_mask().sum(axis=1) == 2).all()


def test_StructureBatch_from_pdb_multiple():
    pdb_paths = ["tests/15c8_HL.pdb", "tests/1ad0_DC.pdb", "tests/5cjx_HL.pdb"]
    sb = StructureBatch.from_pdb(pdb_paths)

    xyz = sb.get_xyz()
    assert len(xyz) == 3

    # two chains for each
    assert (sb.get_n_terminal_mask().sum(axis=1) == 2).all()
    assert (sb.get_c_terminal_mask().sum(axis=1) == 2).all()


def test_StructureBatch_backbone_dihedrals():
    n_proteins, max_n_residues, max_n_atoms = 16, 100, 25
    xyz = np.random.rand(n_proteins, max_n_residues, max_n_atoms, 3)

    chain_idx = np.zeros((n_proteins, max_n_residues))
    chain_idx[:, 20:60] = 1.0
    chain_idx[:, 60:] = 2.0

    chain_ids = [["A", "B", "C"] for _ in range(n_proteins)]

    sb = StructureBatch.from_xyz(xyz, chain_idx=chain_idx, chain_ids=chain_ids)

    dihedrals, dihedral_mask = sb.backbone_dihedrals()
    assert dihedrals.shape == (n_proteins, max_n_residues, 3)

    assert (dihedrals >= -np.pi).all() & (dihedrals <= np.pi).all()
    assert ((dihedrals >= -np.pi) & (dihedrals < 0)).any()
    assert ((dihedrals >= 0) & (dihedrals <= np.pi)).any()

    assert dihedral_mask.shape == (n_proteins, max_n_residues, 3)

    nterm = sb.get_n_terminal_mask()
    cterm = sb.get_c_terminal_mask()

    # phi is not defined for N-term residue. should be zero-filled.
    assert (dihedrals[nterm][:, 0] == 0.0).all()
    # psi and omega are not defined for N-term residue. should be zero-filled.
    assert (dihedrals[cterm][:, [1, 2]] == 0.0).all()


def test_StructureBatch_from_pdb_id():
    pdb_id = "2ZIL"  # Human lysozyme
    sb = StructureBatch.from_pdb_id(pdb_id)

    xyz = sb.get_xyz()
    assert len(xyz) == 1

    # single chain
    assert (sb.get_n_terminal_mask().sum(axis=1) == 1).all()
    assert (sb.get_c_terminal_mask().sum(axis=1) == 1).all()


def test_StructureBatch_from_pdb_ids():
    pdb_id = ["2ZIL", "1REX"]  # Human lysozymes
    sb = StructureBatch.from_pdb_id(pdb_id)

    xyz = sb.get_xyz()
    assert len(xyz) == 2

    # single chains
    assert (sb.get_n_terminal_mask().sum(axis=1) == 1).all()
    assert (sb.get_c_terminal_mask().sum(axis=1) == 1).all()


def test_StructureBatch_pairwise_distance_matrix():
    pdb_id = "1REX"
    sb = StructureBatch.from_pdb_id(pdb_id)

    dist, dist_mask = sb.pairwise_distance_matrix()
    assert dist.shape == (1, 130, 130, 15, 15)
    assert dist_mask.shape == (1, 130, 130, 15, 15)

    ca_dist = dist[:, :, :, ATOM.CA, ATOM.CA]
    cb_dist = dist[:, :, :, ATOM.CB, ATOM.CB]

    assert (ca_dist >= 0).all()
    assert (cb_dist[~torch.isnan(cb_dist)] >= 0).all()

    # test if enum ATOM.CA works correctly
    assert (ca_dist == dist[:, :, :, 1, 1]).all()


def test_StructureBatch_backbone_orientations():
    pdb_id = "1REX"
    sb = StructureBatch.from_pdb_id(pdb_id)

    bb_orientations = sb.backbone_orientations("N", "CA", "C")
    assert bb_orientations.shape == (1, 130, 3, 3)


def test_StructureBatch_backbone_translations():
    pdb_id = "1REX"
    sb = StructureBatch.from_pdb_id(pdb_id)

    for atom in ["N", "CA", "C"]:
        bb_translations = sb.backbone_translations(atom)
        assert bb_translations.shape == (1, 130, 3)


def test_StructureBatch_get_chain_lengths():
    pdb_id = ["1REX", "4EOT"]
    sb = StructureBatch.from_pdb_id(pdb_id)

    chain_lengths = sb.get_total_lengths()
    print(chain_lengths)
    assert (chain_lengths == torch.tensor([130, 184])).all()


def test_StructureBatch_pairwise_dihedrals():
    pdb_id = ["1REX"]
    sb = StructureBatch.from_pdb_id(pdb_id)

    # phi = Dihedral(C_i, N_j, Ca_j, C_j)
    phi = sb.pairwise_dihedrals(atoms_i=["C"], atoms_j=["N", "CA", "C"])
    assert phi.shape == (1, 130, 130)

    # psi = Dihedral(N_i, Ca_i, C_i, N_j)
    psi = sb.pairwise_dihedrals(atoms_i=["N", "CA", "C"], atoms_j=["N"])
    assert psi.shape == (1, 130, 130)


def test_get_local_xyz():
    pdb_id = ["1REX", "4EOT"]
    sb = StructureBatch.from_pdb_id(pdb_id)

    n_atoms = sb.get_max_n_atoms_per_residue()

    local_xyz = sb.get_local_xyz()
    assert local_xyz.shape == (2, 184, n_atoms, 3)


def test_from_backbone_orientations_translations():
    pdb_id = ["1REX"]
    sb = StructureBatch.from_pdb_id(pdb_id)

    orientations = sb.backbone_orientations()
    translations = sb.backbone_translations()
    chain_idx = sb.get_chain_idx()
    chain_ids = sb.get_chain_ids()
    seq = sb.get_seq()

    sb2 = StructureBatch.from_backbone_orientations_translations(
        orientations, translations, chain_idx, chain_ids, seq
    )
    assert sb2.get_max_n_atoms_per_residue() == 15

    sb3 = StructureBatch.from_backbone_orientations_translations(
        orientations, translations, chain_idx, chain_ids, seq, include_cb=True
    )
    assert sb3.get_max_n_atoms_per_residue() == 15


def test_standardize_unstandardize():
    pdb_id = ["1REX"]
    sb = StructureBatch.from_pdb_id(pdb_id)

    sb.standardize()
    sb.unstandardize()


def test_standardized_not_nan():
    pdb_id = ["1REX"]
    sb = StructureBatch.from_pdb_id(pdb_id)

    atom_mask = sb.get_atom_mask()

    sb.standardize()
    xyz = sb.get_xyz()
    assert not torch.isnan(xyz[atom_mask.bool()]).any()


def test_cannot_standardize_twice():
    pdb_id = ["1REX"]
    sb = StructureBatch.from_pdb_id(pdb_id)

    with pytest.raises(ValueError):
        sb.standardize()
        sb.standardize()


def test_cannot_unstandardize_first():
    pdb_id = ["1REX"]
    sb = StructureBatch.from_pdb_id(pdb_id)

    with pytest.raises(ValueError):
        sb.unstandardize()


def test_standardize_and_unstandardize_reverts_original_xyz_correctly():
    pdb_id = ["1REX"]
    sb = StructureBatch.from_pdb_id(pdb_id)

    xyz = sb.get_xyz()
    sb.standardize()
    sb.unstandardize()
    xyz2 = sb.get_xyz()

    assert torch.allclose(xyz, xyz2, equal_nan=True, rtol=1e-4, atol=1e-5)


def test_center_at_origin():
    pdb_id = ["1REX"]
    sb = StructureBatch.from_pdb_id(pdb_id)

    sb.center_at()

    com = sb.center_of_mass()
    assert torch.allclose(com, torch.zeros_like(com), rtol=1e-4, atol=1e-5)


def test_center_at_desired_points():
    pdb_id = ["1REX", "4EOT"]
    sb = StructureBatch.from_pdb_id(pdb_id)

    centers = torch.randn([2, 3])
    sb.center_at(centers)

    assert torch.allclose(sb.center_of_mass(), centers, rtol=1e-4, atol=1e-5)


def test_get_residue_mask():
    pdb_id = ["1REX", "4EOT"]
    sb = StructureBatch.from_pdb_id(pdb_id)

    residue_mask = sb.get_residue_mask()
    assert residue_mask.shape == (2, 184)


def test_seq_idx():
    pdb_id = ["1REX", "4EOT"]
    sb = StructureBatch.from_pdb_id(pdb_id)

    seq_idx = sb.get_seq_idx()
    residue_mask = sb.get_residue_mask()

    assert seq_idx.shape == (2, 184)
    assert (seq_idx[~residue_mask.bool()] == ps.general.AA.UNK).all()
    print(seq_idx[0])
