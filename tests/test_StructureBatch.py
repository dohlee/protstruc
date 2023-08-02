from protstruc import StructureBatch

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

    assert dihedral_mask.shape == (n_proteins, max_n_residues, 3)

    nterm = sb.get_n_terminal_mask()
    cterm = sb.get_c_terminal_mask()

    # phi is not defined for N-term residue. should be zero-filled.
    assert (dihedrals[nterm][:, 0] == 0.0).all()
    # psi and omega are not defined for N-term residue. should be zero-filled.
    assert (dihedrals[cterm][:, [1, 2]] == 0.0).all()
