from protstruc import StructureBatch

import numpy as np
import pytest


def test_StructureBatch_from_xyz():
    n_proteins, max_n_residues, max_n_atoms = 16, 100, 25
    xyz = np.random.rand(n_proteins, max_n_residues, max_n_atoms)
    sb = StructureBatch.from_xyz(xyz)


def test_max_n_atoms_per_residue():
    n_proteins, max_n_residues, max_n_atoms = 16, 100, 25
    xyz = np.random.rand(n_proteins, max_n_residues, max_n_atoms)
    sb = StructureBatch.from_xyz(xyz)

    assert sb.get_max_n_atoms_per_residue() == 25


def test_StructureBatch_from_xyz_with_chain_ids():
    n_proteins, max_n_residues, max_n_atoms = 16, 100, 25
    xyz = np.random.rand(n_proteins, max_n_residues, max_n_atoms)

    chain_ids = np.zeros((n_proteins, max_n_residues))
    chain_ids[:, 20:60] = 1.0
    chain_ids[:, 60:] = 2.0

    sb = StructureBatch.from_xyz(xyz, chain_ids=chain_ids)

    assert sb.get_n_terminal_mask().shape == (n_proteins, max_n_residues)
    assert sb.get_c_terminal_mask().shape == (n_proteins, max_n_residues)

    assert (sb.get_n_terminal_mask().sum(axis=1) == 3).all()
    assert (sb.get_c_terminal_mask().sum(axis=1) == 3).all()


def test_StructureBatch_get_backbone_dihedrals():
    n_proteins, max_n_residues, max_n_atoms = 16, 100, 25
    xyz = np.random.rand(n_proteins, max_n_residues, max_n_atoms, 3)

    chain_ids = np.zeros((n_proteins, max_n_residues))
    chain_ids[:, 20:60] = 1.0
    chain_ids[:, 60:] = 2.0

    sb = StructureBatch.from_xyz(xyz, chain_ids=chain_ids)

    dihedrals, dihedral_mask = sb.get_backbone_dihedrals()
    assert dihedrals.shape == (n_proteins, max_n_residues, 3)
    assert (dihedrals >= -np.pi).all() & (dihedrals <= np.pi).all()

    assert dihedral_mask.shape == (n_proteins, max_n_residues, 3)

    nterm = sb.get_n_terminal_mask()
    cterm = sb.get_c_terminal_mask()

    # phi is not defined for N-term residue. should be zero-filled.
    assert (dihedrals[nterm][:, 0] == 0.0).all()
    # psi and omega are not defined for N-term residue. should be zero-filled.
    assert (dihedrals[cterm][:, [1, 2]] == 0.0).all()
