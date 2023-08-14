import pytest
import torch

from protstruc import AntibodyFvStructureBatch
from protstruc.constants import MAX_N_ATOMS_PER_RESIDUE


def test_AntibodyFvStructureBatch_from_xyz():
    bsz, nres, natoms = 10, 128, 25

    xyz = torch.randn(bsz, nres, natoms, 3)
    batch = AntibodyFvStructureBatch.from_xyz(xyz)

    assert batch.get_xyz().shape == (bsz, nres, natoms, 3)

    dih, dih_mask = batch.backbone_dihedrals()
    assert dih.shape == (bsz, nres, 3)
    assert dih_mask.shape == (bsz, nres, 3)

    orientations = batch.backbone_orientations()
    translations = batch.backbone_translations()
    assert orientations.shape == (bsz, nres, 3, 3)
    assert translations.shape == (bsz, nres, 3)


def test_AntibodyFvStructureBatch_from_pdb():
    batch = AntibodyFvStructureBatch.from_pdb("tests/15c8_HL.pdb")

    assert batch.get_xyz().shape == (1, 229, MAX_N_ATOMS_PER_RESIDUE, 3)

    assert batch.get_heavy_chain_lengths() == torch.tensor([119])
    assert batch.get_light_chain_lengths() == torch.tensor([110])

    assert len(batch.get_heavy_chain_seq()) == 1
    assert len(batch.get_heavy_chain_seq()[0]) == 119

    assert len(batch.get_light_chain_seq()) == 1
    assert len(batch.get_light_chain_seq()[0]) == 110


def test_AntibodyFvStructureBatch_from_multiple_pdbs():
    pdbs = ["tests/15c8_HL.pdb", "tests/1a6v_JN.pdb"]
    batch = AntibodyFvStructureBatch.from_pdb(pdbs)

    assert batch.get_xyz().shape == (2, 230, MAX_N_ATOMS_PER_RESIDUE, 3)

    assert torch.allclose(batch.get_heavy_chain_lengths(), torch.tensor([119, 120]))
    assert torch.allclose(batch.get_light_chain_lengths(), torch.tensor([110, 110]))

    assert len(batch.get_heavy_chain_seq()) == 2
    assert len(batch.get_heavy_chain_seq()[0]) == 119
    assert len(batch.get_heavy_chain_seq()[1]) == 120

    assert len(batch.get_light_chain_seq()) == 2
    assert len(batch.get_light_chain_seq()[0]) == 110
    assert len(batch.get_light_chain_seq()[1]) == 110
