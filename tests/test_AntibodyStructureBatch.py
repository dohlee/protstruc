import pytest
import torch

from protstruc import AntibodyStructureBatch
from protstruc.constants import MAX_N_ATOMS_PER_RESIDUE


def test_AntibodyFvStructureBatch_from_pdb():
    batch = AntibodyStructureBatch.from_pdb(
        "tests/6dc4.pdb", heavy_chain_id="H", light_chain_id="L"
    )

    assert batch.get_xyz().shape == (1, 437, MAX_N_ATOMS_PER_RESIDUE, 3)
