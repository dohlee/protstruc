from protstruc.io import _precompute_internal_index_map

from biopandas.pdb import PandasPdb


def test_precompute_internal_index_map():
    pdb_df = PandasPdb().read_pdb("tests/1ad0_DC.pdb").df["ATOM"]
    internal_index_map = _precompute_internal_index_map(pdb_df)

    assert max(internal_index_map.values()) == 433
    assert len(internal_index_map.values()) == 433
