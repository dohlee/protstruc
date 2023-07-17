from protstruc.constants import ideal


def test_ideal():
    # C-C bond length is about 1.51 ~ 1.53
    assert 1.51 < ideal.CA < 1.53
    assert 1.51 < ideal.AC < 1.53
    assert 1.51 < ideal.AB < 1.53
    assert 1.51 < ideal.BA < 1.53

    # N-Ca bond length is about 1.45 ~ 1.47
    assert 1.45 < ideal.NA < 1.47
    assert 1.45 < ideal.AN < 1.47

    # C-N bond length is about 1.32 ~ 1.33
    assert 1.32 < ideal.C_N < 1.33
