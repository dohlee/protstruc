import pytest
import protstruc.geometry as geom

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
    assert np.allclose(dihedral, np.array([90.0]))
