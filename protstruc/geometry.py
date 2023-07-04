import numpy as np


def dot(x, y):
    return (x * y).sum(axis=-1, keepdims=True)


def norm(x):
    """
    Compute norm of a vector.
    """
    return np.linalg.norm(x, axis=-1, keepdims=True)


def unit(x):
    return x / norm(x)


def angle(a, b, c, to_degree=False):
    """
    Compute angle between three points.
    """
    ba = a - b
    bc = c - b
    cosine_angle = dot(ba, bc) / (norm(ba) * norm(bc))

    if to_degree:
        return np.nan_to_num(np.degrees(np.arccos(cosine_angle)), 0.0)
    else:
        return np.nan_to_num(np.arccos(cosine_angle), 0.0)


def dihedral(a, b, c, d, to_degree=False):
    """
    Compute dihedral angle between four points.
    """
    b0 = a - b
    b1 = c - b
    b2 = d - c

    b0xb1 = np.cross(b0, b1)
    b1xb2 = np.cross(b1, b2)
    b0xb1_x_b1xb2 = np.cross(b0xb1, b1xb2)

    x = dot(b0xb1, b1xb2)
    y = dot(b0xb1_x_b1xb2, b1) / norm(b1)

    if to_degree:
        return np.nan_to_num(np.degrees(np.arctan2(y, x)), 0.0)
    else:
        return np.nan_to_num(np.arctan2(y, x), 0.0)


def place_fourth_atom(a, b, c, length, planar, dihedral):
    """
    Given 3 coords + a length + a planar angle + a dihedral angle, compute a fourth coord
    """
    bc = b - c
    bc = bc / np.linalg.norm(bc, axis=-1, keepdims=True)

    n = np.cross((b - a), bc)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)

    d = [bc, np.cross(n, bc), n]
    m = [
        length * np.cos(planar),
        length * np.sin(planar) * np.cos(dihedral),
        -length * np.sin(planar) * np.sin(dihedral),
    ]
    x = c + sum([magnitude * direction for magnitude, direction in zip(m, d)])
    return x
