import numpy as np

from .constants import ideal


def dot(x, y):
    return (x * y).sum(axis=-1, keepdims=True)


def norm(x):
    return np.linalg.norm(x, axis=-1, keepdims=True)


def unit(x):
    return x / norm(x)


def angle(a: np.array, b: np.array, c: np.array, to_degree=False) -> np.array:
    """_summary_

    Args:
        a (np.array): 3D coordinates of atom a (shape: (n, 3))
        b (np.array): 3D coordinates of atom b (shape: (n, 3))
        c (np.array): 3D coordinates of atom c (shape: (n, 3))
        to_degree (bool, optional):
            Whether to return angles in degree. Defaults to False.

    Returns:
        np.array: Planar angle between three points. (shape: (n, 1))
    """
    ba = a - b
    bc = c - b
    cosine_angle = dot(ba, bc) / (norm(ba) * norm(bc))

    if to_degree:
        return np.nan_to_num(np.degrees(np.arccos(cosine_angle)), 0.0).squeeze(-1)
    else:
        return np.nan_to_num(np.arccos(cosine_angle), 0.0).squeeze(-1)


def dihedral(a: np.array, b: np.array, c: np.array, d: np.array, to_degree=False) -> np.array:
    """Compute dihedral angle between four points.

    Args:
        a (np.array): 3D coordinates of atom a (shape: (n, 3))
        b (np.array): 3D coordinates of atom b (shape: (n, 3))
        c (np.array): 3D coordinates of atom c (shape: (n, 3))
        d (np.array): 3D coordinates of atom d (shape: (n, 3))
        to_degree (bool, optional):
            Whether to return dihedrals in degree. Defaults to False.

    Returns:
        np.array: Dihedral angle between four points. (shape: (n, 1))
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
        return np.nan_to_num(np.degrees(np.arctan2(y, x)), 0.0).squeeze(-1)
    else:
        return np.nan_to_num(np.arctan2(y, x), 0.0).squeeze(-1)


def place_fourth_atom(
    a: np.array,
    b: np.array,
    c: np.array,
    length: np.array,
    planar: np.array,
    dihedral: np.array,
) -> np.array:
    """Place a fourth atom X given three atoms (A, B and C) and
    the bond length (CX), planar angle (XCB), and dihedral angle (XCB vs ACB).

    Args:
        a (np.array): 3D coordinates of atom a (shape: (n, 3))
        b (np.array): 3D coordinates of atom b (shape: (n, 3))
        c (np.array): 3D coordinates of atom c (shape: (n, 3))
        length (np.array):
            Length of the bond between atom c and the new atom (shape: (n, 1))
            i.e., bond length CX
        planar (np.array):
            Planar angle between the new atom and the bond between atom c and the new atom (shape: (n, 1))
            i.e., angle XCB
        dihedral (np.array):
            Dihedral angle between the new atom and the plane defined by atoms a, b, and c (shape: (n, 1))
            i.e., dihedral angle between planes XCB and ACB

    Returns:
        np.array: 3D coordinates of the new atom X (shape: (n, 3))
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


def ideal_local_frame() -> np.array:
    """Compute ideal local coordinate system of a residue centered at N

    Returns:
        np.array:
            Local coordinate system of a residue centered at N,
            with atom order N, CA, C, CB (shape: (4, 3))
    """

    n = np.array([0.0, 0.0, 0.0])
    ca = np.array([0.0, 0.0, ideal.NA])
    cb = np.array(
        [
            0.0,
            ideal.AB * np.sin(ideal.NAB),
            ideal.NA + ideal.AB * np.cos(np.pi - ideal.NAB),
        ]
    )
    c = place_fourth_atom(cb, ca, n, ideal.NC, ideal.ANC, ideal.BANC)
    return np.array([n, ca, c, cb])


def reconstruct_backbone_distmat_from_interresidue_geometry(
    d_cb: np.array, omega: np.array, theta: np.array, phi: np.array
) -> np.array:
    """Reconstruct the backbone distance matrix from interresidue geometry
    including Cb distance matrix (`d_cb`), Ca-Cb-Ca'-Cb' dihedral (`omega`),
    N-Ca-Cb-Cb' dihedral (`theta`), and Ca-Cb-Cb' planar angle (`phi`).

    Args:
        d_cb (np.array): Cb distance matrix (shape: (L, L))
        omega (np.array): Ca-Cb-Ca'-Cb' dihedral matrix (shape: (L, L))
        theta (np.array): N-Ca-Cb-Cb' dihedral matrix (shape: (L, L))
        phi (np.array): Ca-Cb-Cb' planar angle matrix (shape: (L, L))

    Returns:
        np.array: Backbone distance matrix representing the distance between
        N, Ca, C atoms between residues (shape: (3, 3, L, L))
    """
    N_IDX, CA_IDX, C_IDX, CB_IDX = 0, 1, 2, 3

    L = d_cb.shape[0]
    x = ideal_local_frame()[:, np.newaxis]  # (4, 1, 3) in order of N, Ca, C, Cb

    # prepare angles and dihedrals
    d_cb = d_cb.reshape(-1, 1)
    angle_ABB = np.radians(phi.reshape(-1, 1))
    angle_BBA = np.radians(phi.T.reshape(-1, 1))
    dihedral_NABB = np.radians(theta.reshape(-1, 1))
    dihedral_BBAN = np.radians(theta.T.reshape(-1, 1))
    dihedral_ABBA = np.radians(omega.reshape(-1, 1))

    # compute the coordinates of N, Ca, C, Cb of all other residues
    # with respect to the local coordinate system of each residue
    y = np.zeros((4, L * L, 3))

    y[CB_IDX] = place_fourth_atom(
        x[N_IDX], x[CA_IDX], x[CB_IDX], d_cb, angle_ABB, dihedral_NABB
    )
    y[CA_IDX] = place_fourth_atom(
        x[CA_IDX], x[CB_IDX], y[CB_IDX], ideal.BA, angle_BBA, dihedral_ABBA
    )
    y[N_IDX] = place_fourth_atom(
        x[CB_IDX], y[CB_IDX], y[CA_IDX], ideal.AN, ideal.BAN, dihedral_BBAN
    )
    y[C_IDX] = place_fourth_atom(
        y[CB_IDX], y[CA_IDX], y[N_IDX], ideal.NC, ideal.ANC, ideal.BANC
    )

    # only take N, Ca and C coordinates and compute their pairwise distance
    dist_mat = np.zeros((3, 3, L, L))

    atoms = ["N", "A", "C"]
    for atom_i in [N_IDX, CA_IDX, C_IDX]:
        for atom_j in [N_IDX, CA_IDX, C_IDX]:
            pdist = np.linalg.norm(x[atom_i] - y[atom_j], axis=-1).reshape(L, L)

            if atom_i == atom_j:
                pdist[np.diag_indices(L)] = 0.0
            else:
                i, j = atoms[atom_i], atoms[atom_j]
                pdist[np.diag_indices(L)] = ideal.as_dict[f"{i}{j}"]

            dist_mat[atom_i, atom_j] = pdist

    return dist_mat
