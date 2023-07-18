import numpy as np

from sklearn.manifold import MDS
from .constants import ideal

MASK = 12345679


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
        return np.degrees(np.arccos(cosine_angle)).squeeze(-1)
    else:
        return np.arccos(cosine_angle).squeeze(-1)


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
    b1xb2 = np.cross(b2, b1)
    b0xb1_x_b1xb2 = np.cross(b0xb1, b1xb2)

    x = dot(b0xb1, b1xb2)
    y = dot(b0xb1_x_b1xb2, b1) / norm(b1)

    if to_degree:
        return np.degrees(np.arctan2(y, x)).squeeze(-1)
    else:
        return np.arctan2(y, x).squeeze(-1)


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
    bc = bc / norm(bc)

    n = np.cross((b - a), bc)
    n = n / norm(n)

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
            ideal.NA - ideal.AB * np.cos(ideal.NAB),
        ]
    )
    c = place_fourth_atom(cb, ca, n, ideal.NC, ideal.ANC, ideal.BANC)
    return np.array([n, ca, c, cb])


def reconstruct_backbone_distmat_from_interresidue_geometry(
    d_cb: np.array,
    omega: np.array,
    theta: np.array,
    phi: np.array,
    mask: np.array = None,
    chain_breaks: list = None,
) -> np.array:
    """Reconstruct the backbone distance matrix from interresidue geometry
    including Cb distance matrix (`d_cb`), Ca-Cb-Ca'-Cb' dihedral (`omega`),
    N-Ca-Cb-Cb' dihedral (`theta`), and Ca-Cb-Cb' planar angle (`phi`).

    Args:
        d_cb (np.array): Cb distance matrix (shape: (L, L))
        omega (np.array): Ca-Cb-Ca'-Cb' dihedral matrix (shape: (L, L))
        theta (np.array): N-Ca-Cb-Cb' dihedral matrix (shape: (L, L))
        phi (np.array): Ca-Cb-Cb' planar angle matrix (shape: (L, L))
        mask (np.array):
            Mask for valid residue pairs, i.e., pairs of residues whose distance
            can be reconstructed from interresidue geometry (shape: (L, L))
        chain_breaks (list):
            List of chain breaks, i.e., indices of residues that are not in the
            same chain with the next one.

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

    # replace bond lengths with ideal ones
    dist_mat[N_IDX, CA_IDX, np.arange(L), np.arange(L)] = ideal.NA
    dist_mat[CA_IDX, N_IDX, np.arange(L), np.arange(L)] = ideal.NA

    dist_mat[CA_IDX, C_IDX, np.arange(L), np.arange(L)] = ideal.AC
    dist_mat[C_IDX, CA_IDX, np.arange(L), np.arange(L)] = ideal.AC

    dist_mat[C_IDX, N_IDX, np.arange(L - 1), np.arange(1, L)] = ideal.C_N
    dist_mat[N_IDX, C_IDX, np.arange(1, L), np.arange(L - 1)] = ideal.C_N

    if chain_breaks is not None:
        for idx in chain_breaks:
            dist_mat[C_IDX, N_IDX, idx, idx + 1] = MASK
            dist_mat[N_IDX, C_IDX, idx + 1, idx] = MASK

    # replace masked distances with MASK, which will be replaced with Floyd-Warshall
    # shortest path distances later
    if mask is not None:
        dist_mat[:, :, ~mask] = MASK
    dist_mat = np.nan_to_num(dist_mat, nan=MASK)

    # replace MASK with Floyd-Warshall shortest path distance
    # 3 x 3 x L x L
    dist_mat = dist_mat.transpose(0, 2, 1, 3).reshape(3 * L, 3 * L)

    for i in range(3 * L):
        d = dist_mat[i]
        tmp = np.stack([dist_mat, d[None, :] + d[:, None]])
        dist_mat = np.min(tmp, axis=0)

    # symmetrize
    dist_mat = (dist_mat + dist_mat.transpose(1, 0)) / 2.0

    dist_mat = dist_mat.reshape(3, L, 3, L).transpose(0, 2, 1, 3)

    # replace bond lengths with ideal ones, again
    dist_mat[N_IDX, CA_IDX, np.arange(L), np.arange(L)] = ideal.NA
    dist_mat[CA_IDX, N_IDX, np.arange(L), np.arange(L)] = ideal.NA

    dist_mat[CA_IDX, C_IDX, np.arange(L), np.arange(L)] = ideal.AC
    dist_mat[C_IDX, CA_IDX, np.arange(L), np.arange(L)] = ideal.AC

    dist_mat[C_IDX, N_IDX, np.arange(L - 1), np.arange(1, L)] = ideal.C_N
    dist_mat[N_IDX, C_IDX, np.arange(1, L), np.arange(L - 1)] = ideal.C_N

    return dist_mat


def initialize_backbone_with_mds(dist_mat: np.array, max_iter: int = 500) -> np.array:
    """Given a pairwise distance matrix of backbone atoms, initialize
    the coordinates of the backbone atoms using multidimensional scaling.

    Args:
        dist_mat (np.array): Pairwise distance matrix of backbone atoms (shape: (3, 3, L, L)).
        max_iter (int, optional): Maximum number of iterations for MDS. Defaults to 500.

    Returns:
        np.array: Coordinates of backbone atoms (N, CA, C) (shape: (3, L, 3)).
    """
    L = dist_mat.shape[-1]
    pdist = dist_mat.transpose(0, 2, 1, 3).reshape(3 * L, 3 * L)

    mds = MDS(3, max_iter=max_iter, dissimilarity="precomputed")
    coords = mds.fit_transform(pdist).reshape(3, L, 3)

    coords = fix_chirality(coords)  # 3, L, 3

    # place Cb and O atoms at ideal positions
    N_IDX, CA_IDX, C_IDX = 0, 1, 2

    cb_coords = place_fourth_atom(
        coords[C_IDX], coords[N_IDX], coords[CA_IDX], ideal.AB, ideal.NAB, ideal.BANC
    ).reshape(1, L, 3)

    o_coords = place_fourth_atom(
        np.roll(coords[N_IDX], shift=-1, axis=0),
        coords[CA_IDX],
        coords[C_IDX],
        ideal.CO,
        ideal.ACO,
        ideal.NACO,
    ).reshape(1, L, 3)

    coords = np.concatenate([coords, o_coords, cb_coords], axis=0)
    return coords


def fix_chirality(coords: np.array) -> np.array:
    """Fix chirality of the backbone so that all the phi dihedrals
    are negative.

    Args:
        coords (np.array): Coordinates of backbone atoms (N, CA, C) (shape: (3, L, 3)).

    Returns:
        np.array: Fixed coordinates.
    """
    # compute phi dihedral angles
    N_IDX, CA_IDX, C_IDX = 0, 1, 2
    phi = dihedral(
        coords[C_IDX, :-1],
        coords[N_IDX, 1:],
        coords[CA_IDX, 1:],
        coords[C_IDX, 1:],
    )

    # return mirrored coordinates if phi is positive on average
    return coords * np.array([1, 1, -1])[None, None, :] if phi.mean() > 0 else coords
