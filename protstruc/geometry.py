"""Utility functions for computing 3D geometry of protein structures.

This module contains the following functions:

- `angle(a, b, c, to_degree=False)`: Compute planar angles between three points.
- `dihedral(a, b, c, d, to_degree=False)`: Compute dihedral angle between four points.
- `place_fourth_atom(a, b, c, length, planar, dihedral)`: Place a fourth atom X given three atoms (A, B and C) and
    the bond length (CX), planar angle (XCB), and dihedral angle (XCB vs ACB).
"""

import torch
import math
import numpy as np

from typing import Union, Tuple, List
from sklearn.manifold import MDS
from einops import repeat
from .constants import ideal
from .decorator import with_tensor

MASK = 12345679


@with_tensor
def dot(x, y):
    return (x * y).sum(dim=-1, keepdim=True)


@with_tensor
def norm(x):
    return x.norm(dim=-1, keepdim=True)


@with_tensor
def unit(x):
    return x / norm(x)


@with_tensor
def angle(
    a: Union[np.ndarray, torch.Tensor],
    b: Union[np.ndarray, torch.Tensor],
    c: Union[np.ndarray, torch.Tensor],
    to_degree: bool = False,
) -> Union[np.array, torch.Tensor]:
    """Compute planar angles (0 ~ pi) between three (array of) points a, b and c.

    Note:
        The planar angle is computed as the angle between the vectors `ab` and `bc`
        using the dot product followed by `torch.arccos`. If `to_degree` is False, the
        output is in radians between 0 and pi. Otherwise, the output is in degrees
        between 0 and 180.

    Args:
        a: 3D coordinates of atom a. Shape: (n, 3)
        b: 3D coordinates of atom b. Shape: (n, 3)
        c: 3D coordinates of atom c. Shape: (n, 3)
        to_degree:
            Whether to return angles in degree. Defaults to False.

    Returns:
        Planar angle between three points. Shape: (n,)
    """
    ba = a - b
    bc = c - b
    cosine_angle = dot(ba, bc) / (norm(ba) * norm(bc))

    if to_degree:
        return torch.rad2deg(torch.arccos(cosine_angle)).squeeze(-1)
    else:
        return torch.arccos(cosine_angle).squeeze(-1)


@with_tensor
def dihedral(
    a: Union[np.ndarray, torch.Tensor],
    b: Union[np.ndarray, torch.Tensor],
    c: Union[np.ndarray, torch.Tensor],
    d: Union[np.ndarray, torch.Tensor],
    to_degree: bool = False,
) -> Union[np.array, torch.Tensor]:
    """Compute dihedral angle (-pi ~ pi) between (array of) four points a, b, c and d.

    Note:
        The **dihedral angle** is the angle in the clockwise direction of the **fourth atom**
        compared to the **first atom**, while looking down **the axis of the second to the
        third**.

        The dihedral angle is computed as the angle between the plane defined by
        vectors `ba` and `bc` and the plane defined by vectors `bc` and `cd`.
        In short, the dihedral angle (theta) is obtained by first computing cos(theta) and sin(theta)
        using dot and cross products of the normal vectors of the two planes, and then computing
        theta using `torch.atan2`.

    Tip:
        Here is a nice explanation of the computation of dihedral angles:
        [https://leimao.github.io/blog/Dihedral-Angles](https://leimao.github.io/blog/Dihedral-Angles/)

    Args:
        a: 3D coordinates of atom a (shape: (n, 3))
        b: 3D coordinates of atom b (shape: (n, 3))
        c: 3D coordinates of atom c (shape: (n, 3))
        d: 3D coordinates of atom d (shape: (n, 3))
        to_degree:
            Whether to return dihedrals in degree. Defaults to False.

    Returns:
        Dihedral angle between four points. (shape: (n,))
    """
    b0 = a - b
    b1 = c - b
    b2 = d - c

    b0xb1 = np.cross(b0, b1)
    b1xb2 = np.cross(b2, b1)
    b0xb1_x_b1xb2 = np.cross(b0xb1, b1xb2)

    x = dot(b0xb1, b1xb2)  # proportional to cos(theta)
    y = dot(b0xb1_x_b1xb2, b1) / norm(b1)  # proportional to sin(theta)

    if to_degree:
        return np.degrees(np.arctan2(y, x)).squeeze(-1)
    else:
        return np.arctan2(y, x).squeeze(-1)


def place_fourth_atom(
    a: Union[np.array, torch.Tensor],
    b: Union[np.array, torch.Tensor],
    c: Union[np.array, torch.Tensor],
    length: Union[np.array, torch.Tensor],
    planar: Union[np.array, torch.Tensor],
    dihedral: Union[np.array, torch.Tensor],
) -> Union[np.array, torch.Tensor]:
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
        3D coordinates of the new atom X (shape: (n, 3))
    """
    bc = b - c
    bc = bc / norm(bc)

    n = torch.cross((b - a), bc)
    n = n / norm(n)

    d = [bc, torch.cross(n, bc), n]
    m = [
        length * torch.cos(planar),
        length * torch.sin(planar) * torch.cos(dihedral),
        -length * torch.sin(planar) * torch.sin(dihedral),
    ]
    x = c + sum([magnitude * direction for magnitude, direction in zip(m, d)])
    return x


def ideal_local_frame() -> Union[np.array, torch.Tensor]:
    """Compute ideal local coordinate system of a residue centered at N

    Returns:
        Local coordinate system of a residue centered at N, with atom order N, CA, C, CB (shape: (4, 3))
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


def ideal_backbone_coordinates(
    size: Union[Tuple[int], List[int]], include_cb: bool = False
) -> Union[np.array, torch.Tensor]:
    """Return a batch of ideal backbone coordinates (N, Ca, C and optionally Cb)
    with a given batch size and number of residues.

    Args:
        size:
        include_cb: Whether to include Cb atom in the frame. Defaults to False.

    Returns:
        A batch of ideal backbone coordinates (N, Ca, C and optionally Cb).
        Shape: (batch_size, num_residues, 3, 3) if `include_cb` is False,
            otherwise (batch_size, num_residues, 4, 3).
    """
    # let Ca-C vector be always along x-axis
    ca = torch.zeros(3)
    c = torch.tensor([ideal.AC, 0.0, 0.0])
    n = torch.tensor(
        [
            ideal.NA * math.cos(ideal.NAC),
            ideal.NA * math.sin(ideal.NAC),
            0.0,
        ]
    )

    if include_cb:
        _b, _c = (ca - n), (c - ca)
        _a = torch.cross(_b, _c)

        cb = -0.58273431 * _a + 0.56802827 * _b - 0.54067466 * _c + ca
        xyz = torch.stack([n, ca, c, cb])
    else:
        xyz = torch.stack([n, ca, c])

    return xyz.expand(*size, -1, -1)


def reconstruct_backbone_distmat_from_interresidue_geometry(
    d_cb: torch.Tensor,
    omega: torch.Tensor,
    theta: torch.Tensor,
    phi: torch.Tensor,
    mask: torch.Tensor = None,
    chain_breaks: list = None,
) -> torch.Tensor:
    """Reconstruct the backbone distance matrix from interresidue geometry
    including Cb distance matrix (`d_cb`), Ca-Cb-Ca'-Cb' dihedral (`omega`),
    N-Ca-Cb-Cb' dihedral (`theta`), and Ca-Cb-Cb' planar angle (`phi`).

    Args:
        d_cb: Cb distance matrix (shape: (L, L))
        omega: Ca-Cb-Ca'-Cb' dihedral matrix (shape: (L, L))
        theta: N-Ca-Cb-Cb' dihedral matrix (shape: (L, L))
        phi: Ca-Cb-Cb' planar angle matrix (shape: (L, L))
        mask:
            Mask for valid residue pairs, i.e., pairs of residues whose distance
            can be reconstructed from interresidue geometry (shape: (L, L))
        chain_breaks:
            List of chain breaks, i.e., indices of residues that are not in the
            same chain with the next one.

    Returns:
        Backbone distance matrix representing the distance between N, Ca, C atoms between residues (shape: (3, 3, L, L))
    """
    N_IDX, CA_IDX, C_IDX, CB_IDX = 0, 1, 2, 3

    L = d_cb.shape[0]
    x = ideal_local_frame().unsqueeze(1)  # (4, 1, 3) in order of N, Ca, C, Cb

    # prepare angles and dihedrals
    d_cb = d_cb.unsqueeze(-1)
    angle_ABB = phi.unsqueeze(-1)
    angle_BBA = phi.T.unsqueeze(-1)
    dihedral_NABB = theta.unsqueeze(-1)
    dihedral_BBAN = theta.T.unsqueeze(-1)
    dihedral_ABBA = omega.unsqueeze(-1)

    # compute the coordinates of N, Ca, C, Cb of all other residues
    # with respect to the local coordinate system of each residue
    y = torch.zeros(4, L * L, 3)

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
    dist_mat = torch.zeros(3, 3, L, L)

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
    dist_mat[N_IDX, CA_IDX, torch.arange(L), torch.arange(L)] = ideal.NA
    dist_mat[CA_IDX, N_IDX, torch.arange(L), torch.arange(L)] = ideal.NA

    dist_mat[CA_IDX, C_IDX, torch.arange(L), torch.arange(L)] = ideal.AC
    dist_mat[C_IDX, CA_IDX, torch.arange(L), torch.arange(L)] = ideal.AC

    dist_mat[C_IDX, N_IDX, torch.arange(L - 1), torch.arange(1, L)] = ideal.C_N
    dist_mat[N_IDX, C_IDX, torch.arange(1, L), torch.arange(L - 1)] = ideal.C_N

    if chain_breaks is not None:
        for idx in chain_breaks:
            dist_mat[C_IDX, N_IDX, idx, idx + 1] = MASK
            dist_mat[N_IDX, C_IDX, idx + 1, idx] = MASK

    # replace masked distances with MASK, which will be replaced with Floyd-Warshall
    # shortest path distances later
    if mask is not None:
        dist_mat[:, :, ~mask] = MASK
    dist_mat = torch.nan_to_num(dist_mat, nan=MASK)

    # replace MASK with Floyd-Warshall shortest path distance
    # 3 x 3 x L x L
    dist_mat = dist_mat.transpose(0, 2, 1, 3).reshape(3 * L, 3 * L)

    for i in range(3 * L):
        d = dist_mat[i]
        tmp = torch.stack([dist_mat, d[None, :] + d[:, None]])
        dist_mat = torch.min(tmp, axis=0)

    # symmetrize
    dist_mat = (dist_mat + dist_mat.transpose(1, 0)) / 2.0

    dist_mat = dist_mat.reshape(3, L, 3, L).transpose(0, 2, 1, 3)

    # replace bond lengths with ideal ones, again
    dist_mat[N_IDX, CA_IDX, torch.arange(L), torch.arange(L)] = ideal.NA
    dist_mat[CA_IDX, N_IDX, torch.arange(L), torch.arange(L)] = ideal.NA

    dist_mat[CA_IDX, C_IDX, torch.arange(L), torch.arange(L)] = ideal.AC
    dist_mat[C_IDX, CA_IDX, torch.arange(L), torch.arange(L)] = ideal.AC

    dist_mat[C_IDX, N_IDX, torch.arange(L - 1), torch.arange(1, L)] = ideal.C_N
    dist_mat[N_IDX, C_IDX, torch.arange(1, L), torch.arange(L - 1)] = ideal.C_N

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
    # return coords * np.array([1, 1, -1])[None, None, :] if phi.mean() > 0 else coords
    return coords * np.array([1, 1, -1])[None, None, :]


def gram_schmidt(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.FloatTensor:
    """Given three xyz coordinates, compute the orthonormal basis
    using Gram-Schmidt process. Specifically, compute the orthonormal
    basis of the plane defined by vectors (c - b) and (a - b).

    Args:
        a: xyz coordinates of three atoms (shape: (*, 3))
        b: xyz coordinates of three atoms (shape: (*, 3))
        c: xyz coordinates of three atoms (shape: (*, 3))

    Returns:
        Orthonormal basis of the plane defined by vectors `c - b` and `a - b`.
            Shape: (*, 3, 3)
    """

    v1 = c - b
    e1 = v1 / norm(v1)

    v2 = a - b
    u2 = v2 - dot(e1, v2) * e1
    e2 = u2 / norm(u2)

    e3 = torch.cross(e1, e2)

    return torch.stack([e1, e2, e3], dim=-1)
