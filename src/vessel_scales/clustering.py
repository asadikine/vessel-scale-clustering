"""Multi-scale vessel clustering (Sadikine et al., ISBI 2023, Sect. 2.1).

Decomposes a binary vessel mask into scale-specific masks (small, medium,
large vessels) from the estimated radius of each skeleton branch:

1. Surface extraction (Omega) and 3D thinning (Lee et al., 1994) -> skeleton Gamma.
   Omega is a sub-voxel surface mesh by default, or the surface voxels.
2. Local radius at each skeleton voxel: the largest distance among its m
   nearest surface points (Eq. 1).
3. Branch labelling of the skeleton with ``skan.csr.Skeleton.path_label_image``.
4. Branch radius: median of the local radii along the branch (Eq. 2).
5. Branch reconstruction: every mask voxel takes the label of its nearest
   labelled centerline voxel, via the Euclidean distance transform (Eq. 3).
6. Scale assignment from S-1 = 2 thresholds computed per volume as the first
   and third quartiles (Q1, Q3) of the branch radii.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

#: Scale indices used in every output (0 is background).
SMALL, MEDIUM, LARGE = 1, 2, 3
SCALE_NAMES = {SMALL: "small", MEDIUM: "medium", LARGE: "large"}


@dataclass
class ClusteringResult:
    """Outputs of :func:`cluster_vessel_scales`.

    Attributes
    ----------
    scale_map : ndarray of uint8
        Same shape as the input mask; 0 background, 1 small, 2 medium,
        3 large vessels.
    branch_map : ndarray of int32
        Every mask voxel labelled with its branch id (1..n_b), 0 elsewhere.
    skeleton : ndarray of bool
        Voxel-level skeleton of the mask.
    branch_radii : ndarray of float
        ``branch_radii[j - 1]`` is the estimated radius of branch j, in the
        units of ``spacing`` (mm when the voxel spacing is given in mm).
    branch_scales : ndarray of uint8
        ``branch_scales[j - 1]`` is the scale (1, 2 or 3) of branch j.
    thresholds : tuple of float
        The two radius thresholds (t1, t2) used to split scales.
    """

    scale_map: np.ndarray
    branch_map: np.ndarray
    skeleton: np.ndarray
    branch_radii: np.ndarray
    branch_scales: np.ndarray
    thresholds: tuple[float, float]
    spacing: tuple[float, ...] = field(default=(1.0, 1.0, 1.0))
    surface: str = "mesh"

    def scale_masks(self) -> dict[int, np.ndarray]:
        """Binary ground-truth mask y^s for each scale s in {1, 2, 3}."""
        return {s: self.scale_map == s for s in (SMALL, MEDIUM, LARGE)}

    def branch_table(self) -> list[dict]:
        """One row per branch: id, radius, scale, number of voxels."""
        n_vox = np.bincount(self.branch_map.ravel(),
                            minlength=len(self.branch_radii) + 1)
        return [
            {
                "branch_id": j + 1,
                "radius": float(r),
                "scale": SCALE_NAMES[int(s)],
                "n_voxels": int(n_vox[j + 1]),
            }
            for j, (r, s) in enumerate(zip(self.branch_radii,
                                           self.branch_scales))
        ]


def _as_spacing(spacing, ndim: int) -> np.ndarray:
    sp = np.asarray(spacing, dtype=float)
    if sp.ndim == 0:
        sp = np.full(ndim, float(sp))
    if sp.shape != (ndim,):
        raise ValueError(f"spacing must be a scalar or have {ndim} values")
    return sp


def surface_voxels(mask: np.ndarray) -> np.ndarray:
    """Edge surface Omega: mask voxels with at least one face-neighbour outside."""
    mask = mask.astype(bool)
    eroded = ndi.binary_erosion(mask, structure=ndi.generate_binary_structure(mask.ndim, 1),
                                border_value=0)
    return mask & ~eroded


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Voxel-level skeleton Gamma by 3D thinning (Lee et al., 1994)."""
    return skeletonize(mask.astype(bool), method="lee").astype(bool)


def surface_points(mask: np.ndarray, spacing=1.0, method: str = "mesh",
                   mesh_sigma: float = 0.5) -> np.ndarray:
    """Points of the edge surface Omega, in physical coordinates.

    method="voxels"
        Centres of the surface voxels (:func:`surface_voxels`). Distances are
        then restricted to a few values (1, sqrt(2), sqrt(3), 2, ...), so thin
        vessels all get the same radius.
    method="mesh"
        Vertices of a sub-voxel surface: marching cubes at level 0.5 on the
        mask smoothed by a Gaussian of ``mesh_sigma`` voxels, with every mask
        voxel kept inside the surface (no vessel is smoothed away). Gives
        continuous radii. Only for 3D masks; 2D masks fall back to "voxels".
    """
    mask = mask.astype(bool)
    sp = _as_spacing(spacing, mask.ndim)
    if method == "voxels" or mask.ndim != 3:
        return np.argwhere(surface_voxels(mask)) * sp
    if method != "mesh":
        raise ValueError(f"unknown surface method {method!r} (use 'mesh' or 'voxels')")
    from skimage.measure import marching_cubes

    if not mask.any():
        return np.zeros((0, mask.ndim))
    padded = np.pad(mask.astype(np.float32), 1)
    level_set = ndi.gaussian_filter(padded, mesh_sigma) if mesh_sigma > 0 else padded
    # keep every vessel voxel inside the surface, so that no thin vessel is
    # smoothed away (its skeleton would then take the radius of a neighbour)
    level_set = np.maximum(level_set, 0.51 * padded)
    verts = marching_cubes(level_set, 0.5)[0] - 1.0
    return verts * sp


def local_radii(skeleton: np.ndarray, surface: np.ndarray, spacing=1.0,
                n_neighbors: int = 8) -> np.ndarray:
    """Local radius sigma_k at every skeleton voxel (Eq. 1).

    sigma_k is the largest distance between the skeleton voxel gamma_k and its
    ``n_neighbors`` nearest surface points.

    Parameters
    ----------
    surface : ndarray
        Either a boolean image of surface voxels (same shape as ``skeleton``)
        or an (N, ndim) array of surface points in physical coordinates, as
        returned by :func:`surface_points`.

    Returns
    -------
    sigma : ndarray of float, same shape as ``skeleton``
        sigma_k at skeleton voxels, 0 elsewhere.
    """
    sp = _as_spacing(spacing, skeleton.ndim)
    skel_idx = np.argwhere(skeleton)
    if surface.shape == skeleton.shape and surface.dtype == bool:
        surf_pts = np.argwhere(surface) * sp
    else:
        surf_pts = np.asarray(surface, dtype=float).reshape(-1, skeleton.ndim)
    sigma = np.zeros(skeleton.shape, dtype=float)
    if len(skel_idx) == 0 or len(surf_pts) == 0:
        return sigma
    k = min(n_neighbors, len(surf_pts))
    tree = cKDTree(surf_pts)
    dist, _ = tree.query(skel_idx * sp, k=k)
    dist = dist.reshape(len(skel_idx), -1)
    sigma[tuple(skel_idx.T)] = dist.max(axis=1)
    return sigma


def label_branches(skeleton: np.ndarray, spacing=1.0) -> np.ndarray:
    """Label each skeleton branch with ``skan.csr.Skeleton.path_label_image``.

    skan splits the skeleton graph at junction and end points; each path gets
    the value ``path_id + 1``. Isolated skeleton pieces that skan does not
    turn into a path (e.g. a single voxel) receive their own extra labels so
    that every skeleton voxel belongs to a branch.

    Returns
    -------
    labels : ndarray of int32
        Branch labels 1..n_b on skeleton voxels, 0 elsewhere.
    """
    from skan.csr import Skeleton  # imported here so the rest works without skan

    labels = np.zeros(skeleton.shape, dtype=np.int32)
    if not skeleton.any():
        return labels
    sk = Skeleton(skeleton.astype(bool), spacing=_as_spacing(spacing, skeleton.ndim))
    labels = sk.path_label_image().astype(np.int32)
    labels[~skeleton] = 0  # path_label_image may round coordinates; stay on Gamma

    leftover = skeleton & (labels == 0)
    if leftover.any():
        extra, n_extra = ndi.label(leftover,
                                   structure=ndi.generate_binary_structure(skeleton.ndim, 3))
        labels[leftover] = extra[leftover] + labels.max()
    return _relabel_consecutive(labels)


def _relabel_consecutive(labels: np.ndarray) -> np.ndarray:
    ids = np.unique(labels)
    ids = ids[ids > 0]
    lut = np.zeros(labels.max() + 1, dtype=np.int32)
    lut[ids] = np.arange(1, len(ids) + 1, dtype=np.int32)
    return lut[labels]


def branch_radii(skeleton_labels: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Branch radius r_j = median of sigma_k over the voxels of branch j (Eq. 2)."""
    n_b = int(skeleton_labels.max())
    if n_b == 0:
        return np.zeros(0)
    return np.asarray(ndi.median(sigma, labels=skeleton_labels,
                                 index=np.arange(1, n_b + 1)), dtype=float)


def reconstruct_branches(mask: np.ndarray, skeleton_labels: np.ndarray,
                         spacing=1.0) -> np.ndarray:
    """Give every mask voxel the label of its nearest centerline voxel (Eq. 3)."""
    mask = mask.astype(bool)
    branch_map = np.zeros(mask.shape, dtype=np.int32)
    if skeleton_labels.max() == 0:
        return branch_map
    _, nearest = ndi.distance_transform_edt(
        skeleton_labels == 0,
        sampling=_as_spacing(spacing, mask.ndim),
        return_distances=True,
        return_indices=True,
    )
    branch_map[mask] = skeleton_labels[tuple(idx[mask] for idx in nearest)]
    return branch_map


def scale_thresholds(radii: np.ndarray, quantiles=(0.25, 0.75)) -> tuple[float, float]:
    """Per-volume thresholds (Q1, Q3) of the branch radii (paper default)."""
    if len(radii) == 0:
        return (0.0, 0.0)
    t1, t2 = np.quantile(radii, quantiles)
    return float(t1), float(t2)


def assign_scales(radii: np.ndarray, thresholds: tuple[float, float]) -> np.ndarray:
    """Scale of each branch: small if r <= t1, large if r > t2, medium otherwise.

    The paper does not state how ties at a threshold are resolved; this
    convention keeps branches equal to Q1 in "small" and equal to Q3 in
    "medium".
    """
    t1, t2 = thresholds
    scales = np.full(len(radii), MEDIUM, dtype=np.uint8)
    scales[radii <= t1] = SMALL
    scales[radii > t2] = LARGE
    return scales


def cluster_vessel_scales(mask: np.ndarray, spacing=1.0, n_neighbors: int = 8,
                          quantiles=(0.25, 0.75),
                          thresholds: tuple[float, float] | None = None,
                          surface: str = "mesh", mesh_sigma: float = 0.5
                          ) -> ClusteringResult:
    """Decompose a binary vessel mask into small / medium / large vessels.

    Parameters
    ----------
    mask : ndarray
        Binary vessel mask y (2D or 3D).
    spacing : float or sequence of float
        Voxel size along each axis. Radii and thresholds are expressed in the
        same unit (pass the voxel size in mm to get radii in mm).
    n_neighbors : int
        m, the number of nearest surface voxels used for the local radius
        (Eq. 1). The paper uses 8.
    quantiles : (float, float)
        Quantiles of the branch radii used as per-volume thresholds. The
        paper uses the first and third quartiles, (0.25, 0.75).
    thresholds : (float, float), optional
        Fixed radius thresholds that override ``quantiles``. Not used in the
        paper; useful to apply the same absolute thresholds to every volume.
    surface : {"mesh", "voxels"}
        Edge surface Omega used in Eq. 1 (see :func:`surface_points`).
        "mesh" (default) gives sub-voxel, continuous radii; "voxels" uses the
        centres of the surface voxels.
    mesh_sigma : float
        Gaussian smoothing, in voxels, applied before meshing (``surface="mesh"``).

    Returns
    -------
    ClusteringResult
    """
    mask = np.asarray(mask).astype(bool)
    sp = _as_spacing(spacing, mask.ndim)

    skeleton = skeletonize_mask(mask)
    sigma = local_radii(skeleton, surface_points(mask, sp, surface, mesh_sigma),
                        sp, n_neighbors)
    skel_labels = label_branches(skeleton, sp)
    radii = branch_radii(skel_labels, sigma)

    t = tuple(map(float, thresholds)) if thresholds is not None \
        else scale_thresholds(radii, quantiles)
    scales = assign_scales(radii, t)

    branch_map = reconstruct_branches(mask, skel_labels, sp)
    lut = np.concatenate([[0], scales]).astype(np.uint8)
    scale_map = lut[branch_map]
    scale_map[~mask] = 0

    return ClusteringResult(scale_map=scale_map, branch_map=branch_map,
                            skeleton=skeleton, branch_radii=radii,
                            branch_scales=scales, thresholds=t,
                            spacing=tuple(float(s) for s in sp), surface=surface)
