"""Small synthetic vascular trees with known radii, for tests and demos."""
from __future__ import annotations

import numpy as np

# (start, end, radius) in voxels; a trunk, two children and four leaves.
# Every branch has its own radius, so the per-volume quartiles of the true
# radii give a well-defined reference: small 1.3 and 1.6, medium 2.0, 2.4
# and 3.0, large 4.0 and 6.0.
DEFAULT_SEGMENTS = [
    ((8, 48, 48), (40, 48, 48), 6.0),      # trunk
    ((40, 48, 48), (64, 28, 34), 4.0),     # child A
    ((40, 48, 48), (64, 68, 62), 3.0),     # child B
    ((64, 28, 34), (88, 14, 22), 2.4),     # leaves of A
    ((64, 28, 34), (88, 30, 14), 2.0),
    ((64, 68, 62), (88, 82, 74), 1.6),     # leaves of B
    ((64, 68, 62), (88, 66, 82), 1.3),
]


def _segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((points - a) @ ab) / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def make_synthetic_tree(shape=(96, 96, 96), segments=DEFAULT_SEGMENTS):
    """Rasterise a tree of capsules (cylinders with rounded ends).

    Returns
    -------
    mask : ndarray of bool
    true_radius : ndarray of float
        Radius of the segment closest to each mask voxel, 0 outside the mask.
    """
    grid = np.indices(shape).reshape(len(shape), -1).T.astype(float)
    best = np.full(len(grid), np.inf)
    radius = np.zeros(len(grid))
    inside = np.zeros(len(grid), dtype=bool)
    for a, b, r in segments:
        d = _segment_distance(grid, np.asarray(a, float), np.asarray(b, float))
        hit = d <= r
        inside |= hit
        closer = hit & (d / r < best)
        best[closer] = d[closer] / r
        radius[closer] = r
    return inside.reshape(shape), radius.reshape(shape)
