"""3D surface renders of a scale decomposition (needs matplotlib).

In the style of an ITK-SNAP 3D view: the skeleton branches, one colour each,
and the 3-scale clustering (small red, medium blue, large gold). The surface is
the marching-cubes surface of the binary mask itself, without smoothing, so
every vessel of the mask is drawn, including the thinnest ones.
"""
from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .clustering import LARGE, MEDIUM, SCALE_NAMES, SMALL

BACKGROUND = "#000000"
INK = "#f2f2f2"
INK_2 = "#b5b5b5"
SCALE_COLORS = {SMALL: "#e5413a", MEDIUM: "#3a8fc4", LARGE: "#bd8a2c"}


def _hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)]) / 255.0


def branch_colors(n: int, seed: int = 0) -> np.ndarray:
    """n distinct colours (golden-ratio hues), shuffled so neighbours differ."""
    hues = (np.arange(n) * 0.618033988749895) % 1.0
    rng = np.random.default_rng(seed)
    sat = rng.uniform(0.5, 0.85, n)
    val = rng.uniform(0.85, 1.0, n)
    return np.array([colorsys.hsv_to_rgb(h, s, v) for h, s, v in zip(hues, sat, val)])


def surface_mesh(mask: np.ndarray, spacing=(1.0, 1.0, 1.0), smooth: float = 0.0):
    """Surface of ``mask`` (marching cubes at level 0.5).

    ``smooth`` > 0 applies a Gaussian of that many voxels first; every mask
    voxel is still kept inside the surface. The default, 0, meshes the binary
    mask as it is.

    Returns vertices (physical units), faces, and for each face the index of
    the nearest mask voxel (a tuple of index arrays) to look up label maps.
    """
    from skimage.measure import marching_cubes

    sp = np.broadcast_to(np.asarray(spacing, float), (3,))
    padded = np.pad(mask.astype(np.float32), 2)
    level_set = padded
    if smooth > 0:
        level_set = np.maximum(ndi.gaussian_filter(padded, smooth), 0.51 * padded)
    verts, faces, _, _ = marching_cubes(level_set, 0.5)
    verts -= 2.0
    _, nearest = ndi.distance_transform_edt(~mask.astype(bool), return_indices=True)
    centres = np.clip(np.rint(verts[faces].mean(axis=1)).astype(int), 0,
                      np.array(mask.shape) - 1)
    near = tuple(n[tuple(centres.T)] for n in nearest)
    return verts * sp, faces, near


def _shade(rgb: np.ndarray, normals: np.ndarray, light: np.ndarray,
           view: np.ndarray) -> np.ndarray:
    """Two-sided Lambert + soft specular shading, like a surface render."""
    diffuse = np.abs(normals @ light)[:, None]
    half = light + view
    half /= np.linalg.norm(half)
    spec = np.abs(normals @ half)[:, None] ** 30
    return np.clip(rgb * (0.38 + 0.75 * diffuse) + 0.22 * spec, 0, 1)


def _direction(elev, azim):
    el, az = np.deg2rad(elev), np.deg2rad(azim)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def _draw(ax, verts, faces, face_rgb, elev, azim, zoom=1.22):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    tri = verts[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    view = _direction(elev, azim)                 # towards the viewer
    light = _direction(elev + 30, azim - 30)      # from the viewer's upper left
    coll = Poly3DCollection(tri, facecolors=_shade(face_rgb, normals, light, view),
                            edgecolors="none", linewidths=0, antialiased=False)
    ax.add_collection3d(coll)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    centre, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.02
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - half, centre[2] + half)
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_facecolor(BACKGROUND)


def _coloured_mesh(mask, branch_map, scale_map, spacing, smooth):
    """Mesh of the mask cropped to its bounding box, with branch and scale colours."""
    nz = np.argwhere(mask)
    lo, hi = nz.min(axis=0), nz.max(axis=0) + 1
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    mask, branch_map, scale_map = mask[box], branch_map[box], scale_map[box]

    verts, faces, near = surface_mesh(mask, spacing, smooth)
    n_b = int(branch_map.max())
    palette = np.vstack([[0.5, 0.5, 0.5], branch_colors(max(n_b, 1))])
    scale_rgb = np.vstack([[0.5, 0.5, 0.5]] + [_hex_to_rgb(SCALE_COLORS[s])
                                               for s in (SMALL, MEDIUM, LARGE)])
    return verts, faces, palette[branch_map[near]], scale_rgb[scale_map[near]], n_b


def _scale_legend(fig, branch_scales=None, y=0.02):
    from matplotlib.patches import Patch
    handles = []
    for s in (SMALL, MEDIUM, LARGE):
        lab = SCALE_NAMES[s]
        if branch_scales is not None:
            n = int((branch_scales == s).sum())
            lab += f" ({n} branch{'es' if n != 1 else ''})"
        handles.append(Patch(color=SCALE_COLORS[s], label=lab))
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.99, y),
               frameon=False, fontsize=9.5, labelcolor=INK_2, ncol=3)


def save_3d_figure(mask: np.ndarray, branch_map: np.ndarray, scale_map: np.ndarray,
                   out_path, spacing=(1.0, 1.0, 1.0), title: str | None = None,
                   branch_scales: np.ndarray | None = None, note: str | None = None,
                   elev: float = 20, azim: float = -60, smooth: float = 0.0) -> Path:
    """Save a PNG with two 3D renders: labelled branches and 3-scale clustering."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    verts, faces, rgb_b, rgb_s, n_b = _coloured_mesh(mask, branch_map, scale_map,
                                                     spacing, smooth)
    fig = plt.figure(figsize=(11, 5.8), facecolor=BACKGROUND)
    axes = [fig.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]
    _draw(axes[0], verts, faces, rgb_b, elev, azim)
    _draw(axes[1], verts, faces, rgb_s, elev, azim)
    top = 0.86 if title else 0.92
    fig.subplots_adjust(left=0, right=1, bottom=0.07, top=top, wspace=0)
    fig.text(0.25, top, f"labelled branches ({n_b})", color=INK, ha="center", fontsize=12)
    fig.text(0.75, top, "3-scale clustering", color=INK, ha="center", fontsize=12)
    _scale_legend(fig, branch_scales)
    if title:
        fig.text(0.01, 0.965, title, color=INK, fontsize=13, va="top")
    if note:
        fig.text(0.01, 0.025, note, color=INK_2, fontsize=8.5)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=BACKGROUND)
    plt.close(fig)
    return out_path


def save_gallery(cases: list[dict], out_path, smooth: float = 0.0) -> Path:
    """One row per case, columns "labelled branches" and "3-scale clustering".

    Each case is a dict with keys ``title``, ``mask``, ``branch_map``,
    ``scale_map`` and optionally ``spacing``, ``elev``, ``azim``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(cases)
    height = 5.0 * n + 0.5
    fig = plt.figure(figsize=(11, height), facecolor=BACKGROUND)
    row_h = 5.0 / height
    top = 1 - 0.1 / height
    for i, c in enumerate(cases):
        verts, faces, rgb_b, rgb_s, n_b = _coloured_mesh(
            c["mask"], c["branch_map"], c["scale_map"], c.get("spacing", (1, 1, 1)), smooth)
        y0 = top - (i + 1) * row_h
        for j, rgb in enumerate((rgb_b, rgb_s)):
            ax = fig.add_axes([j * 0.5, y0, 0.5, row_h * 0.84], projection="3d")
            _draw(ax, verts, faces, rgb, c.get("elev", 20), c.get("azim", -60),
                  zoom=c.get("zoom", 1.25))
        fig.text(0.01, y0 + row_h * 0.985, c["title"], color=INK, fontsize=13, va="top")
        for x, label in ((0.25, f"labelled branches ({n_b})"), (0.75, "3-scale clustering")):
            fig.text(x, y0 + row_h * 0.875, label, color=INK_2, ha="center", fontsize=10.5)
    _scale_legend(fig, y=0.005)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=BACKGROUND)
    plt.close(fig)
    return out_path
