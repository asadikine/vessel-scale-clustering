"""Compare estimated branch radii with the VascuSynth ground truth.

    python examples/vascusynth/compare_ground_truth.py \
        examples/data/vascusynth_tree.nii.gz treeStructure_12.mat

The tree can be
  * a ``.mat`` file, as in the VascuSynth sample data (struct array ``node``
    with fields type, coord, parent, children, parent_radius; 1-based indices),
  * or the ``tree_structure.xml`` (GXL) file written by the VascuSynth program.
Coordinates are in map voxels and radii in map units; ``--map-voxel-width``
converts radii to voxels (VOXEL_WIDTH of the parameter file, 0.04 by default).
If the image was drawn with another voxel width than the map (last argument of
VascuSynth), pass ``--image-voxel-width`` too.

For every estimated branch, the true radius is the median radius of the true
segments closest to its skeleton voxels. The script prints, for each surface
method, the correlation with the truth and how many branches fall in the same
scale as when the per-volume quartiles are applied to the true radii.
"""
from __future__ import annotations

import argparse

import numpy as np

from vessel_scales import (assign_scales, cluster_vessel_scales,
                           scale_thresholds)


def _load_mat(path):
    import scipy.io as sio
    nodes = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["node"]
    xyz = np.array([np.asarray(n.coord, float) for n in nodes])
    segs = [(xyz[int(n.parent) - 1], xyz[i], float(n.parent_radius))
            for i, n in enumerate(nodes) if n.type != "root"]
    return segs, sum(n.type == "bif" for n in nodes)


def _load_gxl(path):
    import xml.etree.ElementTree as ET
    graph = ET.parse(path).getroot().find("graph")
    pos, n_bif = {}, 0
    for node in graph.findall("node"):
        for attr in node.findall("attr"):
            name = attr.get("name").strip()
            if name == "position":
                pos[node.get("id")] = np.array([float(f.text) for f in attr.iter("float")])
            elif name == "nodeType" and "bif" in attr.find("string").text:
                n_bif += 1
    segs = []
    for edge in graph.findall("edge"):
        radius = next(float(a.find("float").text) for a in edge.findall("attr")
                      if a.get("name").strip() == "radius")
        segs.append((pos[edge.get("from")], pos[edge.get("to")], radius))
    return segs, n_bif


def load_tree(path, map_voxel_width, image_voxel_width=None):
    """Segments (start, end, radius) in image voxels, and number of bifurcations."""
    segs, n_bif = _load_mat(path) if str(path).endswith(".mat") else _load_gxl(path)
    scale = 1.0 if image_voxel_width is None else map_voxel_width / image_voxel_width
    a = np.array([s[0] for s in segs]) * scale
    b = np.array([s[1] for s in segs]) * scale
    r = np.array([s[2] for s in segs]) / (image_voxel_width or map_voxel_width)
    return a, b, r, n_bif


def nearest_segment(points, a, b):
    best = np.full(len(points), np.inf)
    idx = np.zeros(len(points), int)
    for k in range(len(a)):
        ab = b[k] - a[k]
        L = ab @ ab
        t = np.zeros(len(points)) if L == 0 else np.clip(((points - a[k]) @ ab) / L, 0, 1)
        d = np.linalg.norm(points - (a[k] + t[:, None] * ab), axis=1)
        upd = d < best
        best[upd], idx[upd] = d[upd], k
    return idx


def true_branch_radii(res, a, b, r):
    """Median true radius of the segments nearest to each branch's skeleton."""
    from scipy import ndimage as ndi
    skel_idx = np.argwhere(res.skeleton)
    seg = nearest_segment(skel_idx.astype(float), a, b)
    labels = res.branch_map * res.skeleton
    lab = labels[tuple(skel_idx.T)]
    out = np.zeros(len(res.branch_radii))
    for j in range(1, len(out) + 1):
        sel = seg[lab == j]
        if len(sel):
            out[j - 1] = np.median(r[sel])
        else:  # branch whose skeleton voxels were all overwritten at junctions
            vox = np.argwhere(res.branch_map == j).astype(float)
            out[j - 1] = np.median(r[nearest_segment(vox, a, b)])
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mask", help="binary NIfTI mask of the VascuSynth volume")
    p.add_argument("tree", help="VascuSynth tree structure (.mat or tree_structure.xml)")
    p.add_argument("--map-voxel-width", type=float, default=0.04,
                   help="VOXEL_WIDTH of the parameter file (default: 0.04)")
    p.add_argument("--image-voxel-width", type=float,
                   help="voxel width the image was drawn with, if different")
    args = p.parse_args(argv)

    import nibabel as nib
    mask = np.asanyarray(nib.load(args.mask).dataobj) > 0
    a, b, r, n_bif = load_tree(args.tree, args.map_voxel_width, args.image_voxel_width)
    print(f"Tree: {n_bif} bifurcations, {len(r)} segments, true radius "
          f"{r.min():.2f}-{r.max():.2f} voxels\n")
    print("| Surface Ω | Distinct radii | Correlation with true radius | "
          "Median error (voxels) | Small / medium / large | Same scale as truth |")
    print("|---|---|---|---|---|---|")
    for surface in ("voxels", "mesh"):
        res = cluster_vessel_scales(mask, surface=surface)
        t = true_branch_radii(res, a, b, r)
        s_true = assign_scales(t, scale_thresholds(t))
        est = res.branch_radii
        counts = "/".join(str(int((res.branch_scales == s).sum())) for s in (1, 2, 3))
        print(f"| {surface} | {len(np.unique(np.round(est, 4)))} | "
              f"{np.corrcoef(est, t)[0, 1]:.2f} | {np.median(est - t):+.2f} | {counts} | "
              f"{100 * (res.branch_scales == s_true).mean():.0f}% |")
    counts = "/".join(str(int((s_true == s).sum())) for s in (1, 2, 3))
    print(f"\nWith the true radii: {len(t)} branches, small / medium / large {counts}.")


if __name__ == "__main__":
    main()
