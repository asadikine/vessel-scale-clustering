"""Synthetic example: a tree whose radii are known exactly.

    python examples/synthetic/run_synthetic.py

Writes
  examples/data/synthetic_tree.nii.gz   the input mask (1 mm isotropic voxels)
  docs/synthetic_3d.png                 3D render: branches and scales
and prints a Markdown table of true and estimated branch radii for both
surface methods.

The NIfTI file can then be processed like any real mask:
  vessel-scales examples/data/synthetic_tree.nii.gz -o results/ --figure
"""
from pathlib import Path

import numpy as np

from vessel_scales import (SCALE_NAMES, assign_scales, cluster_vessel_scales,
                           scale_thresholds)
from vessel_scales.plot import save_3d_figure
from vessel_scales.synthetic import make_synthetic_tree

ROOT = Path(__file__).resolve().parents[2]


def main():
    mask, true_radius = make_synthetic_tree()
    # (z, y, x) -> (x, y, z) so that the trunk runs along the NIfTI z axis
    mask = np.transpose(mask, (2, 1, 0))
    true_radius = np.transpose(true_radius, (2, 1, 0))

    try:
        import nibabel as nib
        out = ROOT / "examples" / "data" / "synthetic_tree.nii.gz"
        out.parent.mkdir(parents=True, exist_ok=True)
        img = nib.Nifti1Image(mask.astype(np.uint8), np.eye(4))
        img.set_qform(np.eye(4), code=1)
        img.set_sform(np.eye(4), code=1)
        img.header.set_xyzt_units("mm")
        nib.save(img, str(out))
        print(f"saved {out.relative_to(ROOT)}")
    except ImportError:
        print("nibabel not installed: skipping the NIfTI file")

    results = {s: cluster_vessel_scales(mask, surface=s) for s in ("mesh", "voxels")}
    res = results["mesh"]
    fig = save_3d_figure(mask, res.branch_map, res.scale_map,
                         ROOT / "docs" / "synthetic_3d.png",
                         title="Synthetic tree (known radii)",
                         branch_scales=res.branch_scales,
                         note=f"Radius thresholds Q1 / Q3: {res.thresholds[0]:.2f} / "
                              f"{res.thresholds[1]:.2f} mm", elev=25, azim=160)
    print(f"saved {fig.relative_to(ROOT)}\n")

    true = np.array([np.median(true_radius[res.branch_map == j])
                     for j in range(1, len(res.branch_radii) + 1)])
    true_scale = assign_scales(true, scale_thresholds(true))
    print("| True radius (mm) | Scale from true radius | Mesh estimate (mm) | Scale | "
          "Voxel estimate (mm) | Scale |")
    print("|---|---|---|---|---|---|")
    vox = results["voxels"]
    for j in np.argsort(-true):
        print(f"| {true[j]:.1f} | {SCALE_NAMES[int(true_scale[j])]} | "
              f"{res.branch_radii[j]:.2f} | {SCALE_NAMES[int(res.branch_scales[j])]} | "
              f"{vox.branch_radii[j]:.2f} | {SCALE_NAMES[int(vox.branch_scales[j])]} |")
    for name, r in results.items():
        lut = np.concatenate([[0], true_scale])
        agree = (lut[r.branch_map][mask] == r.scale_map[mask]).mean()
        print(f"{name}: voxels in the same scale as with the true radii: {100 * agree:.1f}%")


if __name__ == "__main__":
    main()
