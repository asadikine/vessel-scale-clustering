"""Render the README overview: synthetic tree and VascuSynth tree, side by side.

    python examples/make_gallery.py      # writes docs/overview.png
"""
from pathlib import Path

import nibabel as nib
import numpy as np

from vessel_scales import cluster_vessel_scales
from vessel_scales.plot import save_gallery

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data"

# title, file, camera elevation, azimuth, zoom
CASES = [
    ("Synthetic tree (7 branches, known radii)", "synthetic_tree.nii.gz", 25, 160, 1.3),
    ("VascuSynth tree (56 bifurcations)", "vascusynth_tree.nii.gz", 20, -60, 1.12),
]


def main():
    cases = []
    for title, name, elev, azim, zoom in CASES:
        img = nib.load(str(DATA / name))
        mask = np.asanyarray(img.dataobj) > 0
        spacing = img.header.get_zooms()[:3]
        res = cluster_vessel_scales(mask, spacing=spacing)
        cases.append(dict(title=title, mask=mask, branch_map=res.branch_map,
                          scale_map=res.scale_map, spacing=spacing, elev=elev, azim=azim,
                          zoom=zoom))
    out = save_gallery(cases, ROOT / "docs" / "overview.png")
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
