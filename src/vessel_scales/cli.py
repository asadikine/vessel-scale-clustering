"""Decompose vessel masks into small / medium / large vessels.

examples:
  vessel-scales mask.nii.gz -o results/ --figure
  vessel-scales labelsTr/hepaticvessel_001.nii.gz --labels 1 -o results/ --figure
  vessel-scales masks/*.nii.gz -o results/
  vessel-scales mask.npy --spacing 0.7 0.7 1.6 -o results/

For each input <name>, the output folder (created if needed) receives:
  <name>_scales.nii.gz         scale map: 0 background, 1 small, 2 medium, 3 large
  <name>_small.nii.gz          binary mask of each scale
  <name>_medium.nii.gz
  <name>_large.nii.gz
  <name>_branch_labels.nii.gz  one label per skeleton branch (1..n)
  <name>_itksnap_labels.txt    ITK-SNAP label colours for <name>_scales
  <name>_branches.csv          one row per branch: id, radius, scale, n_voxels
  <name>_summary.json          parameters, thresholds and counts
  <name>_3d.png                3D render: branches and scales (with --figure)

NIfTI outputs keep the input geometry: affine, qform/sform and their codes,
voxel size, orientation and units. Only the data type changes.

To view the result in 3D in ITK-SNAP:
  itksnap -g <name>.nii.gz -s results/<name>_scales.nii.gz -l results/<name>_itksnap_labels.txt
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import __version__
from .clustering import SCALE_NAMES, cluster_vessel_scales

EXTENSIONS = (".nii.gz", ".nii", ".npy")

ITKSNAP_HEADER = """\
################################################
# ITK-SnAP Label Description File
# File format:
# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL
# Fields:
#    IDX:   Zero-based index
#    -R-:   Red color component (0..255)
#    -G-:   Green color component (0..255)
#    -B-:   Blue color component (0..255)
#    -A-:   Label transparency (0.00 .. 1.00)
#    VIS:   Label visibility (0 or 1)
#    MSH:   Label mesh visibility (0 or 1)
#  LABEL:   Label description
################################################
"""


def split_name(path: Path) -> tuple[str, str]:
    """Return (stem, extension), handling the double extension .nii.gz."""
    name = path.name
    for ext in EXTENSIONS:
        if name.endswith(ext):
            return name[: -len(ext)], ext
    raise SystemExit(f"Unsupported file type: {name} (use .nii, .nii.gz or .npy)")


def _nibabel():
    try:
        import nibabel as nib
    except ImportError as e:  # pragma: no cover
        raise SystemExit("Reading NIfTI files needs nibabel: pip install nibabel") from e
    return nib


def load_mask(path: Path, ext: str):
    """Return (data, voxel_size or None, reference image or None)."""
    if ext == ".npy":
        return np.load(path), None, None
    nib = _nibabel()
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    zooms = tuple(float(z) for z in img.header.get_zooms()[: data.ndim])
    return data, zooms, img


def save_like(array: np.ndarray, out: Path, ref, description: str = "",
              dtype=np.uint8) -> None:
    """Save ``array`` as a label image with the geometry and header of ``ref``."""
    array = np.asarray(array, dtype=dtype)
    if ref is None:
        np.save(out, array)
        return
    header = ref.header.copy()
    header.set_data_dtype(dtype)
    img = type(ref)(array, ref.affine, header)
    img.set_qform(ref.get_qform(), code=int(ref.header["qform_code"]))
    img.set_sform(ref.get_sform(), code=int(ref.header["sform_code"]))
    img.header.set_slope_inter(1.0, 0.0)
    img.header["cal_min"], img.header["cal_max"] = 0, int(array.max())
    img.header.set_intent("label")
    img.header["descrip"] = description[:79].encode()
    _nibabel().save(img, str(out))


def write_itksnap_labels(out: Path) -> None:
    from .plot import SCALE_COLORS
    lines = [ITKSNAP_HEADER.rstrip("\n"), '    0     0    0    0        0  0  0    "Clear Label"']
    for s, name in SCALE_NAMES.items():
        h = SCALE_COLORS[s].lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        lines.append(f'{s:5d} {r:5d} {g:4d} {b:4d}        1  1  1    "{name}"')
    out.write_text("\n".join(lines) + "\n")


def select_vessels(data: np.ndarray, labels) -> np.ndarray:
    if labels:
        return np.isin(data, labels)
    return data > 0


def run_one(path: Path, args) -> dict:
    name, ext = split_name(path)
    data, zooms, ref = load_mask(path, ext)
    mask = select_vessels(data, args.labels)
    if not mask.any():
        raise SystemExit(f"{path.name}: no vessel voxel"
                         + (f" with labels {args.labels}" if args.labels else ""))

    if args.spacing is not None:
        spacing = args.spacing[0] if len(args.spacing) == 1 else tuple(args.spacing)
    else:
        spacing = zooms if zooms is not None else 1.0

    res = cluster_vessel_scales(mask, spacing=spacing, n_neighbors=args.neighbors,
                                quantiles=tuple(args.quantiles),
                                thresholds=args.thresholds, surface=args.surface,
                                mesh_sigma=args.mesh_sigma)

    out = args.output
    save_like(res.scale_map, out / f"{name}_scales{ext}", ref,
              "vessel scales: 1 small 2 medium 3 large")
    for s, m in res.scale_masks().items():
        save_like(m, out / f"{name}_{SCALE_NAMES[s]}{ext}", ref,
                  f"vessel scale: {SCALE_NAMES[s]}")
    n_b = len(res.branch_radii)
    save_like(res.branch_map, out / f"{name}_branch_labels{ext}", ref,
              "vessel branch labels", dtype=np.uint16 if n_b < 65535 else np.uint32)
    write_itksnap_labels(out / f"{name}_itksnap_labels.txt")

    with open(out / f"{name}_branches.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["branch_id", "radius", "scale", "n_voxels"])
        w.writeheader()
        w.writerows(res.branch_table())

    counts = {SCALE_NAMES[s]: int(m.sum()) for s, m in res.scale_masks().items()}
    n_branches = {SCALE_NAMES[s]: int((res.branch_scales == s).sum())
                  for s in SCALE_NAMES}
    summary = {
        "input": str(path),
        "version": __version__,
        "labels": args.labels or "all > 0",
        "spacing": list(res.spacing),
        "surface": args.surface,
        "mesh_sigma": args.mesh_sigma if args.surface == "mesh" else None,
        "n_neighbors": args.neighbors,
        "quantiles": None if args.thresholds else list(args.quantiles),
        "thresholds": list(res.thresholds),
        "n_branches": n_b,
        "branches_per_scale": n_branches,
        "voxels_per_scale": counts,
    }
    with open(out / f"{name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if args.figure:
        from .plot import save_3d_figure
        unit = "voxels" if args.spacing is None and ref is None else "mm"
        if ref is not None and args.spacing is None:
            u = ref.header.get_xyzt_units()[0]
            unit = "mm" if u == "unknown" else u
        save_3d_figure(mask, res.branch_map, res.scale_map, out / f"{name}_3d.png",
                       spacing=res.spacing, title=name, branch_scales=res.branch_scales,
                       note=f"Radius thresholds Q1 / Q3: {res.thresholds[0]:.2f} / "
                            f"{res.thresholds[1]:.2f} {unit}",
                       elev=args.view[0], azim=args.view[1])

    print(f"{path.name}: {n_b} branches {n_branches}, thresholds "
          f"{res.thresholds[0]:.3g} / {res.thresholds[1]:.3g} -> {out}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vessel-scales", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", type=Path,
                   help="vessel masks (.nii, .nii.gz or .npy)")
    p.add_argument("-o", "--output", type=Path, default=Path("vessel_scales_output"),
                   help="output folder, created if needed (default: vessel_scales_output)")
    p.add_argument("--labels", type=int, nargs="+",
                   help="label values counted as vessel (default: every value > 0); "
                        "use --labels 1 for MSD Task08, where 2 is tumour")
    p.add_argument("--spacing", type=float, nargs="+",
                   help="voxel size; default: NIfTI header, or 1 for .npy")
    p.add_argument("--surface", choices=("mesh", "voxels"), default="mesh",
                   help="edge surface used for the local radius: sub-voxel mesh "
                        "(default) or surface voxel centres")
    p.add_argument("--mesh-sigma", type=float, default=0.5,
                   help="smoothing in voxels before meshing (default: 0.5)")
    p.add_argument("--neighbors", type=int, default=8,
                   help="m nearest surface points for the local radius (default: 8)")
    p.add_argument("--quantiles", type=float, nargs=2, default=(0.25, 0.75),
                   metavar=("Q_LOW", "Q_HIGH"),
                   help="per-volume radius quantiles (default: 0.25 0.75)")
    p.add_argument("--thresholds", type=float, nargs=2, metavar=("T1", "T2"),
                   help="fixed radius thresholds, overriding --quantiles")
    p.add_argument("--figure", action="store_true",
                   help="also save a 3D render (needs matplotlib)")
    p.add_argument("--view", type=float, nargs=2, default=(20, -60),
                   metavar=("ELEV", "AZIM"),
                   help="camera elevation and azimuth of the 3D render, in degrees")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    for path in args.inputs:
        if not path.exists():
            raise SystemExit(f"File not found: {path}")
        run_one(path, args)


if __name__ == "__main__":
    main()
