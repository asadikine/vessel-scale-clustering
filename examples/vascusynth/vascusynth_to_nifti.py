"""Convert a VascuSynth volume into a binary NIfTI vessel mask.

VascuSynth draws vessels with partial-volume anti-aliasing, so its volumes are
not binary: voxels take values 0, 31, 63, ..., 223 depending on how many
of their sub-samples fall inside a vessel. This script sets every voxel above
``--threshold`` to 1 and every other voxel to 0.

Input can be
  * a folder of 2D slices written by VascuSynth (``original_image/image000.jpg``,
    ``image001.jpg``, ...), or
  * a NIfTI volume (``.nii`` / ``.nii.gz``); its header and geometry are kept.

    python examples/vascusynth/vascusynth_to_nifti.py image1/original_image tree1.nii.gz --spacing 1
    python examples/vascusynth/vascusynth_to_nifti.py testVascuSynth.nii.gz tree.nii.gz

JPEG slices are lossy: compression noise gives small non-zero values in the
background, so use ``--threshold 15`` or higher for them (exact NIfTI/PNG
values allow the default ``--threshold 0``).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

SLICE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def read_slices(folder: Path) -> np.ndarray:
    """Stack the 2D slices of ``folder`` (sorted by slice number) into (x, y, z)."""
    from PIL import Image

    files = [f for f in folder.iterdir() if f.suffix.lower() in SLICE_EXT]
    if not files:
        raise SystemExit(f"No image slices found in {folder}")

    def number(f: Path) -> int:
        m = re.findall(r"\d+", f.stem)
        return int(m[-1]) if m else -1

    files.sort(key=number)
    slices = [np.asarray(Image.open(f).convert("L")) for f in files]
    vol = np.stack(slices, axis=0)            # (z, row=y, col=x)
    return np.transpose(vol, (2, 1, 0))       # (x, y, z), NIfTI order


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="slice folder or NIfTI volume")
    p.add_argument("output", type=Path, help="output .nii or .nii.gz file")
    p.add_argument("--threshold", type=float, default=0,
                   help="voxels with value > threshold become 1 (default: 0)")
    p.add_argument("--spacing", type=float, nargs="+", default=[1.0],
                   help="voxel size for slice folders (default: 1); "
                        "ignored for NIfTI input, whose header is kept")
    args = p.parse_args(argv)

    import nibabel as nib

    if args.input.is_dir():
        data = read_slices(args.input)
        sp = args.spacing * 3 if len(args.spacing) == 1 else args.spacing
        img = nib.Nifti1Image((data > args.threshold).astype(np.uint8),
                              np.diag(list(sp) + [1.0]))
        img.set_qform(img.affine, code=1)
        img.set_sform(img.affine, code=1)
        img.header.set_xyzt_units("mm")
    else:
        ref = nib.load(str(args.input))
        data = np.asanyarray(ref.dataobj)
        header = ref.header.copy()
        header.set_data_dtype(np.uint8)
        img = type(ref)((data > args.threshold).astype(np.uint8), ref.affine, header)
        img.set_qform(ref.get_qform(), code=int(ref.header["qform_code"]))
        img.set_sform(ref.get_sform(), code=int(ref.header["sform_code"]))
        img.header.set_slope_inter(1.0, 0.0)

    img.header["descrip"] = f"VascuSynth binarised (> {args.threshold:g})".encode()[:79]
    img.header["cal_min"], img.header["cal_max"] = 0, 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(args.output))

    values = np.unique(data)
    n = int((data > args.threshold).sum())
    print(f"{args.input.name}: shape {data.shape}, {len(values)} distinct values "
          f"({values.min()}..{values.max()}), {n} vessel voxels -> {args.output}")


if __name__ == "__main__":
    main()
