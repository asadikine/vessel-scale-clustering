"""Convert a 3D-IRCADb-01 patient into a binary NIfTI vessel mask.

Each 3D-IRCADb-01 patient folder (``3Dircadb1.1`` ... ``3Dircadb1.20``) holds a
``MASKS_DICOM`` folder (or ``MASKS_DICOM.zip``) with one DICOM series per
segmented structure. This script merges the vessel structures you choose into
one binary mask (1 = vessel), with the geometry of the DICOM series (voxel
size, orientation, position) written into the NIfTI header.

    python examples/ircadb/ircadb_to_nifti.py 3Dircadb1.12 --list
    python examples/ircadb/ircadb_to_nifti.py 3Dircadb1.12 ircadb12_vessels.nii.gz
    python examples/ircadb/ircadb_to_nifti.py 3Dircadb1.12 ircadb12_vessels.nii.gz \
        --structures portalvein venoussystem --ct ircadb12_ct.nii.gz

By default the hepatic vessels are kept: portal vein and venous system
(hepatic veins). Structure names are matched case-insensitively and by prefix,
so ``portalvein`` also matches ``portalvein1``.
"""
from __future__ import annotations

import argparse
import re
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT_STRUCTURES = ("portalvein", "venoussystem")


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _unzip_if_needed(path: Path, workdir: Path) -> Path:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            z.extractall(workdir / path.stem)
        return workdir / path.stem
    return path


def find_series(root: Path, folder: str, workdir: Path) -> dict[str, list[Path]]:
    """DICOM files of ``folder`` (MASKS_DICOM or PATIENT_DICOM), grouped by sub-folder."""
    candidates = [root / folder, root / f"{folder}.zip", root]
    base = next((c for c in candidates if c.exists() and c.name.lower().startswith(folder.lower())),
                None)
    if base is None:
        raise SystemExit(f"No {folder} folder or {folder}.zip in {root}")
    base = _unzip_if_needed(base, workdir)
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in base.rglob("*"):
        if f.is_file() and not f.name.startswith("."):
            rel = f.relative_to(base).parts
            # skip a top folder repeated inside the zip (MASKS_DICOM/MASKS_DICOM/...)
            parts = [p for p in rel[:-1] if p.lower() != folder.lower()]
            groups[parts[-1] if parts else folder].append(f)
    return dict(groups)


def read_series(files: list[Path], rescale: bool = False):
    """Read one DICOM series; return (volume indexed x, y, z) and a NIfTI (RAS) affine."""
    import pydicom

    slices = [pydicom.dcmread(str(f), force=True) for f in files]
    slices = [s for s in slices if "PixelData" in s]
    if not slices:
        raise SystemExit(f"No DICOM image in {files[0].parent}")
    first = slices[0]
    iop = getattr(first, "ImageOrientationPatient", None) or [1, 0, 0, 0, 1, 0]
    row, col = np.array(iop[:3], float), np.array(iop[3:], float)
    normal = np.cross(row, col)
    has_position = all("ImagePositionPatient" in s for s in slices)
    if has_position:
        slices.sort(key=lambda s: float(np.dot(normal, np.array(s.ImagePositionPatient, float))))
    else:   # fall back on the instance number, then on the file order
        slices.sort(key=lambda s: int(getattr(s, "InstanceNumber", 0) or 0))

    d_row, d_col = (float(v) for v in getattr(slices[0], "PixelSpacing", [1.0, 1.0]))
    thickness = float(getattr(slices[0], "SpacingBetweenSlices", 0) or
                      getattr(slices[0], "SliceThickness", 1.0) or 1.0)
    if has_position:
        p0 = np.array(slices[0].ImagePositionPatient, float)
        if len(slices) > 1:
            step = (np.array(slices[-1].ImagePositionPatient, float) - p0) / (len(slices) - 1)
        else:
            step = normal * thickness
    else:
        p0, step = np.zeros(3), normal * thickness

    planes = []
    for s in slices:
        a = s.pixel_array.astype(np.float32)
        if rescale:
            a = a * float(getattr(s, "RescaleSlope", 1)) + float(getattr(s, "RescaleIntercept", 0))
        planes.append(a.T)                       # (columns, rows) = (x, y)
    volume = np.stack(planes, axis=-1)           # (x, y, z)

    lps = np.eye(4)
    lps[:3, 0] = row * d_col                     # +1 column  -> along the row direction
    lps[:3, 1] = col * d_row                     # +1 row     -> along the column direction
    lps[:3, 2] = step
    lps[:3, 3] = p0
    affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps   # DICOM LPS -> NIfTI RAS
    return volume, affine


def _save(volume, affine, out: Path, dtype, description: str) -> None:
    import nibabel as nib
    img = nib.Nifti1Image(volume.astype(dtype), affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    img.header.set_xyzt_units("mm")
    img.header["descrip"] = description[:79].encode()
    out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("patient", type=Path,
                   help="patient folder (3Dircadb1.N), its MASKS_DICOM folder, or MASKS_DICOM.zip")
    p.add_argument("output", type=Path, nargs="?", help="output vessel mask (.nii.gz)")
    p.add_argument("--structures", nargs="+", default=list(DEFAULT_STRUCTURES),
                   help="structures merged into the vessel mask "
                        "(default: portalvein venoussystem)")
    p.add_argument("--ct", type=Path, help="also save the CT image (PATIENT_DICOM) here")
    p.add_argument("--list", action="store_true", help="list the available structures and exit")
    args = p.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        patient = args.patient
        if not patient.exists():
            raise SystemExit(f"Not found: {patient}")
        if patient.name.lower().startswith("masks_dicom"):
            root = patient.parent
        elif patient.suffix.lower() == ".zip":          # a whole patient downloaded as a zip
            extracted = _unzip_if_needed(patient, tmp)
            hits = [d for d in extracted.rglob("*") if d.name.lower().startswith("masks_dicom")]
            if not hits:
                raise SystemExit(f"No MASKS_DICOM in {patient}")
            root = hits[0].parent
        else:
            root = patient
        masks = find_series(root, "MASKS_DICOM", tmp)
        if args.list or args.output is None:
            print("Structures:", ", ".join(sorted(masks)))
            return

        wanted = [_norm(s) for s in args.structures]
        chosen = sorted(k for k in masks if any(_norm(k).startswith(w) for w in wanted))
        if not chosen:
            raise SystemExit(f"None of {args.structures} found. Available: {', '.join(sorted(masks))}")

        merged, affine = None, None
        for name in chosen:
            vol, aff = read_series(masks[name])
            if merged is None:
                merged, affine = vol > 0, aff
            elif vol.shape != merged.shape or not np.allclose(aff, affine, atol=1e-3):
                raise SystemExit(f"{name}: geometry differs from {chosen[0]}")
            else:
                merged |= vol > 0
        _save(merged, affine, args.output, np.uint8, "IRCADb vessels: " + "+".join(chosen))
        zooms = np.linalg.norm(affine[:3, :3], axis=0)
        print(f"{args.output}: {', '.join(chosen)} -> {int(merged.sum())} vessel voxels, "
              f"shape {merged.shape}, voxel size {np.round(zooms, 3).tolist()} mm")

        if args.ct:
            ct_series = find_series(root, "PATIENT_DICOM", tmp)
            files = max(ct_series.values(), key=len)
            ct, ct_aff = read_series(files, rescale=True)
            _save(ct, ct_aff, args.ct, np.int16, "IRCADb CT")
            print(f"{args.ct}: CT image, shape {ct.shape}")


if __name__ == "__main__":
    main()
