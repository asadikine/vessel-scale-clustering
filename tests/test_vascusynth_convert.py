import importlib.util
from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
Image = pytest.importorskip("PIL.Image")

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "vascusynth" / "vascusynth_to_nifti.py"
spec = importlib.util.spec_from_file_location("vascusynth_to_nifti", SCRIPT)
conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv)


@pytest.fixture
def grey_volume():
    """Anti-aliased tube with VascuSynth-like values 0, 31, ..., 223."""
    vol = np.zeros((20, 24, 16), np.uint8)
    vol[8:12, 10:14, :] = 223
    vol[7, 10:14, :] = 31
    vol[12, 10:14, :] = 95
    return vol


def test_nifti_input_keeps_header(grey_volume, tmp_path):
    affine = np.diag([-0.5, -0.5, 2.0, 1.0])
    ref = nib.Nifti1Image(grey_volume, affine)
    ref.set_qform(affine, code=1)
    ref.set_sform(affine, code=1)
    nib.save(ref, str(tmp_path / "in.nii.gz"))
    conv.main([str(tmp_path / "in.nii.gz"), str(tmp_path / "out.nii.gz")])
    out = nib.load(str(tmp_path / "out.nii.gz"))
    data = np.asanyarray(out.dataobj)
    assert set(np.unique(data)) == {0, 1}
    assert (data == (grey_volume > 0)).all()
    np.testing.assert_allclose(out.affine, affine)


def test_slice_folder_input(grey_volume, tmp_path):
    folder = tmp_path / "original_image"
    folder.mkdir()
    for z in range(grey_volume.shape[2]):
        Image.fromarray(grey_volume[:, :, z].T).save(folder / f"image{z:03d}.png")
    conv.main([str(folder), str(tmp_path / "out.nii.gz"), "--threshold", "50",
               "--spacing", "0.5"])
    out = nib.load(str(tmp_path / "out.nii.gz"))
    assert (np.asanyarray(out.dataobj) == (grey_volume > 50)).all()
    np.testing.assert_allclose(out.header.get_zooms(), (0.5, 0.5, 0.5))
