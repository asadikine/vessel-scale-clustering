import json

import numpy as np
import pytest

from vessel_scales import cli
from vessel_scales.synthetic import make_synthetic_tree

nib = pytest.importorskip("nibabel")


@pytest.fixture
def msd_like_nifti(tmp_path):
    """Synthetic tree as label 1 plus a 'tumour' blob as label 2, with an
    oblique affine and anisotropic voxels, like an MSD Task08 label file."""
    mask, _ = make_synthetic_tree()
    data = mask.astype(np.int16)
    zz, yy, xx = np.indices(data.shape)
    data[(zz - 20) ** 2 + (yy - 80) ** 2 + (xx - 80) ** 2 < 8 ** 2] = 2

    theta = np.deg2rad(10)
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    affine = np.eye(4)
    affine[:3, :3] = rot @ np.diag([0.78, 0.78, 2.5])
    affine[:3, 3] = [-200.0, -150.0, 30.0]
    img = nib.Nifti1Image(data, affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=2)
    img.header.set_xyzt_units("mm", "sec")
    path = tmp_path / "hepaticvessel_000.nii.gz"
    nib.save(img, str(path))
    return path, img


def test_outputs_keep_geometry(msd_like_nifti, tmp_path):
    path, ref = msd_like_nifti
    out = tmp_path / "my" / "results"          # nested folder that does not exist
    cli.main([str(path), "--labels", "1", "-o", str(out), "--figure"])

    for suffix in ("scales", "small", "medium", "large"):
        img = nib.load(str(out / f"hepaticvessel_000_{suffix}.nii.gz"))
        np.testing.assert_allclose(img.affine, ref.affine, atol=1e-5)
        np.testing.assert_allclose(img.header.get_zooms(), ref.header.get_zooms(),
                                   atol=1e-5)
        assert img.shape == ref.shape
        assert img.get_data_dtype() == np.uint8
        assert int(img.header["qform_code"]) == 1
        assert int(img.header["sform_code"]) == 2
        assert img.header.get_xyzt_units() == ("mm", "sec")
        assert nib.aff2axcodes(img.affine) == nib.aff2axcodes(ref.affine)

    scales = np.asarray(nib.load(str(out / "hepaticvessel_000_scales.nii.gz")).dataobj)
    labels = np.asarray(ref.dataobj)
    assert ((scales > 0) == (labels == 1)).all()     # tumour (label 2) excluded

    summary = json.loads((out / "hepaticvessel_000_summary.json").read_text())
    np.testing.assert_allclose(summary["spacing"], [0.78, 0.78, 2.5], atol=1e-5)
    assert summary["labels"] == [1]
    assert sum(summary["voxels_per_scale"].values()) == int((labels == 1).sum())
    assert (out / "hepaticvessel_000_branches.csv").exists()
    assert (out / "hepaticvessel_000_3d.png").stat().st_size > 0
    assert summary["surface"] == "mesh"

    branches = nib.load(str(out / "hepaticvessel_000_branch_labels.nii.gz"))
    np.testing.assert_allclose(branches.affine, ref.affine, atol=1e-5)
    bl = np.asanyarray(branches.dataobj)
    assert bl.max() == summary["n_branches"]
    assert ((bl > 0) == (labels == 1)).all()

    lines = (out / "hepaticvessel_000_itksnap_labels.txt").read_text().splitlines()
    entries = [l.split() for l in lines if l and not l.startswith("#")]
    assert [e[0] for e in entries] == ["0", "1", "2", "3"]
    assert entries[1][-1] == '"small"'
    assert entries[3][-1] == '"large"'


def test_all_positive_labels_by_default(msd_like_nifti, tmp_path):
    path, ref = msd_like_nifti
    cli.main([str(path), "-o", str(tmp_path / "out")])
    scales = np.asarray(nib.load(str(tmp_path / "out/hepaticvessel_000_scales.nii.gz")).dataobj)
    assert ((scales > 0) == (np.asarray(ref.dataobj) > 0)).all()


def test_npy_input(tmp_path):
    mask, _ = make_synthetic_tree()
    np.save(tmp_path / "tree.npy", mask)
    cli.main([str(tmp_path / "tree.npy"), "--spacing", "0.5", "-o", str(tmp_path / "o")])
    out = np.load(tmp_path / "o/tree_scales.npy")
    assert ((out > 0) == mask).all()
    summary = json.loads((tmp_path / "o/tree_summary.json").read_text())
    assert summary["spacing"] == [0.5, 0.5, 0.5]


def test_missing_file(tmp_path):
    with pytest.raises(SystemExit):
        cli.main([str(tmp_path / "nope.nii.gz")])


def test_split_name():
    from pathlib import Path
    assert cli.split_name(Path("a/case.nii.gz")) == ("case", ".nii.gz")
    assert cli.split_name(Path("case.nii")) == ("case", ".nii")
