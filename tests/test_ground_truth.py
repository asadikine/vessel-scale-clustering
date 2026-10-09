import importlib.util
from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")
pytest.importorskip("scipy.io")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data"
spec = importlib.util.spec_from_file_location(
    "compare_ground_truth", ROOT / "examples" / "vascusynth" / "compare_ground_truth.py")
gt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gt)

pytestmark = pytest.mark.skipif(not (DATA / "vascusynth_tree_structure.mat").exists(),
                                reason="VascuSynth example data not present")


def test_mesh_radii_follow_vascusynth_ground_truth():
    from vessel_scales import cluster_vessel_scales
    mask = np.asanyarray(nib.load(str(DATA / "vascusynth_tree.nii.gz")).dataobj) > 0
    a, b, r, n_bif = gt.load_tree(DATA / "vascusynth_tree_structure.mat", 0.04)
    assert n_bif == 56
    res = cluster_vessel_scales(mask)
    true = gt.true_branch_radii(res, a, b, r)
    assert np.corrcoef(res.branch_radii, true)[0, 1] > 0.9


def test_script_runs(capsys):
    gt.main([str(DATA / "vascusynth_tree.nii.gz"), str(DATA / "vascusynth_tree_structure.mat")])
    out = capsys.readouterr().out
    assert "| mesh |" in out and "| voxels |" in out
