import numpy as np
import pytest

from vessel_scales import (
    LARGE, MEDIUM, SMALL, assign_scales, cluster_vessel_scales, scale_thresholds,
)
from vessel_scales.synthetic import make_synthetic_tree


@pytest.fixture(scope="module")
def tree():
    mask, true_radius = make_synthetic_tree()
    return mask, true_radius, cluster_vessel_scales(mask)


def test_scales_partition_the_mask(tree):
    mask, _, res = tree
    assert ((res.scale_map > 0) == mask).all()
    masks = res.scale_masks()
    assert (masks[SMALL] | masks[MEDIUM] | masks[LARGE]).sum() == mask.sum()
    assert not (masks[SMALL] & masks[LARGE]).any()


def _true_branch_radii(res, true_radius):
    return np.array([np.median(true_radius[res.branch_map == j])
                     for j in range(1, len(res.branch_radii) + 1)])


def test_one_branch_per_segment(tree):
    _, true_radius, res = tree
    assert len(res.branch_radii) == 7
    np.testing.assert_allclose(np.sort(_true_branch_radii(res, true_radius)),
                               [1.3, 1.6, 2.0, 2.4, 3.0, 4.0, 6.0])
    assert (res.branch_scales == LARGE).sum() == 2
    assert (res.branch_scales == MEDIUM).sum() == 3
    assert (res.branch_scales == SMALL).sum() == 2


def test_mesh_radius_accuracy(tree):
    _, true_radius, res = tree
    err = res.branch_radii - _true_branch_radii(res, true_radius)
    assert np.abs(err).max() < 0.5          # sub-voxel accuracy on every branch


def test_radius_ordering(tree):
    _, _, res = tree
    r = np.sort(res.branch_radii)
    assert r[-1] > r[-2] >= r[-3] > r[0]


@pytest.mark.parametrize("surface, min_agreement", [("mesh", 0.99), ("voxels", 0.9)])
def test_scales_match_true_radius_quartiles(tree, surface, min_agreement):
    mask, true_radius, _ = tree
    res = cluster_vessel_scales(mask, surface=surface)
    true = _true_branch_radii(res, true_radius)
    lut = np.concatenate([[0], assign_scales(true, scale_thresholds(true))])
    assert (lut[res.branch_map][mask] == res.scale_map[mask]).mean() >= min_agreement


def test_radii_scale_with_spacing(tree):
    mask, _, res = tree
    res2 = cluster_vessel_scales(mask, spacing=2.0)
    np.testing.assert_allclose(res2.branch_radii, 2 * res.branch_radii)
    np.testing.assert_array_equal(res2.scale_map, res.scale_map)


def test_fixed_thresholds_override_quartiles(tree):
    mask, _, _ = tree
    res = cluster_vessel_scales(mask, thresholds=(0.0, 100.0))
    assert (res.scale_map[mask] == MEDIUM).all()
    assert res.thresholds == (0.0, 100.0)


def test_tie_convention():
    r = np.array([1.0, 1.0, 2.0, 3.0, 3.0])
    np.testing.assert_array_equal(assign_scales(r, (1.0, 3.0)),
                                  [SMALL, SMALL, MEDIUM, MEDIUM, MEDIUM])


@pytest.mark.parametrize("surface", ["mesh", "voxels"])
def test_empty_mask(surface):
    res = cluster_vessel_scales(np.zeros((16, 16, 16), bool), surface=surface)
    assert res.scale_map.sum() == 0
    assert len(res.branch_radii) == 0


def test_unknown_surface(tree):
    mask, _, _ = tree
    with pytest.raises(ValueError):
        cluster_vessel_scales(mask, surface="nope")
