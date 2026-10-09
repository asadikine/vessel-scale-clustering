from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from vessel_scales.plot import surface_mesh
from vessel_scales.synthetic import make_synthetic_tree

DATA = Path(__file__).resolve().parents[1] / "examples" / "data"


def _n_surface_pieces(mask):
    verts, faces, _ = surface_mesh(mask)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]])
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                       shape=(len(verts), len(verts)))
    return connected_components(graph, directed=False)[0]


def _n_mask_pieces(mask):
    return ndi.label(mask, structure=ndi.generate_binary_structure(3, 1))[1]


def test_render_keeps_synthetic_tree_in_one_piece():
    mask, _ = make_synthetic_tree()
    assert _n_surface_pieces(mask) == _n_mask_pieces(mask) == 1


@pytest.mark.skipif(not (DATA / "vascusynth_tree.nii.gz").exists(),
                    reason="VascuSynth example data not present")
def test_render_does_not_break_thin_vessels():
    nib = pytest.importorskip("nibabel")
    mask = np.asanyarray(nib.load(str(DATA / "vascusynth_tree.nii.gz")).dataobj) > 0
    # the rendered surface must not split the tree into more pieces than the mask has
    assert _n_surface_pieces(mask) == _n_mask_pieces(mask)
