import numpy as np

from nbh.flow_viz_utils import normalize_edge_risks


def test_normalize_edge_risks_handles_flat():
    costs = np.array([0.5, 0.5, 0.5], dtype=float)
    norm = normalize_edge_risks(costs)
    assert np.allclose(norm, 0.0)


def test_normalize_edge_risks_scales_between_0_and_1():
    costs = np.array([0.1, 0.4, 0.9], dtype=float)
    norm = normalize_edge_risks(costs)
    assert np.isclose(norm.min(), 0.0)
    assert np.isclose(norm.max(), 1.0)
