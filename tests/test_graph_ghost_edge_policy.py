import numpy as np

from nbh.graph_utils import CellManager


def _make_manager(connectivity_cfg=None):
    return CellManager(
        cell_size=4,
        start_pose=np.array([2, 2]),
        promotion_cfg={
            "graph_max_ghost_distance": 1,
            "graph_obs_blocked_ratio": 0.3,
            "graph_unknown_ratio_threshold": 0.5,
            "graph_centroid_blocked_threshold": 0.8,
            "graph_ghost_pred_mean_free_threshold": 0.4,
            "graph_ghost_pred_var_max_threshold": 0.3,
        },
        connectivity_cfg=connectivity_cfg or {},
        debug_cfg={},
    )


def _base_maps():
    obs = np.full((8, 8), 0.5, dtype=np.float32)
    obs[0:4, 0:4] = 0.0
    pred_mean = np.full_like(obs, 0.05)
    pred_var = np.full_like(obs, 0.1)
    return obs, pred_mean, pred_var


def test_ghost_edge_allows_high_pred_below_hard_threshold():
    obs, pred_mean, pred_var = _base_maps()
    pred_mean[0:4, 3] = 0.8
    pred_mean[0:4, 4] = 0.8

    manager = _make_manager({
        "graph_unknown_as_occ": True,
        "graph_pred_wall_threshold": 0.7,
        "graph_pred_hard_wall_threshold": 0.95,
    })

    manager.update_graph(np.array([2, 2]), obs, pred_mean, pred_var, None)
    manager.update_graph(np.array([2, 2]), obs, pred_mean, pred_var, None)

    real = manager.cells.get((0, 0))
    ghost = manager.cells.get((0, 1))
    assert real is not None
    assert ghost is not None
    assert any(n.index == ghost.index for n in real.neighbors)


def test_ghost_edge_blocks_when_pred_exceeds_hard_threshold():
    obs, pred_mean, pred_var = _base_maps()
    pred_mean[0:4, 3] = 0.97
    pred_mean[0:4, 4] = 0.97

    manager = _make_manager({
        "graph_unknown_as_occ": True,
        "graph_pred_wall_threshold": 0.7,
        "graph_pred_hard_wall_threshold": 0.95,
    })

    manager.update_graph(np.array([2, 2]), obs, pred_mean, pred_var, None)
    manager.update_graph(np.array([2, 2]), obs, pred_mean, pred_var, None)

    real = manager.cells.get((0, 0))
    ghost = manager.cells.get((0, 1))
    assert real is not None
    assert ghost is not None
    assert not any(n.index == ghost.index for n in real.neighbors)


def test_rr_edge_invalidates_on_new_wall():
    obs = np.zeros((8, 8), dtype=np.float32)
    pred_mean = np.zeros_like(obs)
    pred_var = np.zeros_like(obs)

    manager = _make_manager({
        "graph_unknown_as_occ": True,
        "graph_pred_wall_threshold": 0.7,
    })

    manager.update_graph(np.array([2, 2]), obs, pred_mean, pred_var, None)
    manager.update_graph(np.array([2, 2]), obs, pred_mean, pred_var, None)

    real = manager.cells.get((0, 0))
    assert real is not None
    assert any(n.index == (0, 1) for n in real.neighbors)

    obs2 = obs.copy()
    obs2[2, 4] = 1.0

    manager.update_graph(np.array([2, 2]), obs2, pred_mean, pred_var, None)
    manager.update_graph(np.array([2, 2]), obs2, pred_mean, pred_var, None)

    real2 = manager.cells.get((0, 0))
    assert real2 is not None
    assert not any(n.index == (0, 1) for n in real2.neighbors)
