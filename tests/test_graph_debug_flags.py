import numpy as np

from nbh.graph_utils import CellManager


def test_debug_edges_toggle(capsys):
    obs = np.zeros((8, 8), dtype=np.float32)
    pred_mean = np.zeros_like(obs)
    pred_var = np.zeros_like(obs)
    inflated = np.where(obs >= 0.8, np.inf, 1.0)

    manager = CellManager(
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
        connectivity_cfg={
            "graph_pred_wall_threshold": 0.7,
            "graph_unknown_as_occ": True,
            "graph_dilate_diam": 3,
        },
        debug_cfg={"graph_debug_edges": True},
    )

    manager.update_graph(np.array([2, 2]), obs, pred_mean, pred_var, inflated)
    manager.update_graph(np.array([2, 2]), obs, pred_mean, pred_var, inflated)
    out = capsys.readouterr().out
    assert "edge_type" in out

    manager2 = CellManager(
        cell_size=4,
        start_pose=np.array([2, 2]),
        promotion_cfg=manager.promotion_cfg,
        connectivity_cfg=manager.connectivity_cfg,
        debug_cfg={"graph_debug_edges": False},
    )
    manager2.update_graph(np.array([2, 2]), obs, pred_mean, pred_var, inflated)
    manager2.update_graph(np.array([2, 2]), obs, pred_mean, pred_var, inflated)
    out2 = capsys.readouterr().out
    assert "edge_type" not in out2


def test_debug_edge_samples_prints(capsys):
    obs = np.zeros((8, 8), dtype=np.float32)
    pred_mean = np.zeros_like(obs)
    pred_var = np.zeros_like(obs)
    inflated = np.where(obs >= 0.8, np.inf, 1.0)

    manager = CellManager(
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
        connectivity_cfg={
            "graph_pred_wall_threshold": 0.7,
            "graph_unknown_as_occ": True,
            "graph_dilate_diam": 3,
        },
        debug_cfg={"graph_debug_edges": False},
    )

    manager.debug_edge_samples = {((0, 0), (0, 1))}
    manager.update_graph(np.array([2, 2]), obs, pred_mean, pred_var, inflated)
    manager.update_graph(np.array([2, 2]), obs, pred_mean, pred_var, inflated)
    out = capsys.readouterr().out
    assert "[RR DEBUG]" in out


def test_parse_debug_edge_samples():
    from nbh.graph_utils import parse_debug_edge_samples

    samples = parse_debug_edge_samples(
        [
            [[-4, 4], [-3, 4]],
            [[0, 1], [0, 0]],
        ]
    )
    assert ((-4, 4), (-3, 4)) in samples
    assert ((0, 0), (0, 1)) in samples
    assert parse_debug_edge_samples(None) == set()
