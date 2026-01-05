import numpy as np

from nbh.graph_utils import CellManager


def _make_manager(connectivity_cfg=None):
    return CellManager(
        cell_size=4,
        start_pose=np.array([2, 2]),
        promotion_cfg={
            "graph_max_ghost_distance": 2,
            "graph_obs_blocked_ratio": 0.3,
            "graph_unknown_ratio_threshold": 0.5,
            "graph_centroid_blocked_threshold": 0.8,
            "graph_ghost_pred_mean_free_threshold": 0.4,
            "graph_ghost_pred_var_max_threshold": 0.3,
        },
        connectivity_cfg=connectivity_cfg or {},
        debug_cfg={},
    )


def test_dijkstra_prefers_lower_risk_path():
    obs = np.full((8, 8), 0.5, dtype=np.float32)
    obs[0:4, 0:4] = 0.0  # seed real cell at (0, 0)

    pred_mean = np.full_like(obs, 0.05)
    pred_var = np.zeros_like(obs)
    pred_mean[0:4, 4:8] = 0.35  # higher risk on the right cell

    manager = _make_manager({
        "graph_unknown_as_occ": False,
        "graph_pred_wall_threshold": 0.7,
        "graph_mini_astar_ttl_steps": 10,
        "graph_edge_risk_strip_divisor": 6,
        "graph_edge_risk_eps": 1e-4,
    })

    for _ in range(2):
        manager.update_graph(
            robot_pose=np.array([2, 2]),
            obs_map=obs,
            pred_mean_map=pred_mean,
            pred_var_map=pred_var,
            inflated_occ_grid=None,
        )

    start = manager.cells.get((0, 0))
    target = manager.cells.get((1, 1))
    assert start is not None
    assert target is not None

    path = manager.find_path_to_target(start, target)
    assert path is not None
    indices = [cell.index for cell in path]
    assert indices == [(0, 0), (1, 0), (1, 1)]
