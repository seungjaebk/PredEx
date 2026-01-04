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


def _free_grid(shape, wall_col=None):
    obs = np.zeros(shape, dtype=np.float32)
    if wall_col is not None:
        obs[:, wall_col] = 1.0
    return obs


def test_rr_thin_wall_blocks_edge():
    obs = _free_grid((8, 8), wall_col=4)
    pred_mean = np.zeros_like(obs)
    pred_var = np.zeros_like(obs)
    inflated = np.where(obs >= 0.8, np.inf, 1.0)

    manager = _make_manager({
        "graph_portal_fallback_max_obs_ratio": 0.2,
        "graph_pred_wall_threshold": 0.7,
        "graph_unknown_as_occ": True,
        "graph_dilate_diam": 3,
    })

    manager.update_graph(
        robot_pose=np.array([2, 2]),
        obs_map=obs,
        pred_mean_map=pred_mean,
        pred_var_map=pred_var,
        inflated_occ_grid=inflated,
    )
    manager.update_graph(
        robot_pose=np.array([2, 2]),
        obs_map=obs,
        pred_mean_map=pred_mean,
        pred_var_map=pred_var,
        inflated_occ_grid=inflated,
    )

    node = manager.cells.get((0, 0))
    assert node is not None
    assert not any(n.index == (0, 1) for n in node.neighbors)


def test_portal_false_negative_falls_back_to_mini_astar():
    obs = _free_grid((8, 8), wall_col=None)
    pred_mean = np.zeros_like(obs)
    pred_var = np.zeros_like(obs)
    inflated = np.full_like(obs, np.inf)  # portal will be False

    manager = _make_manager({
        "graph_portal_fallback_max_obs_ratio": 0.2,
        "graph_pred_wall_threshold": 0.7,
        "graph_unknown_as_occ": True,
        "graph_dilate_diam": 3,
    })

    manager.update_graph(
        robot_pose=np.array([2, 2]),
        obs_map=obs,
        pred_mean_map=pred_mean,
        pred_var_map=pred_var,
        inflated_occ_grid=inflated,
    )
    manager.update_graph(
        robot_pose=np.array([2, 2]),
        obs_map=obs,
        pred_mean_map=pred_mean,
        pred_var_map=pred_var,
        inflated_occ_grid=inflated,
    )

    node = manager.cells.get((0, 0))
    assert node is not None
    assert any(n.index == (0, 1) for n in node.neighbors)


def test_rr_mini_astar_is_local_patch():
    obs = _free_grid((12, 12), wall_col=None)
    obs[0:6, 4] = 1.0  # local barrier; gap below local patch
    pred_mean = np.zeros_like(obs)
    pred_var = np.zeros_like(obs)
    inflated = np.full_like(obs, np.inf)  # portal will be False

    manager = _make_manager({
        "graph_portal_fallback_max_obs_ratio": 0.2,
        "graph_pred_wall_threshold": 0.7,
        "graph_unknown_as_occ": True,
        "graph_dilate_diam": 3,
    })

    manager.update_graph(
        robot_pose=np.array([2, 2]),
        obs_map=obs,
        pred_mean_map=pred_mean,
        pred_var_map=pred_var,
        inflated_occ_grid=inflated,
    )
    manager.update_graph(
        robot_pose=np.array([2, 2]),
        obs_map=obs,
        pred_mean_map=pred_mean,
        pred_var_map=pred_var,
        inflated_occ_grid=inflated,
    )

    node = manager.cells.get((0, 0))
    assert node is not None
    assert not any(n.index == (0, 1) for n in node.neighbors)


def test_los_side_lines_allow_clear_offset():
    obs = _free_grid((8, 8), wall_col=None)
    obs[2, 4] = 1.0  # block the exact centroid line only

    manager = _make_manager()

    assert manager._check_path_clear((0, 0), (0, 1), obs)
