import numpy as np

from nbh.graph_utils import CellManager


def _make_manager(graph_cfg=None):
    cfg = graph_cfg or {}
    return CellManager(
        cell_size=4,
        start_pose=np.array([2, 2]),
        promotion_cfg=cfg,
        connectivity_cfg={},
        debug_cfg={},
    )


def _build_maps(pred_mean_val=0.2, pred_var_val=0.1):
    obs = np.full((8, 8), 0.5, dtype=np.float32)
    obs[0:4, 0:4] = 0.0  # make cell (0,0) observed free

    pred_mean = np.full((8, 8), pred_mean_val, dtype=np.float32)
    pred_var = np.full((8, 8), pred_var_val, dtype=np.float32)
    return obs, pred_mean, pred_var


def test_ghost_promotion_uses_patch_thresholds():
    obs, pred_mean, pred_var = _build_maps(pred_mean_val=0.2, pred_var_val=0.1)
    manager = _make_manager({
        "graph_max_ghost_distance": 1,
        "graph_ghost_pred_mean_free_threshold": 0.4,
        "graph_ghost_pred_var_max_threshold": 0.3,
        "graph_obs_blocked_ratio": 0.3,
        "graph_unknown_ratio_threshold": 0.5,
        "graph_centroid_blocked_threshold": 0.8,
    })

    manager.update_graph(
        robot_pose=np.array([2, 2]),
        obs_map=obs,
        pred_mean_map=pred_mean,
        pred_var_map=pred_var,
        inflated_occ_grid=None,
    )
    manager.update_graph(
        robot_pose=np.array([2, 2]),
        obs_map=obs,
        pred_mean_map=pred_mean,
        pred_var_map=pred_var,
        inflated_occ_grid=None,
    )

    ghost = manager.cells.get((0, 1))
    assert ghost is not None
    assert ghost.is_ghost is True


def test_ghost_promotion_respects_var_threshold():
    obs, pred_mean, pred_var = _build_maps(pred_mean_val=0.2, pred_var_val=0.5)
    manager = _make_manager({
        "graph_max_ghost_distance": 1,
        "graph_ghost_pred_mean_free_threshold": 0.4,
        "graph_ghost_pred_var_max_threshold": 0.3,
        "graph_obs_blocked_ratio": 0.3,
        "graph_unknown_ratio_threshold": 0.5,
        "graph_centroid_blocked_threshold": 0.8,
    })

    manager.update_graph(
        robot_pose=np.array([2, 2]),
        obs_map=obs,
        pred_mean_map=pred_mean,
        pred_var_map=pred_var,
        inflated_occ_grid=None,
    )
    manager.update_graph(
        robot_pose=np.array([2, 2]),
        obs_map=obs,
        pred_mean_map=pred_mean,
        pred_var_map=pred_var,
        inflated_occ_grid=None,
    )

    ghost = manager.cells.get((0, 1))
    assert ghost is None


def test_diffuse_on_update_gates_propagation():
    obs, pred_mean, pred_var = _build_maps(pred_mean_val=0.2, pred_var_val=0.1)
    manager = _make_manager({
        "graph_max_ghost_distance": 1,
        "graph_ghost_pred_mean_free_threshold": 0.4,
        "graph_ghost_pred_var_max_threshold": 0.3,
        "graph_obs_blocked_ratio": 0.3,
        "graph_unknown_ratio_threshold": 0.5,
        "graph_centroid_blocked_threshold": 0.8,
        "graph_diffuse_on_update": False,
    })

    manager.update_graph(
        robot_pose=np.array([2, 2]),
        obs_map=obs,
        pred_mean_map=pred_mean,
        pred_var_map=pred_var,
        inflated_occ_grid=None,
    )
    manager.update_graph(
        robot_pose=np.array([2, 2]),
        obs_map=obs,
        pred_mean_map=pred_mean,
        pred_var_map=pred_var,
        inflated_occ_grid=None,
    )

    ghost = manager.cells.get((0, 1))
    assert ghost is not None
    assert ghost.base_value > 0.0
    assert ghost.propagated_value == 0.0

    manager_on = _make_manager({
        "graph_max_ghost_distance": 1,
        "graph_ghost_pred_mean_free_threshold": 0.4,
        "graph_ghost_pred_var_max_threshold": 0.3,
        "graph_obs_blocked_ratio": 0.3,
        "graph_unknown_ratio_threshold": 0.5,
        "graph_centroid_blocked_threshold": 0.8,
        "graph_diffuse_on_update": True,
    })

    manager_on.update_graph(
        robot_pose=np.array([2, 2]),
        obs_map=obs,
        pred_mean_map=pred_mean,
        pred_var_map=pred_var,
        inflated_occ_grid=None,
    )
    manager_on.update_graph(
        robot_pose=np.array([2, 2]),
        obs_map=obs,
        pred_mean_map=pred_mean,
        pred_var_map=pred_var,
        inflated_occ_grid=None,
    )

    ghost_on = manager_on.cells.get((0, 1))
    assert ghost_on is not None
    assert ghost_on.propagated_value > 0.0
