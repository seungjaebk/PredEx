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


def _build_maps():
    obs = np.full((8, 8), 0.5, dtype=np.float32)
    obs[0:4, 0:4] = 0.0
    pred_mean = np.full_like(obs, 0.2)
    pred_var = np.full_like(obs, 0.1)
    inflated = np.where(obs >= 0.8, np.inf, 1.0)
    return obs, pred_mean, pred_var, inflated


def test_mini_astar_cache_reuses_result():
    obs, pred_mean, pred_var, inflated = _build_maps()
    manager = _make_manager({
        "graph_mini_astar_ttl_steps": 100,
        "graph_cache_change_ratio_threshold": 0.0,
        "graph_pred_wall_threshold": 0.7,
        "graph_unknown_as_occ": True,
        "graph_dilate_diam": 3,
    })

    manager.update_graph(np.array([2, 2]), obs, pred_mean, pred_var, inflated)
    manager.update_graph(np.array([2, 2]), obs, pred_mean, pred_var, inflated)

    assert hasattr(manager, "_mini_astar_cache")
    first_keys = set(manager._mini_astar_cache.keys())
    assert first_keys

    manager.update_graph(np.array([2, 2]), obs, pred_mean, pred_var, inflated)
    manager.update_graph(np.array([2, 2]), obs, pred_mean, pred_var, inflated)

    second_keys = set(manager._mini_astar_cache.keys())
    assert second_keys == first_keys


def test_cache_invalidates_on_obs_change():
    obs, pred_mean, pred_var, inflated = _build_maps()
    manager = _make_manager({
        "graph_mini_astar_ttl_steps": 100,
        "graph_cache_change_ratio_threshold": 0.0,
        "graph_pred_wall_threshold": 0.7,
        "graph_unknown_as_occ": True,
        "graph_dilate_diam": 3,
    })

    manager.update_graph(np.array([2, 2]), obs, pred_mean, pred_var, inflated)
    manager.update_graph(np.array([2, 2]), obs, pred_mean, pred_var, inflated)
    old_keys = set(getattr(manager, "_mini_astar_cache", {}).keys())

    obs2 = obs.copy()
    obs2[:, 4] = 1.0
    inflated2 = np.where(obs2 >= 0.8, np.inf, 1.0)

    manager.update_graph(np.array([2, 2]), obs2, pred_mean, pred_var, inflated2)
    manager.update_graph(np.array([2, 2]), obs2, pred_mean, pred_var, inflated2)
    new_keys = set(getattr(manager, "_mini_astar_cache", {}).keys())

    assert old_keys
    assert new_keys
    assert old_keys.isdisjoint(new_keys)
