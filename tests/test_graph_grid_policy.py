import numpy as np

from nbh.graph_utils import CellManager


def test_full_map_grid_creates_all_real_cells():
    obs_map = np.zeros((20, 20), dtype=np.float32)
    pred_mean = np.zeros_like(obs_map)
    pred_var = np.zeros_like(obs_map)

    cm = CellManager(
        cell_size=10,
        promotion_cfg={"graph_grid_policy": "full_map"},
    )
    cm.update_graph(
        np.array([0, 0]),
        obs_map,
        pred_mean,
        pred_var_map=pred_var,
    )

    assert len(cm.cells) == 4
    assert all((not c.is_ghost) and (not c.is_blocked) for c in cm.cells.values())


def test_frontier_expand_clamps_indices_to_bounds():
    obs_map = np.zeros((20, 20), dtype=np.float32)
    pred_mean = np.zeros_like(obs_map)
    pred_var = np.zeros_like(obs_map)

    cm = CellManager(
        cell_size=10,
        promotion_cfg={
            "graph_grid_policy": "frontier_expand",
            "graph_max_ghost_distance": 1,
        },
    )
    cm.update_graph(
        np.array([0, 0]),
        obs_map,
        pred_mean,
        pred_var_map=pred_var,
    )

    rows = int(np.ceil(obs_map.shape[0] / cm.cell_size))
    cols = int(np.ceil(obs_map.shape[1] / cm.cell_size))
    assert all(0 <= r < rows and 0 <= c < cols for r, c in cm.cells.keys())


def test_full_map_allows_ghosts_without_distance_gating():
    obs_map = np.full((20, 20), 0.5, dtype=np.float32)
    pred_mean = np.zeros_like(obs_map)
    pred_var = np.zeros_like(obs_map)

    cm = CellManager(
        cell_size=10,
        promotion_cfg={"graph_grid_policy": "full_map"},
    )
    cm.update_graph(
        np.array([0, 0]),
        obs_map,
        pred_mean,
        pred_var_map=pred_var,
    )

    assert len(cm.cells) == 4
    assert all(c.is_ghost and (not c.is_blocked) for c in cm.cells.values())


def test_frontier_expand_clamps_indices_without_skips():
    obs_map = np.zeros((20, 20), dtype=np.float32)
    pred_mean = np.zeros_like(obs_map)
    pred_var = np.zeros_like(obs_map)

    cm = CellManager(
        cell_size=10,
        promotion_cfg={
            "graph_grid_policy": "frontier_expand",
            "graph_max_ghost_distance": 1,
        },
    )
    cm.update_graph(
        np.array([0, 0]),
        obs_map,
        pred_mean,
        pred_var_map=pred_var,
    )

    assert cm.last_graph_stats["skipped_outside"] == 0
