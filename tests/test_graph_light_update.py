import numpy as np

from nbh.graph_utils import CellManager


def _make_manager():
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
        connectivity_cfg={},
        debug_cfg={},
    )


def test_update_graph_light_tracks_current_cell():
    manager = _make_manager()

    manager.update_graph_light(np.array([2, 2]))
    assert manager.current_cell is not None
    assert manager.current_cell.index == (0, 0)
    assert manager.cells[(0, 0)].visit_count == 1

    manager.update_graph_light(np.array([2, 2]))
    assert manager.current_cell.index == (0, 0)
    assert manager.cells[(0, 0)].visit_count == 2

    manager.update_graph_light(np.array([6, 2]))
    assert manager.current_cell.index == (1, 0)
    assert manager.cells[(1, 0)].visit_count == 1
    assert manager.current_cell.parent.index == (0, 0)
    assert len(manager.visited_stack) == 1


def test_update_graph_light_backtracks_stack():
    manager = _make_manager()

    manager.update_graph_light(np.array([2, 2]))
    manager.update_graph_light(np.array([6, 2]))
    assert len(manager.visited_stack) == 1

    manager.update_graph_light(np.array([2, 2]))
    assert manager.current_cell.index == (0, 0)
    assert len(manager.visited_stack) == 0
