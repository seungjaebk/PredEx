import numpy as np
from nbh.waypoint_utils import pick_farthest_los_cell, select_waypoint_cell


def test_pick_farthest_los_cell_skips_blocked_cells():
    obs = np.zeros((10, 10), dtype=np.float32)
    obs[4, 4:8] = 1.0  # wall line
    path_cells = [
        type("Cell", (), {"center": np.array([2, 2])})(),
        type("Cell", (), {"center": np.array([3, 3])})(),
        type("Cell", (), {"center": np.array([5, 5])})(),  # blocked by wall
        type("Cell", (), {"center": np.array([7, 7])})(),
    ]
    cur_pose = np.array([2, 2])
    chosen = pick_farthest_los_cell(cur_pose, path_cells, obs)
    assert np.allclose(chosen.center, np.array([3, 3]))


def test_select_waypoint_cell_prefers_farthest_los():
    obs = np.zeros((10, 10), dtype=np.float32)
    path_cells = [
        type("Cell", (), {"center": np.array([2, 2])})(),
        type("Cell", (), {"center": np.array([3, 3])})(),
        type("Cell", (), {"center": np.array([4, 4])})(),
    ]
    cur_pose = np.array([2, 2])
    chosen = select_waypoint_cell(cur_pose, path_cells, obs)
    assert np.allclose(chosen.center, np.array([4, 4]))
