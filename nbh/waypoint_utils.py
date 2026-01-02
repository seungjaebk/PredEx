import numpy as np


def check_line_of_sight(start: np.ndarray, end: np.ndarray, obs_map: np.ndarray) -> bool:
    """Check if there's a clear path from start to end (no walls)."""
    r0, c0 = int(start[0]), int(start[1])
    r1, c1 = int(end[0]), int(end[1])

    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    r, c = r0, c0
    while True:
        if 0 <= r < obs_map.shape[0] and 0 <= c < obs_map.shape[1]:
            if obs_map[r, c] == 1.0:
                return False
        else:
            return False

        if r == r1 and c == c1:
            break

        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc

    return True


def pick_farthest_los_cell(cur_pose: np.ndarray, path_cells, obs_map: np.ndarray):
    """Return the farthest path cell with line-of-sight from current pose."""
    last_visible = None
    for cell in path_cells:
        if check_line_of_sight(cur_pose, cell.center, obs_map):
            last_visible = cell
        else:
            break
    return last_visible


def select_waypoint_cell(cur_pose: np.ndarray, path_cells, obs_map: np.ndarray):
    """Choose the farthest LOS cell, falling back to the next path cell."""
    if not path_cells:
        return None

    los_cell = pick_farthest_los_cell(cur_pose, path_cells, obs_map)
    if los_cell is not None and los_cell is not path_cells[0]:
        return los_cell

    if len(path_cells) > 1:
        return path_cells[1]

    return None
