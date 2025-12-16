"""
Frontier Utils Module

Frontier detection and exploration functions for flow-based exploration.
"""

import numpy as np


def update_mission_status(start_time, cur_step, mission_complete, fail_reason, mission_status_save_path):
    """Update mission status to file."""
    import time
    import json
    mission_status = {}
    mission_status['start_time'] = start_time
    mission_status["cur_step"] = cur_step
    mission_status["mission_complete"] = mission_complete
    mission_status["fail_reason"] = fail_reason
    mission_status["last_exp_time_s"] = time.time() - mission_status['start_time']
    with open(mission_status_save_path, 'w') as f:
        json.dump(mission_status, f)


def is_locked_frontier_center_valid(locked_frontier_center, occ_grid_pyastar, cur_pose, collect_opts, pixel_per_meter):
    """Check if a locked frontier center is still valid for navigation."""
    if locked_frontier_center is None:
        return False
    if occ_grid_pyastar[locked_frontier_center[0], locked_frontier_center[1]] == np.inf:
        return False
    if np.linalg.norm(locked_frontier_center - cur_pose) < collect_opts.cur_pose_dist_threshold_m * pixel_per_meter:
        return False
    return True


def reselect_frontier_from_frontier_region_centers(frontier_region_centers, total_cost, t, start_exp_time, mission_status_save_path):
    """Select a new frontier from available frontier regions."""
    frontier_selected = False
    
    if len(frontier_region_centers) == 0:
        update_mission_status(start_time=start_exp_time, cur_step=t, mission_complete=False, 
                            fail_reason="frontier_region_centers", mission_status_save_path=mission_status_save_path)
        return frontier_selected, None, None, None
    
    min_cost_idx = np.argmin(total_cost)
    locked_frontier_center = frontier_region_centers[min_cost_idx]
    
    frontier_region_centers = np.delete(frontier_region_centers, min_cost_idx, axis=0)
    total_cost = np.delete(total_cost, min_cost_idx, axis=0)
    
    if len(frontier_region_centers) == 0:
        update_mission_status(start_time=start_exp_time, cur_step=t, mission_complete=False, 
                            fail_reason="frontier_region_centers", mission_status_save_path=mission_status_save_path)
        return frontier_selected, None, None, None
    
    frontier_selected = True
    return frontier_selected, locked_frontier_center, frontier_region_centers, total_cost


def determine_local_planner(mode):
    """Determine which local planner to use. Only nbh mode supported."""
    if mode == 'nbh':
        return 'nbh'
    else:
        raise ValueError(f"Unsupported mode: {mode}. Only 'nbh' mode is supported.")


# Legacy functions removed:
# - get_hector_exploration_transform_map (hector baseline)
# - gradient_planner (hector baseline)
