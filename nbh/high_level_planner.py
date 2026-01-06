"""
High-Level Planner Module

Graph-based planning and waypoint management.
Handles cell navigation, path following, and stale detection.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Any

from .exploration_config import (
    CELL_SIZE, 
    WAYPOINT_REACHED_TOLERANCE, 
    WAYPOINT_STALE_STEPS
)


@dataclass
class WaypointState:
    """State for waypoint navigation."""
    current_waypoint: Optional[np.ndarray] = None
    locked_target_cell: Any = None  # CellNode
    locked_path_to_target: Optional[List] = None
    locked_trajectory: Optional[np.ndarray] = None
    locked_trajectory_samples: Optional[List] = None
    steps_counter: int = 0
    
    def clear(self):
        """Clear all waypoint state."""
        self.current_waypoint = None
        self.locked_target_cell = None
        self.locked_path_to_target = None
        self.locked_trajectory = None
        self.locked_trajectory_samples = None
        self.steps_counter = 0
    
    def is_active(self) -> bool:
        """Check if there's an active waypoint."""
        return self.current_waypoint is not None


def check_waypoint_status(
    cur_pose: np.ndarray,
    waypoint_state: WaypointState,
    cell_manager,
    tolerance: float = WAYPOINT_REACHED_TOLERANCE,
    max_steps: int = WAYPOINT_STALE_STEPS
) -> Tuple[str, Optional[np.ndarray]]:
    """
    Check if current waypoint is reached or stale.
    
    Args:
        cur_pose: Current robot position (row, col)
        waypoint_state: Current waypoint state
        cell_manager: CellManager instance
        tolerance: Distance to consider waypoint reached
        max_steps: Steps before marking as stale
        
    Returns:
        status: 'REACHED', 'STALE', or 'ACTIVE'
        target_pos: Position to navigate to (if ACTIVE)
    """
    if not waypoint_state.is_active():
        return 'NO_WAYPOINT', None
    
    # Calculate graph-based distance (number of remaining cells × cell_size)
    # This is the TRUE distance through the graph, NOT Euclidean!
    remaining_cells = 0
    if waypoint_state.locked_path_to_target is not None and len(waypoint_state.locked_path_to_target) > 0:
        remaining_cells = len(waypoint_state.locked_path_to_target)
        if cell_manager.current_cell is not None:
            for i, cell in enumerate(waypoint_state.locked_path_to_target):
                if cell.index == cell_manager.current_cell.index:
                    remaining_cells = len(waypoint_state.locked_path_to_target) - i - 1
                    break
        graph_dist = remaining_cells * CELL_SIZE
    else:
        # Fallback to Euclidean only if no path info
        graph_dist = np.linalg.norm(cur_pose - waypoint_state.current_waypoint)
    
    # Use GRAPH DISTANCE for waypoint reached check, not Euclidean!
    # Waypoint is "reached" when we're in the same cell or adjacent cell
    # (remaining_cells <= 1 means we're at or next to target)
    dist_to_wp = graph_dist  # USE GRAPH DISTANCE!
    
    # Also check Euclidean for final cell precision
    euclidean_dist = np.linalg.norm(cur_pose - waypoint_state.current_waypoint)
    
    # Reached if: (in target cell) OR (very close Euclidean within same cell)
    in_target_cell = remaining_cells == 0
    close_enough = euclidean_dist <= tolerance
    
    if in_target_cell or close_enough:
        # Waypoint reached!
        print(f"Waypoint Reached! (graph_dist={graph_dist:.1f}px, euclidean={euclidean_dist:.1f}px, remaining={remaining_cells} cells)")
        return 'REACHED', None
    
    # Increment step counter
    waypoint_state.steps_counter += 1
    
    if waypoint_state.steps_counter >= max_steps:
        # Stale - stuck for too long
        print(f"Cell STALE! Stuck for {waypoint_state.steps_counter} steps (graph_dist={graph_dist:.1f}px). Moving to next target.")
        if waypoint_state.locked_target_cell is not None:
            waypoint_state.locked_target_cell.visit_count += 5  # Penalize
        return 'STALE', None
    
    # Still active
    return 'ACTIVE', waypoint_state.current_waypoint


def plan_next_waypoint(
    cell_manager,
    path_to_target: List,
    obs_map: np.ndarray
) -> Tuple[Any, str]:
    """
    Get the next cell to navigate to.
    
    Tries risk path first, falls back to greedy neighbor selection.
    
    Args:
        cell_manager: CellManager instance
        path_to_target: Dijkstra-computed risk path to target
        obs_map: Observation map for collision checking
        
    Returns:
        next_cell: CellNode to navigate to
        decision_type: Description of how cell was chosen
    """
    # First try: Follow risk path
    next_cell, decision_type = cell_manager.get_next_path_cell(path_to_target)
    
    # Fallback: Greedy neighbor selection
    if next_cell is None:
        target_cell = path_to_target[-1] if path_to_target else None
        next_cell, decision_type = cell_manager.pick_best_neighbor(
            target_cell, obs_map
        )
    
    return next_cell, decision_type


def lock_new_waypoint(
    waypoint_state: WaypointState,
    next_cell,
    target_cell,
    path_to_target: List,
    cur_pose: np.ndarray,
    trajectory: np.ndarray = None,
    trajectory_samples: List = None
):
    """
    Lock a new waypoint for navigation.
    
    Args:
        waypoint_state: WaypointState to update
        next_cell: CellNode for immediate navigation
        target_cell: High-level target cell
        path_to_target: Full risk path
        cur_pose: Current robot position
        trajectory: Selected flow trajectory
        trajectory_samples: All sampled trajectories (for visualization)
    """
    target_pos = next_cell.center
    
    waypoint_state.current_waypoint = target_pos
    waypoint_state.locked_target_cell = target_cell
    waypoint_state.locked_path_to_target = path_to_target
    waypoint_state.locked_trajectory = trajectory
    waypoint_state.locked_trajectory_samples = trajectory_samples
    waypoint_state.steps_counter = 0
    
    # Print info
    path_cells = len(path_to_target) if path_to_target else 1
    graph_distance = path_cells * CELL_SIZE
    print(f"New Waypoint Locked: Cell {next_cell.index}, path={path_cells} cells, dist={graph_distance:.1f}px")


def compute_goal_vector(cur_pose: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """
    Compute normalized goal direction vector.
    
    Args:
        cur_pose: Current position (row, col)
        target_pos: Target position (row, col)
        
    Returns:
        goal_vec: (2,) normalized direction vector [row_diff, col_diff]
    """
    diff = target_pos - cur_pose
    dist = np.linalg.norm(diff)
    
    if dist > 1e-3:
        return (diff / dist).astype(np.float32)
    else:
        return np.zeros(2, dtype=np.float32)
