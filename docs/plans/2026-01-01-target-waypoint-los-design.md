# Target Lock + LOS Waypointing Design

## Context
The current flow navigation locks a waypoint at every adjacent cell center along the BFS path. This forces the robot to visit each intermediate cell centroid even when a longer line-of-sight (LOS) hop is possible. Also, cell `propagated_value` only updates when `find_exploration_target()` is called, which is gated by waypoint clearing. This makes the propagated values stale across timesteps.

## Goals
- Keep target locking behavior: once a target cell is selected, it stays locked unless it becomes invalid (blocked, reached, unreachable).
- Update `propagated_value` every timestep so target selection reflects the latest prediction state.
- When a target is locked, choose a waypoint that avoids unnecessary intermediate centroids by jumping to the farthest visible cell on the BFS path.

## Non-Goals
- Changing target selection heuristics beyond stale value refresh.
- Replacing BFS path planning or flow planner behavior.
- Introducing new map representations or large refactors.

## Proposed Changes
### 1) Refresh propagated values each timestep
- Call `cell_manager.diffuse_scent(unpadded_pred_var)` after `cell_manager.update_graph(...)` in the flow navigation loop.
- This updates `propagated_value` without affecting `locked_target_cell` until the next target selection event.

### 2) Farthest-LOS waypoint selection
- When `current_waypoint` is `None` and `path_to_target` exists, choose the **farthest cell on the path** that has LOS from the current pose.
- Reuse the existing `check_line_of_sight` (Bresenham) helper in `nbh/explore.py`.
- Iterate path cells in order and keep the last LOS-valid cell; lock that cell center as the new waypoint.

### 3) Fallbacks
- If no path exists, preserve existing unreachable handling (target reselection).
- If no LOS cell is found, fall back to the immediate next path cell (`get_next_path_cell`) or greedy neighbor selection (`pick_best_neighbor`) as today.

## Data Flow
1. Update graph structure and current cell.
2. Diffuse scent each timestep using the latest prediction variance.
3. If waypoint active, keep navigating; otherwise:
   - Validate target (blocked/reached/unreachable) and reselect only if needed.
   - Compute path to target.
   - Select farthest LOS waypoint on the path.
   - Lock waypoint and trajectory for local navigation.

## Error Handling
- LOS returns `False` for any out-of-bounds or wall-hit point.
- If path is stale and does not include the current cell, fall back to greedy neighbor selection.

## Testing
- Unit test: farthest-LOS selection on a toy grid with walls, ensuring it selects the last visible path cell.
- Regression test: `diffuse_scent()` updates `propagated_value` each step without changing `locked_target_cell` unless it becomes invalid.

## Rollout
- Land changes behind existing behavior; no config changes required.
- Validate in simulation with visualization overlays for LOS-chosen waypoints.
