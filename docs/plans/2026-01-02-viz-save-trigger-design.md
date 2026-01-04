# Viz Save Trigger Design

## Context
Saving the `run_viz` and `graph_map` figures every step is slow. The goal is to save these figures only on meaningful events (when a waypoint/target cell is reached) or at a configurable periodic interval.

## Goals
- Save `run_viz` and `graph_map` images on a tunable step interval.
- Save both images when a waypoint is reached.
- Keep behavior configurable from `configs/base.yaml`.

## Non-Goals
- Change the rendering content of either figure.
- Remove existing graph_map save toggle (`save_graph_map`).

## Config
Add two new NBH options in `configs/base.yaml`:
- `nbh.viz_save_every_steps` (int): periodic interval; 0 or negative disables periodic saving.
- `nbh.viz_save_on_waypoint_reached` (bool): enable saving on waypoint reached.

## Behavior
Each step computes `should_save_fig` using the two conditions:
- Periodic: `t % viz_save_every_steps == 0` if `viz_save_every_steps > 0`.
- Event: waypoint reached in the current step and `viz_save_on_waypoint_reached` is true.

If `should_save_fig` is true, visualization rendering is forced for that step and both figures are saved. `save_graph_map` still gates the graph_map image, so only `run_viz` is written when graph_map saving is disabled.

## Implementation Notes
- Add a small helper in `nbh/viz_utils.py` to compute `should_save_fig`.
- Track a `waypoint_reached_this_step` flag in the main loop.
- Gate `plt.savefig` calls in `nbh/explore.py` with `should_save_fig`.

## Testing
- Unit tests for the helper function in `tests/test_viz_utils.py`.
