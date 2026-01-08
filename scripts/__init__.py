"""
NBH (Next Best Hallucination) Package

Hierarchical robot exploration using ghost cells and flow matching.

Modules:
- exploration_config: Constants and configuration loading
- graph_utils: Cell graph + Ghost cells (CellNode, CellManager)
- high_level_planner: Waypoint management and target selection
- flow_planner: Flow matching model inference
- flow_viz_utils: Trajectory sampling and visualization
- frontier_utils: Frontier detection helpers
- lama_utils: LAMA prediction utilities
- visualization: Plotting functions
- explore: Main entry point

Usage:
    from nbh.exploration_config import DELTA_SCALE, CELL_SIZE
    from nbh.graph_utils import CellManager
    from nbh.flow_planner import load_flow_model
"""

# Only expose config constants at package level (minimal dependencies)
from .exploration_config import (
    DELTA_SCALE,
    CELL_SIZE,
    WAYPOINT_REACHED_TOLERANCE,
    WAYPOINT_STALE_STEPS,
    MAX_TARGET_DISTANCE,
    MAX_GHOST_DISTANCE,
    MAX_VAR_THRESHOLD,
    FLOW_CROP_RADIUS,
    FLOW_NUM_STEPS,
    TRAJECTORY_NUM_SAMPLES,
    GOAL_PERTURB_STD,
)
