"""
NBH (Next Best Hallucination) Package

Hierarchical robot exploration using ghost cells and flow matching.

Modules:
- config_utils: Configuration loading utilities
- graph_utils: Cell graph + Ghost cells (CellNode, CellManager)
- high_level_planner: Waypoint management and target selection
- flow_planner: Flow matching model inference
- flow_viz_utils: Trajectory sampling and visualization
- frontier_utils: Frontier detection helpers
- lama_utils: LAMA prediction utilities
- visualization: Plotting functions
- explore: Main entry point

Usage:
    from scripts.config_utils import get_options_dict_from_yml, build_promotion_cfg
    from scripts.graph_utils import CellManager
    from scripts.flow_planner import load_flow_model
"""

# Expose config utilities at package level
from .config_utils import (
    get_options_dict_from_yml,
    build_promotion_cfg,
    get_graph_update_mode,
    should_run_full_update,
    should_run_light_update,
    get_flow_config,
)
