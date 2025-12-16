"""
Exploration Configuration Module

Constants and configuration loading for the graph-flow exploration system.
"""

import os

# ============================================================================
# NORMALIZATION CONSTANTS (must match training)
# ============================================================================
DELTA_SCALE = 10.0  # Scale factor for flow model output

# ============================================================================
# GRAPH/CELL CONSTANTS
# ============================================================================
CELL_SIZE = 25  # Pixels per cell (2.5m in real world at 10px/m)

# ============================================================================
# WAYPOINT NAVIGATION CONSTANTS
# ============================================================================
WAYPOINT_REACHED_TOLERANCE = 1.0  # Pixels - distance to consider waypoint reached
WAYPOINT_STALE_STEPS = 30  # Max steps per cell before marking stale
MAX_TARGET_DISTANCE = 60.0  # Pixels (~6m) - max distance for target selection

# ============================================================================
# GHOST CELL CONSTANTS
# ============================================================================
MAX_GHOST_DISTANCE = 3  # Max cells away from real cells
MAX_VAR_THRESHOLD = 0.1  # Minimum variance to create ghost cell

# ============================================================================
# FLOW MODEL CONSTANTS
# ============================================================================
FLOW_CROP_RADIUS = 64  # Half of 128x128 patch
FLOW_NUM_STEPS = 10  # ODE integration steps
TRAJECTORY_NUM_SAMPLES = 50  # Number of trajectory samples
GOAL_PERTURB_STD = 0.5  # Radians (~30°) for trajectory diversity


def get_options_dict_from_yml(config_name):
    """Load configuration from YAML file using Hydra."""
    from omegaconf import OmegaConf
    import hydra
    
    # Get NBH root directory (parent of nbh/ module)
    nbh_module_dir = os.path.dirname(os.path.abspath(__file__))
    nbh_root = os.path.dirname(nbh_module_dir)
    hydra_config_dir_path = os.path.join(nbh_root, 'configs')
    
    print(f"Loading config from: {hydra_config_dir_path}")
    with hydra.initialize_config_dir(config_dir=hydra_config_dir_path, version_base=None):
        cfg = hydra.compose(config_name=config_name)
    options_dict = OmegaConf.to_container(cfg)
    options = OmegaConf.create(options_dict)
    default_flow_logging = OmegaConf.create({
        "enabled": False,
        "save_every": 1,
        "crop_radius": 128,
        "include_prediction": True,
        "output_dir": "flow_datasets",
    })
    if "flow_logging" in options:
        options.flow_logging = OmegaConf.merge(default_flow_logging, options.flow_logging)
    else:
        options.flow_logging = default_flow_logging
    return options


def get_flow_config(flow_runtime=None):
    """Get flow planner configuration from runtime or defaults."""
    config = {
        'model': None,
        'device': None,
        'num_steps': FLOW_NUM_STEPS,
        'crop_radius': FLOW_CROP_RADIUS,
        'delta_scale': DELTA_SCALE,
        'num_samples': TRAJECTORY_NUM_SAMPLES,
        'goal_perturb_std': GOAL_PERTURB_STD,
    }
    
    if flow_runtime is not None:
        config['model'] = flow_runtime.get('model')
        config['device'] = flow_runtime.get('device')
        config['num_steps'] = flow_runtime.get('num_steps', FLOW_NUM_STEPS)
        config['crop_radius'] = flow_runtime.get('crop_radius', FLOW_CROP_RADIUS)
    
    return config
