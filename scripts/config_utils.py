"""
Configuration Utilities Module

Helper functions for loading and processing configuration from YAML files.
"""

import os


def _get_cfg_value(cfg, key, default):
    """Get value from config object (dict or OmegaConf)."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    getter = getattr(cfg, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(cfg, key, default)


def build_promotion_cfg(nbh_cfg):
    """Build promotion config dict from NBH config section."""
    return {
        "graph_grid_policy": _get_cfg_value(nbh_cfg, "graph_grid_policy", "full_map"),
        "graph_max_ghost_distance": _get_cfg_value(nbh_cfg, "graph_max_ghost_distance", 2),
        "graph_obs_blocked_ratio": _get_cfg_value(nbh_cfg, "graph_obs_blocked_ratio", 0.3),
        "graph_unknown_ratio_threshold": _get_cfg_value(nbh_cfg, "graph_unknown_ratio_threshold", 0.5),
        "graph_centroid_blocked_threshold": _get_cfg_value(nbh_cfg, "graph_centroid_blocked_threshold", 0.8),
        "graph_ghost_pred_mean_free_threshold": _get_cfg_value(nbh_cfg, "graph_ghost_pred_mean_free_threshold", 0.4),
        "graph_ghost_pred_var_max_threshold": _get_cfg_value(nbh_cfg, "graph_ghost_pred_var_max_threshold", 0.3),
        "graph_diffuse_gamma": _get_cfg_value(nbh_cfg, "graph_diffuse_gamma", 0.95),
        "graph_diffuse_iterations": _get_cfg_value(nbh_cfg, "graph_diffuse_iterations", 50),
        "graph_diffuse_on_update": _get_cfg_value(nbh_cfg, "graph_diffuse_on_update", False),
        "graph_target_risk_lambda": _get_cfg_value(nbh_cfg, "graph_target_risk_lambda", 0.5),
    }


# Graph update mode constants
GRAPH_UPDATE_MODE_DEFAULT = "target_change"
GRAPH_UPDATE_MODES = ("full", "target_change", "light_only")


def get_graph_update_mode(nbh_cfg):
    """Get graph update mode from NBH config."""
    mode = _get_cfg_value(nbh_cfg, "graph_update_mode", GRAPH_UPDATE_MODE_DEFAULT)
    if mode is None:
        mode = GRAPH_UPDATE_MODE_DEFAULT
    if mode not in GRAPH_UPDATE_MODES:
        raise ValueError(f"Invalid graph_update_mode: {mode}")
    return mode


def should_run_full_update(mode, has_graph, need_new_target):
    """Determine if full graph update should run based on mode and state."""
    if mode == "full":
        return True
    if mode == "target_change":
        return bool(need_new_target)
    if mode == "light_only":
        return not bool(has_graph)
    raise ValueError(f"Invalid graph_update_mode: {mode}")


def should_run_light_update(mode):
    """Determine if light graph update should run based on mode."""
    if mode not in GRAPH_UPDATE_MODES:
        raise ValueError(f"Invalid graph_update_mode: {mode}")
    return mode in ("target_change", "light_only")


def get_options_dict_from_yml(config_name):
    """Load configuration from YAML file using Hydra."""
    from omegaconf import OmegaConf
    import hydra
    
    # Get project root directory (parent of scripts/ module)
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(scripts_dir)
    hydra_config_dir_path = os.path.join(project_root, 'configs')
    
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


def get_flow_config(options=None):
    """
    Get flow planner configuration from options.
    
    Reads values from YAML config (options.flow section) with defaults.
    """
    # Defaults (fallback if not in YAML)
    defaults = {
        'delta_scale': 10.0,
        'num_steps': 10,
        'crop_radius': 64,
        'num_samples': 50,
        'goal_perturb_std': 0.5,
    }
    
    config = {
        'model': None,
        'device': None,
    }
    
    # Get flow section from options
    flow_cfg = None
    if options is not None:
        flow_cfg = _get_cfg_value(options, 'flow', None)
    
    for key, default_val in defaults.items():
        config[key] = _get_cfg_value(flow_cfg, key, default_val)
    
    return config


# ============================================================================
# COORDINATE CONVERSION UTILITIES
# ============================================================================
# Maps are processed with: 2x reduction (block_reduce) then 500px padding
# Original coords -> Processed coords: [row/2 + 500, col/2 + 500]
# Processed coords -> Original coords: [(row - 500) * 2, (col - 500) * 2]

MAP_BLOCK_SIZE = 2
MAP_PADDING = 500


def original_to_processed_coords(row, col):
    """Convert original map coordinates to processed (2x reduced + padded) coordinates."""
    return (row // MAP_BLOCK_SIZE + MAP_PADDING, col // MAP_BLOCK_SIZE + MAP_PADDING)


def processed_to_original_coords(row, col):
    """Convert processed coordinates back to original map coordinates."""
    return ((row - MAP_PADDING) * MAP_BLOCK_SIZE, (col - MAP_PADDING) * MAP_BLOCK_SIZE)
