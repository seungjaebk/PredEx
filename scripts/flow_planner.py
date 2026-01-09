"""
Flow Planner Module

Local trajectory planning using Conditional Flow Matching model.
Handles model loading, trajectory sampling, and path selection.
"""

import numpy as np
import torch
import os
from contextlib import contextmanager

# Default values (can be overridden via YAML config)
DELTA_SCALE = 10.0
FLOW_NUM_STEPS = 10
TRAJECTORY_NUM_SAMPLES = 50
GOAL_PERTURB_STD = 0.5


@contextmanager
def dummy_context_manager():
    """Dummy context manager for optional torch.cuda.amp.autocast."""
    yield


def load_flow_model(checkpoint_path, device):
    """Load trained flow matching model from checkpoint."""
    from training.train_flow_matching import FlowMatchingModel
    
    assert os.path.exists(checkpoint_path), f"Flow checkpoint not found: {checkpoint_path}"
    
    model = FlowMatchingModel().to(device)
    # weights_only=False for PyTorch 2.6+ compatibility
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle both direct state_dict and wrapped checkpoint formats
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    
    model.eval()
    print(f"Loaded flow model from {checkpoint_path} on {device}")
    return model


def build_flow_input_tensor(obs_patch, mean_patch, var_patch, goal_vector):
    """Build input tensor for flow model from observation patches."""
    if obs_patch is None:
        return None, None
    obs = obs_patch.astype(np.float32)
    h, w = obs.shape
    if mean_patch is None:
        mean = np.zeros((h, w), dtype=np.float32)
        mask_val = 0.0
    else:
        mean = mean_patch.astype(np.float32)
        mask_val = 1.0
    if var_patch is None:
        var = np.zeros((h, w), dtype=np.float32)
    else:
        var = var_patch.astype(np.float32)
        mask_val = 1.0
    mask = np.full((h, w), mask_val, dtype=np.float32)
    stacked = np.stack([obs, mean, var, mask], axis=0)
    tensor = torch.from_numpy(stacked).unsqueeze(0) / 255.0
    
    if goal_vector is None:
        goal_vector = np.zeros(2, dtype=np.float32)
    goal_tensor = torch.from_numpy(goal_vector).unsqueeze(0).float()
    
    return tensor, goal_tensor


def sample_flow_delta(flow_model, flow_input_tensor, flow_goal_tensor, flow_device, num_steps=1):
    """Sample a single flow delta from the model using Euler integration."""
    if flow_input_tensor is None:
        return None
    flow_input_tensor = flow_input_tensor.to(flow_device)
    flow_goal_tensor = flow_goal_tensor.to(flow_device)
    
    # Start from Gaussian noise (z0)
    z = torch.randn((1, 2), device=flow_device)
    
    # Euler Integration from t=0 to t=1
    dt = 1.0 / num_steps
    with torch.no_grad():
        for i in range(num_steps):
            t_scalar = i / num_steps
            t = torch.full((1, 1), t_scalar, device=flow_device)
            vel = flow_model(flow_input_tensor, z, t, flow_goal_tensor)
            z = z + vel * dt
            
    # Un-normalize: scale back to pixel units
    delta_pixels = z.squeeze().cpu().numpy() * DELTA_SCALE
    return delta_pixels


def extract_local_patch(grid, center, radius, pad_value=0.0):
    """
    Return a square patch of size (2*radius+1, 2*radius+1) centered at center (row, col).
    Pads out-of-bound areas with pad_value.
    """
    if grid is None:
        return None
    arr = grid.detach().cpu().numpy() if torch.is_tensor(grid) else np.asarray(grid)
    radius = int(radius)
    if radius <= 0:
        return arr.copy()
    pad_width = ((radius, radius), (radius, radius))
    arr_padded = np.pad(arr, pad_width, mode='constant', constant_values=pad_value)
    r = int(center[0]) + radius
    c = int(center[1]) + radius
    return arr_padded[r - radius:r + radius + 1, c - radius:c + radius + 1]


def sample_trajectory_for_cell(
    flow_model, 
    obs_patch, 
    mean_patch, 
    var_patch, 
    goal_vector,
    obs_map,
    cur_pose,
    flow_device,
    num_samples=None,
    num_steps=None,
    goal_perturb_std=None
):
    """Sample multiple trajectories for a cell transition and select the best one."""
    from .flow_viz_utils import sample_multiple_trajectories, select_best_trajectory
    
    if num_samples is None:
        num_samples = TRAJECTORY_NUM_SAMPLES
    if num_steps is None:
        num_steps = FLOW_NUM_STEPS
    if goal_perturb_std is None:
        goal_perturb_std = GOAL_PERTURB_STD
    
    flow_tensor, goal_tensor = build_flow_input_tensor(
        obs_patch, mean_patch, var_patch, goal_vector
    )
    
    if flow_tensor is None:
        return goal_vector * DELTA_SCALE, [], 0, 0
    
    all_trajectories = sample_multiple_trajectories(
        flow_model, flow_tensor, goal_tensor, flow_device,
        num_samples=num_samples,
        num_steps=num_steps,
        delta_scale=DELTA_SCALE,
        goal_perturb_std=goal_perturb_std
    )
    
    best_trajectory, best_idx, collision_free_count = select_best_trajectory(
        all_trajectories, cur_pose, obs_map, goal_vector
    )
    
    return best_trajectory, all_trajectories, best_idx, collision_free_count
