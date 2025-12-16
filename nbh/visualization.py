"""
Visualization Module

Plotting utilities for exploration visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap


def create_obs_colormap():
    """Create colormap for observed map visualization."""
    white = "#FFFFFF"  # Free space
    blue = "#0000FF"   # Wall (observed)
    gray = "#808080"   # Unknown
    colors = [white, gray, blue]
    obs_cmap = ListedColormap(colors)
    return obs_cmap


def visualize_sensor_range(
    ax, 
    cur_pose, 
    mapper, 
    pd_size,
    show_legend=True
):
    """
    Visualize sensor visibility and occluded regions.
    
    Args:
        ax: Matplotlib axis
        cur_pose: Current robot position (row, col)
        mapper: Mapper instance with lidar_sim_configs
        pd_size: Padding size
        show_legend: Whether to show legend
    """
    try:
        # Get instant observation
        instant_obs = mapper.get_instant_obs_at_pose(cur_pose)
        vis_ind = instant_obs['vis_ind']
        
        # Theoretical Range Mask (Circle) - read from mapper config
        range_pix = mapper.lidar_sim_configs['laser_range_m'] * mapper.lidar_sim_configs['pixel_per_meter']
        
        y_grid, x_grid = np.ogrid[:mapper.obs_map.shape[0], :mapper.obs_map.shape[1]]
        dist_from_pose = np.sqrt((x_grid - cur_pose[1])**2 + (y_grid - cur_pose[0])**2)
        range_mask = dist_from_pose <= range_pix
        
        # Visible Mask
        vis_mask = np.zeros_like(mapper.obs_map, dtype=bool)
        vis_mask[vis_ind[:, 0], vis_ind[:, 1]] = True
        
        # Visible within range
        visible_in_range = vis_mask & range_mask
        
        # Occluded = Range AND NOT Visible
        occluded_mask = range_mask & (~visible_in_range)
        
        # Get indices for plotting
        vis_rows, vis_cols = np.where(visible_in_range)
        occ_rows, occ_cols = np.where(occluded_mask)
        
        # Plot
        ax.scatter(vis_cols - pd_size, vis_rows - pd_size, c='cyan', s=1, alpha=0.05, label='Visible Area')
        ax.scatter(occ_cols - pd_size, occ_rows - pd_size, c='orange', s=1, alpha=0.05, label='Occluded Area')
        
    except Exception as e:
        pass  # Silently handle visualization errors


def visualize_flow_vectors(
    ax, 
    cur_pose, 
    delta_norm, 
    delta_pixels, 
    repulsive_force,
    pd_size,
    viz_scale=10
):
    """
    Visualize flow vectors and combined velocity.
    
    Args:
        ax: Matplotlib axis
        cur_pose: Current robot position
        delta_norm: Normalized flow delta
        delta_pixels: Final movement delta in pixels
        repulsive_force: Repulsive force from obstacles
        pd_size: Padding size
        viz_scale: Scale factor for visualization
    """
    # 1. Plot Raw Flow (Cyan)
    if delta_norm is not None:
        d_raw = delta_norm * 3.0
        ax.arrow(
            cur_pose[1] - pd_size, cur_pose[0] - pd_size,
            d_raw[1] * viz_scale, d_raw[0] * viz_scale,
            color='cyan', head_width=3, label='Flow Vec'
        )
    
    # 2. Plot Repulsion (Yellow)
    if repulsive_force is not None and np.linalg.norm(repulsive_force) > 0.1:
        d_rep = repulsive_force * 3.0
        ax.arrow(
            cur_pose[1] - pd_size, cur_pose[0] - pd_size,
            d_rep[1] * viz_scale, d_rep[0] * viz_scale,
            color='yellow', head_width=3, label='Repulsion'
        )
    
    # 3. Plot Final Result (Red)
    if delta_pixels is not None:
        d_row, d_col = delta_pixels[0] * viz_scale, delta_pixels[1] * viz_scale
        ax.arrow(
            cur_pose[1] - pd_size, cur_pose[0] - pd_size,
            d_col, d_row,
            color='red', head_width=5, label='Combined Vel'
        )


def visualize_target_cell(ax, target_cell, pd_size):
    """
    Highlight target cell on the map.
    
    Args:
        ax: Matplotlib axis
        target_cell: Target CellNode
        pd_size: Padding size
    """
    if target_cell is not None:
        tc_r, tc_c = target_cell.center
        ax.scatter(tc_c - pd_size, tc_r - pd_size, c='blue', s=30, marker='o')


def add_observation_legend(ax):
    """Add legend to observation map axis."""
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Target Cell', markerfacecolor='blue', markersize=8),
        Line2D([0], [0], color='cyan', lw=2, label='Flow Vec'),
        Line2D([0], [0], color='yellow', lw=2, label='Repulsion'),
        Line2D([0], [0], color='red', lw=2, label='Combined Vel'),
        Line2D([0], [0], marker='o', color='w', label='Visible', markerfacecolor='cyan', markersize=5, alpha=0.5),
        Line2D([0], [0], marker='o', color='w', label='Occluded', markerfacecolor='orange', markersize=5, alpha=0.5),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize='small')


def create_figure_layout(mode='nbh'):
    """
    Create figure layout for exploration visualization.
    
    Args:
        mode: Exploration mode
        
    Returns:
        fig: Matplotlib figure
        axes: Dictionary of axes
    """
    if mode == 'nbh':
        fig, ax_array = plt.subplots(3, 2, figsize=(12, 14))
        ax_flatten = ax_array.flatten()
        axes = {
            'gt': ax_flatten[0],
            'obs': ax_flatten[1],
            'var': ax_flatten[2],
            'mean': ax_flatten[3],
            'traj_dist': ax_flatten[4],
            'cell_graph': ax_flatten[5],
        }
    else:
        fig, ax_array = plt.subplots(2, 2, figsize=(10, 10))
        ax_flatten = ax_array.flatten()
        axes = {
            'gt': ax_flatten[0],
            'obs': ax_flatten[1],
            'var': ax_flatten[2],
            'mean': ax_flatten[3],
        }
    
    return fig, axes


def save_visualization(fig, save_path, step_idx, exp_title):
    """
    Save visualization figure to file.
    
    Args:
        fig: Matplotlib figure
        save_path: Base save path
        step_idx: Current step index
        exp_title: Experiment title
    """
    import os
    filename = f"{exp_title}_{step_idx:08d}.png"
    filepath = os.path.join(save_path, filename)
    fig.savefig(filepath, dpi=100, bbox_inches='tight')
    print(f"saving fig: {step_idx}")
