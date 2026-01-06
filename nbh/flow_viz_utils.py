"""
Visualization utilities for Flow Matching exploration.

1. Trajectory Distribution: Sample multiple trajectories and visualize
2. Cell Graph: Visualize Real/Ghost nodes and their connectivity
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import torch


# ============================================================================
# 1. TRAJECTORY DISTRIBUTION VISUALIZATION
# ============================================================================

def normalize_edge_risks(costs):
    costs = np.asarray(costs, dtype=float)
    if costs.size == 0:
        return costs
    min_val = float(np.min(costs))
    max_val = float(np.max(costs))
    if max_val <= min_val:
        return np.zeros_like(costs, dtype=float)
    return np.clip((costs - min_val) / (max_val - min_val), 0.0, 1.0)

def get_graph_path_label():
    return "Risk Path"


def sample_multiple_trajectories(flow_model, flow_input_tensor, flow_goal_tensor, 
                                  flow_device, num_samples=100, num_steps=10,
                                  trajectory_len=20, delta_scale=20.0,
                                  goal_perturb_std=0.8):  # Increased from 0.3 to 0.8 (≈45°)
    """
    Sample multiple trajectory predictions from the flow model.
    
    GoalFlow-style: Perturb goal direction to create diversity.
    CFM is deterministic for fixed conditioning, so we create diversity by
    sampling different goal directions around the target.
    
    Args:
        flow_model: Trained flow matching model
        flow_input_tensor: Observation conditioning
        flow_goal_tensor: Goal direction conditioning  
        flow_device: Device for computation
        num_samples: Number of trajectory samples
        num_steps: ODE integration steps
        trajectory_len: Length of trajectory (for future multi-step models)
        delta_scale: Scale factor for unnormalization
        goal_perturb_std: Standard deviation for goal angle perturbation (radians)
        
    Returns:
        (num_samples, 2) array of delta predictions (endpoints)
    """
    if flow_input_tensor is None:
        return None
    
    flow_input_tensor = flow_input_tensor.to(flow_device)
    flow_goal_tensor = flow_goal_tensor.to(flow_device)
    
    # GoalFlow approach: Create diversity by perturbing goal direction
    batch_size = num_samples
    
    # Expand observation conditioning
    flow_input_batch = flow_input_tensor.expand(batch_size, -1, -1, -1)  # [N, C, H, W]
    
    # Perturb goal directions with angular noise
    # Original goal as unit vector
    goal_np = flow_goal_tensor.cpu().numpy().flatten()
    goal_angle = np.arctan2(goal_np[0], goal_np[1])  # row, col -> angle
    goal_mag = np.linalg.norm(goal_np)
    
    # Generate perturbed angles
    angle_perturbations = np.random.normal(0, goal_perturb_std, size=batch_size)
    perturbed_angles = goal_angle + angle_perturbations
    
    # Convert back to goal vectors
    perturbed_goals = np.stack([
        np.sin(perturbed_angles) * goal_mag,  # row direction
        np.cos(perturbed_angles) * goal_mag,  # col direction
    ], axis=1)  # [N, 2]
    
    flow_goal_batch = torch.from_numpy(perturbed_goals).float().to(flow_device)
    
    # Start from noise (same for all - diversity comes from goal)
    z = torch.randn((batch_size, 2), device=flow_device)
    
    with torch.no_grad():
        # Euler Integration from t=0 to t=1
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t_scalar = i / num_steps
            t = torch.full((batch_size, 1), t_scalar, device=flow_device)
            
            # Predict velocity field for ALL samples with different goals
            vel = flow_model(flow_input_batch, z, t, flow_goal_batch)
            
            # Step: z_{t+1} = z_t + v_t * dt
            z = z + vel * dt
        
        # Un-normalize all samples
        delta_pixels = z.cpu().numpy() * delta_scale  # [N, 2]
    
    return delta_pixels


def is_path_collision_free(start, end, obs_map, num_checks=10, wall_threshold=0.8):
    """
    Check if path from start to end is collision-free using line interpolation.
    
    Args:
        start: (row, col) start position in pixels
        end: (row, col) end position in pixels
        obs_map: Observation map (0=free, 0.5=unknown, 1=occupied)
        num_checks: Number of points to check along the line
        wall_threshold: Value above which is considered a wall
        
    Returns:
        True if path is collision-free, False otherwise
    """
    H, W = obs_map.shape
    
    for i in range(num_checks + 1):
        t = i / num_checks
        r = int(start[0] * (1 - t) + end[0] * t)
        c = int(start[1] * (1 - t) + end[1] * t)
        
        # Bounds check
        if r < 0 or r >= H or c < 0 or c >= W:
            return False
        
        # Collision check
        if obs_map[r, c] >= wall_threshold:
            return False
    
    return True


def is_trajectory_in_corridor(trajectory_delta, cur_pose, path_cells, cell_size):
    """
    Check if trajectory endpoint stays within the path corridor ("fence").
    
    The corridor is defined by the cells in path_cells.
    A trajectory is valid if its endpoint lands in one of the path cells.
    
    Args:
        trajectory_delta: (2,) delta from current position
        cur_pose: (row, col) current position
        path_cells: List of CellNode objects defining the corridor
        cell_size: Size of each cell in pixels
        
    Returns:
        True if trajectory stays in corridor, False otherwise
    """
    if path_cells is None or len(path_cells) == 0:
        return True  # No fence constraint if no path
    
    endpoint = cur_pose + trajectory_delta
    
    # Get cell index of endpoint
    endpoint_cell_r = int(endpoint[0] // cell_size)
    endpoint_cell_c = int(endpoint[1] // cell_size)
    endpoint_cell_idx = (endpoint_cell_r, endpoint_cell_c)
    
    # Check if endpoint cell is in the path corridor
    path_indices = {cell.index for cell in path_cells}
    
    # Also allow current cell (robot might not have left yet)
    current_cell_r = int(cur_pose[0] // cell_size)
    current_cell_c = int(cur_pose[1] // cell_size)
    path_indices.add((current_cell_r, current_cell_c))
    
    return endpoint_cell_idx in path_indices


def select_best_trajectory(trajectories, cur_pose, obs_map, goal_direction=None,
                           path_cells=None, cell_size=25):
    """
    Select the best trajectory from sampled candidates.
    
    Priority:
    1. Collision-free AND in corridor ("fence")
    2. Among valid, closest to goal direction
    3. Fallback: collision-free but outside corridor
    4. Final fallback: mean trajectory if all collide
    
    Args:
        trajectories: (N, 2) array of delta predictions
        cur_pose: (row, col) current robot position
        obs_map: Observation map for collision checking
        goal_direction: Optional (row, col) preferred direction
        path_cells: Optional list of CellNodes defining the corridor fence
        cell_size: Cell size in pixels (for fence check)
        
    Returns:
        best_delta: (2,) best trajectory delta
        best_idx: Index of selected trajectory (-1 if using mean fallback)
        collision_free_count: Number of collision-free trajectories
    """
    if trajectories is None or len(trajectories) == 0:
        return np.array([0, 0]), -1, 0
    
    # Categorize trajectories:
    # - in_fence_free: collision-free AND in corridor
    # - out_fence_free: collision-free but outside corridor
    in_fence_free = []
    in_fence_indices = []
    out_fence_free = []
    out_fence_indices = []
    
    use_fence = path_cells is not None and len(path_cells) > 0
    
    for i, delta in enumerate(trajectories):
        end_pose = cur_pose + delta
        if is_path_collision_free(cur_pose, end_pose, obs_map):
            if use_fence:
                if is_trajectory_in_corridor(delta, cur_pose, path_cells, cell_size):
                    in_fence_free.append(delta)
                    in_fence_indices.append(i)
                else:
                    out_fence_free.append(delta)
                    out_fence_indices.append(i)
            else:
                # No fence constraint
                in_fence_free.append(delta)
                in_fence_indices.append(i)
    
    collision_free_count = len(in_fence_free) + len(out_fence_free)
    in_fence_count = len(in_fence_free)
    
    # Debug output
    if use_fence:
        print(f"  [FENCE] {in_fence_count}/{collision_free_count} trajectories in corridor")
    
    # Priority 1: Select from in-fence trajectories
    if len(in_fence_free) > 0:
        candidates = in_fence_free
        candidate_indices = in_fence_indices
    elif len(out_fence_free) > 0:
        # Fallback: use out-of-fence but collision-free
        print(f"  [FENCE] WARNING: No in-corridor trajectories! Using out-of-corridor fallback.")
        candidates = out_fence_free
        candidate_indices = out_fence_indices
    else:
        # Final fallback: mean trajectory
        print(f"  [FENCE] WARNING: All trajectories collide! Using mean fallback.")
        return np.mean(trajectories, axis=0), -1, 0
    
    # Select best among candidates (closest to goal direction)
    if goal_direction is not None and len(candidates) > 1:
        goal_norm = goal_direction / (np.linalg.norm(goal_direction) + 1e-6)
        best_score = -np.inf
        best_idx = 0
        
        for i, delta in enumerate(candidates):
            delta_norm = delta / (np.linalg.norm(delta) + 1e-6)
            score = np.dot(delta_norm, goal_norm)  # Cosine similarity
            if score > best_score:
                best_score = score
                best_idx = i
        
        return candidates[best_idx], candidate_indices[best_idx], collision_free_count
    
    # Return first valid trajectory
    return candidates[0], candidate_indices[0], collision_free_count


def visualize_trajectory_distribution(ax, cur_pose, trajectories, goal_pose=None,
                                       patch_offset=None, alpha=0.3, 
                                       obs_map=None, best_idx=None, collision_free_count=None):
    """
    Visualize sampled trajectories on the given axes.
    
    Args:
        ax: matplotlib axes
        cur_pose: (row, col) current robot position
        trajectories: (N, 2) array of delta predictions from current pose
        goal_pose: optional (row, col) target position
        patch_offset: (row, col) offset if visualizing on cropped patch
        alpha: transparency for trajectory lines
        obs_map: Optional observation map for collision coloring
        best_idx: Index of selected best trajectory
        collision_free_count: Number of collision-free trajectories
    """
    if trajectories is None or len(trajectories) == 0:
        return
    
    offset = patch_offset if patch_offset is not None else np.array([0, 0])
    cur_viz = cur_pose - offset
    
    # Categorize trajectories by collision status if obs_map provided
    collision_free_lines = []
    collision_lines = []
    
    for i, delta in enumerate(trajectories):
        end_viz = cur_viz + delta
        line = [cur_viz[::-1], end_viz[::-1]]  # (col, row) for matplotlib
        
        if obs_map is not None:
            end_pose = cur_pose + delta
            if is_path_collision_free(cur_pose, end_pose, obs_map):
                collision_free_lines.append(line)
            else:
                collision_lines.append(line)
        else:
            collision_free_lines.append(line)
    
    # Draw collision paths (red, dim)
    if collision_lines:
        lc_collision = LineCollection(collision_lines, colors='red', alpha=alpha*0.5, linewidths=0.5)
        ax.add_collection(lc_collision)
    
    # Draw collision-free paths (green)
    if collision_free_lines:
        lc_free = LineCollection(collision_free_lines, colors='lime', alpha=alpha, linewidths=0.8)
        ax.add_collection(lc_free)
    
    # Draw selected best trajectory (thick blue)
    if best_idx is not None and 0 <= best_idx < len(trajectories):
        best_delta = trajectories[best_idx]
        best_end = cur_viz + best_delta
        ax.plot([cur_viz[1], best_end[1]], [cur_viz[0], best_end[0]], 
                'b-', linewidth=3, label='Selected', zorder=10)
    else:
        # Draw mean trajectory as fallback
        mean_delta = np.mean(trajectories, axis=0)
        mean_end = cur_viz + mean_delta
        ax.plot([cur_viz[1], mean_end[1]], [cur_viz[0], mean_end[0]], 
                'r-', linewidth=2, label='Mean (fallback)')
    
    # Draw current position
    ax.scatter([cur_viz[1]], [cur_viz[0]], c='blue', s=100, marker='o', 
               zorder=5, label='Robot')
    
    # Draw goal if provided
    if goal_pose is not None:
        goal_viz = goal_pose - offset
        ax.scatter([goal_viz[1]], [goal_viz[0]], c='green', s=100, marker='*',
                   zorder=5, label='Goal')
    
    # Calculate and show statistics
    std_delta = np.std(trajectories, axis=0)
    spread = np.linalg.norm(std_delta)
    
    if collision_free_count is not None:
        title = f'Trajectory Distribution (N={len(trajectories)}, σ={spread:.1f}px, {collision_free_count} free)'
    else:
        title = f'Trajectory Distribution (N={len(trajectories)}, σ={spread:.1f}px)'
    ax.set_title(title)


# ============================================================================
# 2. CELL GRAPH VISUALIZATION
# ============================================================================

def visualize_cell_graph(ax, cell_manager, obs_map=None, pred_mean_map=None,
                          current_pose=None, target_cell=None, 
                          path_to_target=None, show_edges=True, pd_size=500,
                          overlay_mode=False, start_cell=None, astar_path=None, show_scores=False,
                          show_cell_boundaries=False):
    """
    Visualize the cell graph with Real (observed) and Ghost (hallucinated) nodes.
    
    Args:
        ax: matplotlib axes
        cell_manager: CellManager instance
        obs_map: observation map for background
        pred_mean_map: prediction map (optional, for ghost coloring)
        current_pose: (row, col) robot position
        target_cell: target CellNode
        path_to_target: list of CellNodes representing path from current to target
        show_edges: whether to draw edges between connected cells
        pd_size: padding size to subtract for visualization
        overlay_mode: if True, don't draw background map (assumes already drawn)
        start_cell: start cell when target was locked (shown with red diamond)
        astar_path: A* local path to visualize (numpy array of [row, col] coordinates)
        show_scores: whether to show node scores as text labels
        show_cell_boundaries: draw dashed cell boundaries
    """
    if not overlay_mode and obs_map is not None:
        # Show map as background (subtract padding for visualization)
        # Invert colormap: 0=free→white, 1=occupied→black
        h, w = obs_map.shape
        display_obs = obs_map[pd_size:h-pd_size, pd_size:w-pd_size] if pd_size > 0 else obs_map
        ax.imshow(1 - display_obs, cmap='gray', vmin=0, vmax=1)  # Invert!
    
    # Collect nodes by type
    real_nodes = []
    ghost_nodes = []
    blocked_nodes = []
    
    for idx, cell in cell_manager.cells.items():
        center = cell.center - pd_size  # Adjust for padding
        if center[0] < 0 or center[1] < 0:
            continue
            
        if cell.is_blocked:
            blocked_nodes.append((center, cell))
        elif cell.is_ghost:
            ghost_nodes.append((center, cell))
        else:
            real_nodes.append((center, cell))

    # Draw cell boundaries (mild dashed grid) for displayed cells
    if show_cell_boundaries:
        half = cell_manager.cell_size / 2.0
        for _, cell in cell_manager.cells.items():
            if cell.is_blocked:
                continue
            center = cell.center - pd_size
            if center[0] < 0 or center[1] < 0:
                continue
            top = float(center[0] - half)
            left = float(center[1] - half)
            rect = mpatches.Rectangle(
                (left, top),  # (x, y)
                cell_manager.cell_size,
                cell_manager.cell_size,
                fill=False,
                edgecolor=(0.4, 0.4, 0.4, 0.25),
                linewidth=0.6,
                linestyle=(0, (3, 3)),
                zorder=2,
            )
            ax.add_patch(rect)
    
    # Draw edges first (so nodes appear on top)
    if show_edges:
        real_edge_lines = []
        real_edge_costs = []
        ghost_edge_lines = []
        ghost_edge_costs = []
        cmap = plt.get_cmap("magma")
        
        for idx, cell in cell_manager.cells.items():
            if cell.is_blocked:
                continue
            center = cell.center - pd_size
            if center[0] < 0 or center[1] < 0:
                continue
            for neighbor in cell.neighbors:
                if neighbor.is_blocked:
                    continue
                n_center = neighbor.center - pd_size
                if n_center[0] < 0 or n_center[1] < 0:
                    continue
                # Draw edge (use (col, row) for matplotlib)
                edge = [center[::-1], n_center[::-1]]
                
                edge_key = tuple(sorted([cell.index, neighbor.index]))
                edge_cost = cell_manager.edge_costs.get(edge_key, 0.0)
                if cell.is_ghost or neighbor.is_ghost:
                    ghost_edge_lines.append(edge)
                    ghost_edge_costs.append(edge_cost)
                else:
                    real_edge_lines.append(edge)
                    real_edge_costs.append(edge_cost)
        
        if real_edge_lines:
            norm_vals = normalize_edge_risks(np.array(real_edge_costs, dtype=float))
            lc = LineCollection(
                real_edge_lines,
                colors=cmap(norm_vals),
                alpha=0.7,
                linewidths=1.1,
            )
            ax.add_collection(lc)
        if ghost_edge_lines:
            norm_vals = normalize_edge_risks(np.array(ghost_edge_costs, dtype=float))
            lc = LineCollection(
                ghost_edge_lines,
                colors=cmap(norm_vals),
                alpha=0.6,
                linewidths=1.0,
                linestyles='dashed',
            )
            ax.add_collection(lc)
    
    # Draw nodes (smaller icons)
    if real_nodes:
        real_centers = np.array([n[0] for n in real_nodes])
        ax.scatter(real_centers[:, 1], real_centers[:, 0], 
                   c='blue', s=18, marker='s', alpha=0.7, label='Real Cells')
    
    if ghost_nodes:
        ghost_centers = np.array([n[0] for n in ghost_nodes])
        # Color by propagated value (uncertainty flow)
        values = [n[1].propagated_value for n in ghost_nodes]
        scatter = ax.scatter(ghost_centers[:, 1], ghost_centers[:, 0],
                            c=values, cmap='Oranges', s=25, marker='o', 
                            alpha=0.8, edgecolors='orange', linewidths=0.8,
                            label='Ghost Cells', vmin=0)
    
    # Only show blocked nodes that have been actively explored (have neighbors)
    # This filters out the massive grid of blocked nodes outside valid space
    if blocked_nodes:
        # Filter: only show blocked nodes that have at least 1 neighbor
        active_blocked = [(c, n) for c, n in blocked_nodes if len(n.neighbors) > 0]
        if active_blocked:
            blocked_centers = np.array([n[0] for n in active_blocked])
            ax.scatter(blocked_centers[:, 1], blocked_centers[:, 0],
                       c='red', s=20, marker='x', alpha=0.5, label='Blocked')
    
    # Highlight current cell (red diamond, smaller)
    if current_pose is not None:
        cur_cell_idx = cell_manager.get_cell_index(current_pose)
        cur_cell = cell_manager.get_cell(cur_cell_idx)
        if cur_cell is not None:
            center = cur_cell.center - pd_size
            ax.scatter([center[1]], [center[0]], c='red', s=80, marker='D',
                       zorder=10, label='Current Cell', edgecolors='black', linewidths=0.5)
    
    # Highlight target cell (diamond shape, transparent inner, lime border)
    if target_cell is not None:
        center = target_cell.center - pd_size
        ax.scatter([center[1]], [center[0]], marker='D', s=120,
                   facecolors='none', edgecolors='lime', linewidths=2.0,
                   zorder=10, label='Target Cell')
    
    # Highlight start cell (red diamond, same size as target)
    if start_cell is not None:
        center = start_cell.center - pd_size
        ax.scatter([center[1]], [center[0]], marker='D', s=120,
                   facecolors='none', edgecolors='red', linewidths=2.0,
                   zorder=10, label='Start Cell')
    
    # Highlight path from current cell to target (magenta dashed line)
    if path_to_target is not None and len(path_to_target) > 1:
        # Draw thick magenta dashed path line connecting cell centers
        path_centers = np.array([c.center - pd_size for c in path_to_target])
        ax.plot(path_centers[:, 1], path_centers[:, 0], 
               color='#FF00FF', linestyle='--', linewidth=2.5, 
               alpha=0.9, label=get_graph_path_label(), zorder=8)
    
    # Highlight A* local path (red solid line)
    if astar_path is not None and len(astar_path) > 1:
        # A* path is in (row, col) format, convert to display coordinates
        astar_display = astar_path - pd_size
        ax.plot(astar_display[:, 1], astar_display[:, 0],
               'r-', linewidth=2.0, alpha=0.8, label='A* Path', zorder=9)
    
    # Show node scores as text labels (for debugging)
    if show_scores:
        for idx, cell in cell_manager.cells.items():
            center = cell.center - pd_size
            if center[0] < 0 or center[1] < 0:
                continue
            # Use propagated_value as the score (final value after diffusion)
            score = cell.propagated_value
            if score > 0:  # Only show non-zero scores
                ax.text(center[1], center[0], f'{score:.2f}',
                       fontsize=6, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f'Cell Graph (Real: {len(real_nodes)}, Ghost: {len(ghost_nodes)})')


def create_combined_visualization(fig, cur_pose, trajectories, cell_manager,
                                   obs_map, pred_mean_map, pred_var_map,
                                   goal_pose=None, target_cell=None, pd_size=500):
    """
    Create a combined 2x2 visualization figure.
    
    Layout:
        [Obs Map + Trajectories] [GT Map / Variance]
        [Cell Graph]            [Mean Prediction]
    """
    axes = fig.subplots(2, 2)
    
    # Top-left: Observation + Trajectory Distribution
    h, w = obs_map.shape
    display_obs = obs_map[pd_size:h-pd_size, pd_size:w-pd_size] if pd_size > 0 else obs_map
    axes[0, 0].imshow(display_obs, cmap='gray', vmin=0, vmax=1)
    visualize_trajectory_distribution(
        axes[0, 0], cur_pose, trajectories, 
        goal_pose=goal_pose, patch_offset=np.array([pd_size, pd_size])
    )
    axes[0, 0].set_title('Trajectory Distribution')
    
    # Top-right: Variance Map
    if pred_var_map is not None:
        display_var = pred_var_map[pd_size:h-pd_size, pd_size:w-pd_size] if pd_size > 0 else pred_var_map
        axes[0, 1].imshow(display_var, cmap='hot')
        axes[0, 1].set_title('Prediction Variance')
    
    # Bottom-left: Cell Graph
    visualize_cell_graph(
        axes[1, 0], cell_manager, obs_map, pred_mean_map,
        current_pose=cur_pose, target_cell=target_cell, pd_size=pd_size
    )
    
    # Bottom-right: Mean Prediction
    if pred_mean_map is not None:
        display_mean = pred_mean_map[pd_size:h-pd_size, pd_size:w-pd_size] if pd_size > 0 else pred_mean_map
        axes[1, 1].imshow(display_mean, cmap='gray', vmin=0, vmax=1)
        axes[1, 1].set_title('Mean Prediction')
    
    plt.tight_layout()
    return axes
