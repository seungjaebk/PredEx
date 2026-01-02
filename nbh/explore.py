"""
NBH (Next Best Hallucination) - Main Exploration Script

This script runs the NBH exploration algorithm using:
- Ghost cells for predictive exploration
- Scent diffusion for target selection
- A* or Flow Matching for local planning
"""

import numpy as np
import os 
import sys

# ==============================================================================
# Path setup for NBH repository structure
# ==============================================================================
nbh_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, nbh_root)

# Add external dependencies to path
external_path = os.path.join(nbh_root, 'external')
lama_path = os.path.join(external_path, 'lama')
sys.path.insert(0, lama_path)

import cv2
import time
from omegaconf import OmegaConf
import hydra 
import torch 
from torchvision.transforms import ToTensor, Resize, Compose
import pyastar2d    
import json
from skimage.measure import block_reduce
import albumentations as A
import traceback
import argparse
from pdb import set_trace as bp
from collections import deque
from queue import PriorityQueue
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# ==============================================================================
# NBH module imports
# ==============================================================================
from nbh.exploration_config import (
    DELTA_SCALE, CELL_SIZE, WAYPOINT_REACHED_TOLERANCE, WAYPOINT_STALE_STEPS,
    MAX_TARGET_DISTANCE
)
from nbh.graph_utils import CellManager
from nbh.flow_planner import load_flow_model
from nbh.high_level_planner import WaypointState
from nbh.waypoint_utils import select_waypoint_cell
from nbh.flow_viz_utils import (
    sample_multiple_trajectories, visualize_trajectory_distribution,
    visualize_cell_graph, select_best_trajectory
)
from nbh.flow_planner import (
    build_flow_input_tensor, sample_flow_delta, extract_local_patch, sample_trajectory_for_cell
)
from nbh.lama_utils import (
    get_pred_maputils_from_viz, get_lama_padding_transform, get_padded_obs_map, get_padded_gt_map
)
from nbh.frontier_utils import (
    update_mission_status, is_locked_frontier_center_valid, reselect_frontier_from_frontier_region_centers,
    determine_local_planner
)
from nbh.exploration_config import get_options_dict_from_yml

# ==============================================================================
# Utility imports
# ==============================================================================
from utils.lama_pred_utils import load_lama_model, visualize_prediction, get_lama_transform, convert_obsimg_to_model_input
from utils.calc_metrics import calculate_iou_kth
from utils import simple_mask_utils as smu 
from utils import sim_utils
from training.train_flow_matching import FlowMatchingModel
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime
from contextlib import contextmanager

@contextmanager
def dummy_context_manager():
    yield None


DEBUG_PROPAGATION_STATS = True


def align_pred_map(pred_map, obs_shape, fill_value):
    if pred_map is None:
        return None
    pred_h, pred_w = pred_map.shape
    obs_h, obs_w = obs_shape
    aligned = np.full((obs_h, obs_w), fill_value, dtype=pred_map.dtype)

    if pred_h >= obs_h:
        src_top = (pred_h - obs_h) // 2
        dst_top = 0
        copy_h = obs_h
    else:
        src_top = 0
        dst_top = (obs_h - pred_h) // 2
        copy_h = pred_h

    if pred_w >= obs_w:
        src_left = (pred_w - obs_w) // 2
        dst_left = 0
        copy_w = obs_w
    else:
        src_left = 0
        dst_left = (obs_w - pred_w) // 2
        copy_w = pred_w

    aligned[dst_top:dst_top+copy_h, dst_left:dst_left+copy_w] = pred_map[
        src_top:src_top+copy_h, src_left:src_left+copy_w
    ]
    return aligned


def get_lama_pred_from_obs(cur_obs_img, lama_model, lama_map_transform, device):
    """Get LAMA model prediction from observation image."""
    cur_obs_img_3chan = np.stack([cur_obs_img, cur_obs_img, cur_obs_img], axis=2)
    input_lama_batch, lama_mask = convert_obsimg_to_model_input(cur_obs_img_3chan, lama_map_transform, device)
    lama_pred_alltrain = lama_model(input_lama_batch)
    lama_pred_alltrain_viz = visualize_prediction(lama_pred_alltrain, lama_mask)
    return cur_obs_img_3chan, input_lama_batch, lama_mask, lama_pred_alltrain, lama_pred_alltrain_viz

def run_exploration_comparison_for_map(args):
    map_folder_path = args['map_folder_path']
    models_list = args['models_list']
    lama_model = args['lama_model']
    lama_map_transform = args['lama_map_transform']
    pred_vis_configs = args['pred_vis_configs']
    lidar_sim_configs = args['lidar_sim_configs']
    start_pose = args['start_pose']
    modes_to_test = args['modes_to_test']
    unknown_as_occ = args['unknown_as_occ']
    use_distance_transform_for_planning = args['use_distance_transform_for_planning']
    upen_config  = args['upen_config']
    print("Running exploration comparison for map: ", map_folder_path)
    
    # Load in occupancy map and valid space map
    map_occ_npy_path = os.path.join(map_folder_path, 'occ_map.npy')
    assert os.path.exists(map_occ_npy_path), "Occupancy map path does not exist: {}".format(map_occ_npy_path)
    map_valid_space_npy_path = os.path.join(map_folder_path, 'valid_space.npy')
    assert os.path.exists(map_valid_space_npy_path), "Valid space map path does not exist: {}".format(map_valid_space_npy_path)
    occ_map, validspace_map = sim_utils.get_kth_occ_validspace_map(map_occ_npy_path, map_valid_space_npy_path)
    map_name = os.path.dirname(map_occ_npy_path)

    # Sample random start pose, if start_pose is None
    if start_pose is None:
        buffer_start_pose = 2
        start_pose = smu.sample_free_position_given_buffer(occ_map, validspace_map, buffer_start_pose)
        assert start_pose is not None, "Could not sample start pose"

    plt.imshow(occ_map)
    plt.scatter(start_pose[1], start_pose[0], c='r', s=10, marker='*')
    plt.title('GT Map (Red: Start Pose)')
    plt.savefig('gt_map.png')
    plt.close()
        
    # get overall experiment name
    # exp_title based on start time YYYYMMDD_HHMMSS, and name of map
    folder_name = os.path.basename(map_name)
    comp_exp_title = time.strftime("%Y%m%d_%H%M%S") + '_' + folder_name + '_' + str(start_pose[0]) + '_' + str(start_pose[1])
        
    flow_runtime = args.get('flow_runtime')
    for mode in modes_to_test:
        run_exploration_for_map(
            occ_map,
            comp_exp_title,
            models_list,
            lama_model,
            lama_map_transform,
            pred_vis_configs,
            lidar_sim_configs,
            mode,
            start_pose,
            unknown_as_occ,
            use_distance_transform_for_planning=use_distance_transform_for_planning,
            upen_config=upen_config,
            flow_runtime=flow_runtime,
            validspace_map=validspace_map,
        )

def run_exploration_for_map(occ_map, exp_title, models_list,lama_alltrain_model, lama_map_transform, pred_vis_configs, lidar_sim_configs, mode, \
    start_pose, unknown_as_occ, use_distance_transform_for_planning, upen_config=None, flow_runtime=None, validspace_map=None):
    t = 0
    mission_status_save_path = None
    try: 
        print("exp_title:", exp_title)
        start_exp_time = time.time()
        pixel_per_meter = lidar_sim_configs['pixel_per_meter']
        use_model = determine_use_model(mode) #determine if the mode requires lama model
        flow_logging_cfg = collect_opts.flow_logging if hasattr(collect_opts, "flow_logging") else None
        flow_logging_enabled = bool(flow_logging_cfg.enabled) if flow_logging_cfg is not None else False
        flow_model = None
        flow_device = None
        flow_num_steps = 1
        flow_crop_radius = getattr(collect_opts, "flow_crop_radius", 128)
        if flow_runtime is not None:
            flow_model = flow_runtime.get("model")
            flow_device = flow_runtime.get("device")
            flow_num_steps = flow_runtime.get("num_steps", 1)
            flow_crop_radius = flow_runtime.get("crop_radius", flow_crop_radius)
        use_flow_planner = (mode == 'nbh') and (flow_model is not None)
        
        # ==============================================================
        # LOCAL PLANNER MODE SWITCH (for debugging high-level planner)
        # ==============================================================
        # 'astar': Use A* for local planning (proven to work) - debug high-level
        # 'flow': Use Flow model for local planning (what we're developing)
        USE_ASTAR_FOR_LOCAL = (collect_opts.local_planner == 'astar')
        CELL_SIZE_CONFIG = collect_opts.cell_size  # From yaml/CLI
        nbh_cfg = getattr(collect_opts, "nbh", {})

        promotion_cfg = {
            "graph_max_ghost_distance": nbh_cfg.get("graph_max_ghost_distance", 2),
            "graph_obs_blocked_ratio": nbh_cfg.get("graph_obs_blocked_ratio", 0.3),
            "graph_unknown_ratio_threshold": nbh_cfg.get("graph_unknown_ratio_threshold", 0.5),
            "graph_centroid_blocked_threshold": nbh_cfg.get("graph_centroid_blocked_threshold", 0.8),
            "graph_ghost_pred_mean_free_threshold": nbh_cfg.get("graph_ghost_pred_mean_free_threshold", 0.4),
            "graph_ghost_pred_var_max_threshold": nbh_cfg.get("graph_ghost_pred_var_max_threshold", 0.3),
            "graph_diffuse_gamma": nbh_cfg.get("graph_diffuse_gamma", 0.95),
            "graph_diffuse_iterations": nbh_cfg.get("graph_diffuse_iterations", 50),
            "graph_diffuse_on_update": nbh_cfg.get("graph_diffuse_on_update", False),
        }

        connectivity_cfg = {}
        debug_cfg = {}
        
        if USE_ASTAR_FOR_LOCAL:
            print("=" * 60)
            print("⚠️  DEBUG MODE: Using A* for local planning (not Flow)")
            print("    This tests the HIGH-LEVEL planner (cell graph, targets)")
            print(f"    Config: local_planner={collect_opts.local_planner}, cell_size={CELL_SIZE_CONFIG}px")
            print("=" * 60)
        else:
            print("=" * 60)
            print("🚀 PRODUCTION MODE: Using Flow for local planning")
            print(f"    Config: local_planner={collect_opts.local_planner}, cell_size={CELL_SIZE_CONFIG}px")
            print("=" * 60)

        # Planner setup (Mapper, Planner)
        mapper = sim_utils.Mapper(occ_map, lidar_sim_configs, use_distance_transform_for_planning=use_distance_transform_for_planning)
        
        # Initialize Graph Planner (Cell Manager)
        cell_manager = None
        if mode == 'nbh':
            # Robot-centric cell manager: start_pose becomes centroid of cell (0, 0)
            # Cell indices can be negative (robot can move "backwards" from start)
            cell_manager = CellManager(
                cell_size=CELL_SIZE_CONFIG,
                start_pose=start_pose,
                valid_space_map=validspace_map,
                promotion_cfg=promotion_cfg,
                connectivity_cfg=connectivity_cfg,
                debug_cfg=debug_cfg,
            )
            print(f"Initialized NBH Cell Manager (Cell Size: {CELL_SIZE_CONFIG}px = {CELL_SIZE_CONFIG/10}m, Origin: {start_pose})")
            
        if mode != 'upen':
            planner_mode = mode
            if mode == 'nbh':
                planner_mode = 'mapex' # Fallback for initialization
            frontier_planner = sim_utils.FrontierPlanner(score_mode=planner_mode)

        # Create a new directory for experiment/exp_title
        exp_title = exp_title + '_' + mode
        exp_dir = os.path.join(output_root_dir, exp_title)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        # Create subdirectory (global_obs, run_viz, graph_map) and save paths
        global_obs_dir = os.path.join(exp_dir, 'global_obs')
        run_viz_dir = os.path.join(exp_dir, 'run_viz')
        graph_map_dir = os.path.join(exp_dir, 'graph_map')
        for dir_path in [global_obs_dir, run_viz_dir, graph_map_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        gt_map_save_path = os.path.join(exp_dir, 'gt_map.png')
        odom_npy_save_path = os.path.join(exp_dir, 'odom.npy')

        # Flow logging setup
        flow_logging_dir = None
        flow_episode_samples = []
        flow_meta = None
        if flow_logging_enabled:
            flow_logging_dir = os.path.join(exp_dir, 'flow_samples')
            os.makedirs(flow_logging_dir, exist_ok=True)
            flow_meta = {
                "exp_title": exp_title,
                "mode": mode,
                "start_pose": start_pose.tolist() if hasattr(start_pose, "tolist") else list(start_pose),
                "pixel_per_meter": pixel_per_meter,
                "use_model": use_model,
            }


        # Visualization Setup 
        if mode in ['pipe', 'mapex', 'nbh']:
            plt_row = 2
            plt_col = 2
            fig, ax = plt.subplots(plt_row, plt_col, figsize=(16, 12))
            ax_flatten = ax.flatten()
            ax_gt = ax_flatten[0]
            ax_obs = ax_flatten[1]
            ax_mean_map = ax_flatten[2]
            ax_cell_graph = ax_flatten[3]
            pred_maputils = None
            var_map = None
            mean_map = None
            lama_reduced_pred_var = None 
            padded_gt_map = None
            sampled_trajectories = None  # Store for visualization
            path_to_target = None  # Store path for visualization
        elif mode in ['nbv-2d', 'pw-nbv-2d']:
            plt_row = 2
            plt_col = 1
            fig, ax = plt.subplots(plt_row, plt_col, figsize=(15, 10))
            ax_flatten = ax.flatten()
            ax_gt = ax_flatten[0]
            ax_obs = ax_flatten[1]
            pred_maputils = None
            var_map = None
            mean_map = None
            lama_reduced_pred_var = None 
            padded_gt_map = None
        else:
            raise ValueError(f"Unknown method: {mode}")

        # Initialize mission status save path
        mission_status_save_path = os.path.join(exp_dir, 'mission_status.json')
        mission_failed = False
        update_mission_status(start_time=start_exp_time, cur_step=0, mission_complete=False, fail_reason="", mission_status_save_path=mission_status_save_path)

        #initial observation
        cur_pose = np.array(start_pose)
        mapper.observe_and_accumulate_given_pose(cur_pose)
        ind_to_move_per_step = 3
        pose_list = np.atleast_2d(cur_pose) # Assumes last pose is the current pose

        pd_size = 500 #padding_size; padded pixels for raycast at the boundary of map
        #there is a small offset between observed map size and lama prediction output size
        gt_h, gt_w = mapper.gt_map.shape[0],mapper.gt_map.shape[1]
        pad_h = gt_h%16
        pad_w = gt_w%16
        if pad_h == 0:
            pad_h1 = 0
            pad_h2 = 0
        else:
            pad_h1 = int((16-pad_h)/2)
            pad_h2 = 16-pad_h - pad_h1
        if pad_w == 0:
            pad_w1 = 0
            pad_w2 = 0
        else:
            pad_w1 = int((16-pad_w)/2)
            pad_w2 = 16-pad_w - pad_w1

        # initial saves (gt_map, pose_list)
        cv2.imwrite(gt_map_save_path, smu.convert_01_single_channel_to_0_255_3_channel(mapper.gt_map))
        np.save(odom_npy_save_path, pose_list)

        locked_frontier_center = None
        latest_padded_obs_map = None
        latest_pred_mean_map = None
        latest_pred_var_map = None
        
        # Stagnation Detection
        stagnation_window = 50
        stagnation_dist_thresh = 50.0 # Increased from 30.0 to catch larger loops
        recent_poses = deque(maxlen=stagnation_window)
        stagnation_mode = False
        stagnation_counter = 0
        
        # Oscillation Override
        oscillation_override = False
        oscillation_override_counter = 0
        OSCILLATION_OVERRIDE_DURATION = 20 # Steps to force direct movement
        
        # Gateway Locking & Waypoint Logic
        current_waypoint = None
        locked_target_cell = None  # Store the cell object for visualization
        locked_path_to_target = None  # Store BFS path for graph-based distance
        locked_trajectory = None  # Store the selected flow trajectory (computed once per cell)
        locked_trajectory_samples = None  # Store all sampled trajectories for visualization
        WAYPOINT_REACHED_TOLERANCE = 1.0 # Pixels - must reach very close to exact center
        CELL_SIZE = CELL_SIZE_CONFIG  # From yaml/CLI
        
        # Stale Waypoint Detection - give up if stuck for too long
        waypoint_steps_counter = 0
        WAYPOINT_STALE_STEPS = 30  # Max steps per cell before marking stale
        
        # Maximum target distance - don't target centroids too far away
        MAX_TARGET_DISTANCE = 60.0  # pixels (~6m) - keeps targets local and achievable
        
        ### Main Loop
        # Create a timestamp string with date and time, e.g. "20250319_155031"
        if collect_opts.log_iou:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_subdirectory_name = f"{timestamp}_test"

            # Use this for your log directory
            log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments', output_subdirectory_name)
            os.makedirs(log_dir, exist_ok=True)
            iou_log_file = os.path.join(log_dir, f"{exp_title}.txt")


        # Determine time_step based on exp_title
        parts = exp_title.split('_')
        file_name = '_'.join(parts[2:-3])
        # Define time step settings for each category: (with_log, without_log)
        time_settings = {
            'large': (6001, 2001),
            'small': (1501, 501),
            'medium': (3001, 1001)
        }

        # Determine the map category based on file_name
        if file_name in ('50015847', '50015848'):
            category = 'large'
        elif file_name in ('50052749', '50052750'):
            category = 'small'
        else:
            category = 'medium'

        # Choose the appropriate time step based on whether logging is enabled
        time_step = time_settings[category][0] if collect_opts.log_iou else time_settings[category][1]

        iou_90_flag = False
        iou_95_flag = False
        context = open(iou_log_file, "a") if collect_opts.log_iou else dummy_context_manager()

        with context as log_file:
            for t in range(time_step):
                start_mission_i_time = time.time()
                show_plt = (t % collect_opts.show_plt_freq == 0) or (t == collect_opts.mission_time - 1)
                
                # Frontier detection (flow mode only)
                if t == 0:
                    frontier_region_centers_unscored, filtered_map, num_large_regions = frontier_planner.get_frontier_centers_given_obs_map(mapper.obs_map)
                
                    if len(frontier_region_centers_unscored) == 0:
                        mission_failed = True
                        update_mission_status(start_time=start_exp_time, cur_step=t, mission_complete=False, fail_reason="frontier_region_no_large_region", mission_status_save_path=mission_status_save_path)
                        break

                # Get inflated obs map for local planner
                occ_grid_pyastar = mapper.get_inflated_planning_maps(unknown_as_occ=unknown_as_occ)   
                
                # Check if close enough to locked frontier center, if so unlock 
                if locked_frontier_center is not None:
                    if np.linalg.norm(locked_frontier_center - cur_pose) < collect_opts.cur_pose_dist_threshold_m * pixel_per_meter:
                        locked_frontier_center = None

                # Check if we need a new frontier target
                need_new_locked_frontier = not is_locked_frontier_center_valid(locked_frontier_center, occ_grid_pyastar, cur_pose, collect_opts, pixel_per_meter)
                                    
                if need_new_locked_frontier:
                    show_plt = True
                                                        
                    # Predict map using LAMA
                    pred_maputils = None
                    var_map = None
                    var_map_np_for_flow = None
                    mean_map = None
                    padded_gt_map = None
                    
                    if use_model:
                        cur_obs_img = mapper.obs_map.copy()
                        # LAMA global prediction (trained with all training set)
                        cur_obs_img_3chan, input_lama_batch, lama_mask, lama_pred_alltrain, lama_pred_alltrain_viz = \
                            get_lama_pred_from_obs(cur_obs_img, lama_alltrain_model, lama_map_transform, device)
                        # Get ensemble prediction trained on subsets
                        lama_pred_list = []
                        for model_i, model in enumerate(model_list):
                            pred_time_start = time.time()
                            lama_pred = model(input_lama_batch)
                            lama_pred_viz = visualize_prediction(lama_pred, lama_mask)
                            lama_pred_onechan = lama_pred['inpainted'][0][0]
                            lama_pred_list.append(lama_pred_onechan)
                        
                        # Get variance across batch dimension 
                        lama_pred_list = torch.stack(lama_pred_list)
                        var_map = torch.var(lama_pred_list, dim=0)
                        var_map_np_for_flow = var_map.detach().cpu().numpy()
                        mean_map = np.mean(lama_pred_list.cpu().numpy(), axis=0)

                        pred_maputils = get_pred_maputils_from_viz(lama_pred_alltrain_viz)
                    
                    padded_obs_map = get_padded_obs_map(mapper.obs_map)
                    padded_gt_map = get_padded_gt_map(mapper.gt_map)
                    latest_padded_obs_map = padded_obs_map.copy()
                    latest_pred_mean_map = mean_map.copy() if mean_map is not None else None
                    latest_pred_var_map = var_map_np_for_flow.copy() if var_map_np_for_flow is not None else None

                    # Flow mode: get frontiers but don't score them (use cell-based navigation)
                    frontier_region_centers_unscored, filtered_map, num_large_regions = frontier_planner.get_frontier_centers_given_obs_map(mapper.obs_map)
                    frontier_region_centers = frontier_region_centers_unscored
                    frontier_cost_list = np.zeros(len(frontier_region_centers))
                    viz_most_flooded_grid = None
                    viz_medium_flooded_grid = None
                    best_ind = 0
                    medium_ind = 0
                    locked_frontier_center = None  # Flow mode uses cell-based navigation, not locked frontier

                # Local planning (flow mode only)
                if mode == 'nbh':
                    chosen_local_planner = 'nbh'
                else:
                    chosen_local_planner = determine_local_planner(mode)

                flow_obs_patch = None
                flow_mean_patch = None
                flow_var_patch = None
                capture_flow_patch = (flow_logging_enabled or use_flow_planner) and latest_padded_obs_map is not None
                if capture_flow_patch and flow_crop_radius > 0:
                    flow_obs_patch = extract_local_patch(latest_padded_obs_map, cur_pose, flow_crop_radius, pad_value=0.5)
                    include_prediction = True if not flow_logging_enabled else flow_logging_cfg.include_prediction
                    if include_prediction:
                        if latest_pred_mean_map is not None:
                            flow_mean_patch = extract_local_patch(latest_pred_mean_map, cur_pose, flow_crop_radius, pad_value=0.0)
                        if latest_pred_var_map is not None:
                            flow_var_patch = extract_local_patch(latest_pred_var_map, cur_pose, flow_crop_radius, pad_value=0.0)
                
                path = None
                if chosen_local_planner == 'astar':
                    path = pyastar2d.astar_path(occ_grid_pyastar, cur_pose, locked_frontier_center, allow_diagonal=False)
                    while path is None:
                        frontier_selected, locked_frontier_center, frontier_region_centers, frontier_cost_list = reselect_frontier_from_frontier_region_centers(frontier_region_centers, frontier_cost_list, t, start_exp_time, mission_status_save_path)
                        if not frontier_selected:
                            mission_failed = True
                            update_mission_status(start_time=start_exp_time, cur_step=t, mission_complete=False, fail_reason="frontier_region_centers", mission_status_save_path=mission_status_save_path)
                            break
                        path = pyastar2d.astar_path(occ_grid_pyastar, cur_pose, locked_frontier_center, allow_diagonal=False)
                    if mission_failed:
                        break
                
                    plan_x = path[:,0]
                    plan_y = path[:,1]        
                        
                    next_pose = sim_utils.psuedo_traj_controller(plan_x, plan_y, plan_ind_to_use=ind_to_move_per_step)
                elif chosen_local_planner == 'gradient':
                    # Given current pose, find the steepest gradient direction to go in
                    for _ in range(ind_to_move_per_step): # Move the same steps as psuedo_traj_controller + A*
                        next_pose = gradient_planner(cur_pose, cost_transform_map=cost_transform_map) 
                        cur_pose = next_pose
                elif chosen_local_planner == 'nbh':
                    if not use_flow_planner and not USE_ASTAR_FOR_LOCAL:
                        raise ValueError("Flow planner selected but flow model is not available")
                    
                    # --- GRAPH FLOW LOGIC ---
                    # Update Graph with current observation AND prediction (for Ghosts)
                    # Align prediction maps to obs_map size; pad unknown outside.
                    obs_h, obs_w = mapper.obs_map.shape
                    unpadded_pred_mean = align_pred_map(latest_pred_mean_map, (obs_h, obs_w), fill_value=0.5)
                    unpadded_pred_var = align_pred_map(latest_pred_var_map, (obs_h, obs_w), fill_value=1.0)
                    
                    cell_manager.update_graph(
                        cur_pose, 
                        mapper.obs_map, 
                        unpadded_pred_mean, # Cropped to match obs_map coordinates
                        pred_var_map=unpadded_pred_var,  # Cropped to match obs_map coordinates
                        inflated_occ_grid=occ_grid_pyastar
                    )

                    if DEBUG_PROPAGATION_STATS and cell_manager.cells:
                        values = np.array(
                            [node.propagated_value for node in cell_manager.cells.values()],
                            dtype=np.float32,
                        )
                        nonzero = int(np.count_nonzero(values > 0))
                        print(
                            f"[PROP] t={t} min={float(values.min()):.6f} "
                            f"max={float(values.max()):.6f} mean={float(values.mean()):.6f} "
                            f"nonzero={nonzero}/{len(values)}"
                        )
                    
                    # Ensure we have latest prediction (updated in need_new_locked_frontier block)
                    # If flow mode didn't trigger update, we must check.
                    # Flow mode logic: need_new_locked_frontier is usually False because locked_frontier is None or invalid?
                    # Actually, lines 595+ ensure prediction happens if 'need_new_locked_frontier' is true.
                    # We need to FORCE prediction update periodically for the graph to smell fresh scent.
                    # Let's reuse the prediction variables 'latest_pred_var_map'
                    
                    if latest_pred_var_map is None:
                         # Force initial prediction if missing
                         # Copy-paste prediction logic or rely on loop? 
                         # Let's assume the loop triggered it once at start.
                         pass
                         
                    # --- WAYPOINT LOGIC (STRICT LOCK) ---
                    # We only query the graph for a new target when the current waypoint is reached.
                    # This prevents the target from jumping around as the robot moves.
                    
                    goal_vec_np = np.zeros(2, dtype=np.float32)
                    target_cell = locked_target_cell  # Preserve high-level target until reached/stale
                    
                    if current_waypoint is not None:
                        # We have an active waypoint - check if reached
                        
                        # Graph-based distance: (remaining cells) × CELL_SIZE
                        if locked_path_to_target is not None and len(locked_path_to_target) > 0:
                            # Find current position in path
                            remaining_cells = len(locked_path_to_target)
                            if cell_manager.current_cell is not None:
                                for i, cell in enumerate(locked_path_to_target):
                                    if cell.index == cell_manager.current_cell.index:
                                        remaining_cells = len(locked_path_to_target) - i - 1
                                        break
                            graph_dist = remaining_cells * CELL_SIZE
                        else:
                            # Fallback to Euclidean if no path
                            graph_dist = np.linalg.norm(cur_pose - current_waypoint)
                        
                        # Check if reached: within tolerance of immediate next waypoint
                        dist_to_next_wp = np.linalg.norm(cur_pose - current_waypoint)
                        
                        if dist_to_next_wp > WAYPOINT_REACHED_TOLERANCE:
                            # NOT reached yet - keep targeting locked waypoint
                            target_pos = current_waypoint
                            target_cell = locked_target_cell  # Use stored cell for visualization
                            
                            # --- STALE WAYPOINT DETECTION (Step-based per cell) ---
                            waypoint_steps_counter += 1
                            
                            if waypoint_steps_counter >= WAYPOINT_STALE_STEPS:
                                print(f"Cell STALE! Stuck for {waypoint_steps_counter} steps (graph_dist={graph_dist:.1f}px). Moving to next target.")
                                if locked_target_cell is not None:
                                    locked_target_cell.visit_count += 5  # Penalize this cell
                                # Clear waypoint to get a new target
                                current_waypoint = None
                                locked_target_cell = None
                                locked_path_to_target = None
                                locked_trajectory = None
                                locked_trajectory_samples = None
                                waypoint_steps_counter = 0
                        else:
                            # Waypoint Reached! Clear only the waypoint; keep high-level target
                            print(f"Waypoint Reached! (dist={dist_to_next_wp:.2f}, path_remaining={remaining_cells if 'remaining_cells' in dir() else '?'} cells)")
                            current_waypoint = None
                            locked_trajectory = None
                            locked_trajectory_samples = None
                            waypoint_steps_counter = 0
                    
                    # Only get new target if waypoint is cleared
                    if current_waypoint is None:
                        # Decision Making: Where to flow next?
                        # HYBRID NAVIGATION:
                        # 1. Find high-level target (highest uncertainty ghost cell) - when target changes
                        # 2. Propagate goal scent from target - O(cells), but only when target changes
                        # 3. Use pick_best_neighbor for local navigation - O(4) per step
                        
                        # --- STEP 1: Find/update high-level exploration target ---
                        # Check if target is still reachable
                        target_unreachable = False
                        if target_cell is not None and cell_manager.current_cell is not None:
                            path_to_target = cell_manager.find_path_to_target(cell_manager.current_cell, target_cell)
                            if path_to_target is None:
                                target_unreachable = True
                                print(f"[HIGH-LEVEL] Target {target_cell.index} is now UNREACHABLE, selecting new target")
                        
                        target_change_reasons = []
                        if target_cell is None:
                            target_change_reasons.append("none")
                        elif target_cell.is_blocked:
                            target_change_reasons.append("blocked")
                        if target_cell is not None and cell_manager.current_cell is not None:
                            if target_cell.index == cell_manager.current_cell.index:
                                target_change_reasons.append("reached")
                        if target_unreachable:
                            target_change_reasons.append("unreachable")

                        need_new_high_level_target = len(target_change_reasons) > 0
                        
                        if need_new_high_level_target:
                            reasons_str = ", ".join(target_change_reasons)
                            print(f"[HIGH-LEVEL] Reselecting target (reason: {reasons_str})")
                            # Find best ghost cell (highest uncertainty) that is REACHABLE from current cell
                            exploration_target = cell_manager.find_exploration_target(
                                latest_pred_var_map if latest_pred_var_map is not None else np.zeros_like(mapper.obs_map),
                                current_cell=cell_manager.current_cell  # Only consider reachable cells
                            )
                            
                            if exploration_target is not None:
                                target_cell = exploration_target
                                print(f"[HIGH-LEVEL] New exploration target: {target_cell.index}")
                                
                                # --- STEP 2: Propagate goal scent from target (O(cells), once) ---
                                cell_manager.propagate_goal_scent(target_cell, decay=0.9, iterations=30)
                                
                                # --- Compute path for visualization ---
                                path_to_target = cell_manager.find_path_to_target(
                                    cell_manager.current_cell, target_cell
                                )
                                if path_to_target:
                                    print(f"[PATH] Found path with {len(path_to_target)} cells")
                                else:
                                    print(f"[PATH] No path found to target!")
                            else:
                                print("[HIGH-LEVEL] No exploration target found!")
                                path_to_target = None
                        
                        # --- STEP 3: Get next cell from BFS path (primary) or greedy fallback ---
                        next_cell = None
                        decision_type = "NO_PATH"
                        if path_to_target is not None and cell_manager.current_cell is not None:
                            on_path = any(
                                cell.index == cell_manager.current_cell.index
                                for cell in path_to_target
                            )
                            if on_path and len(path_to_target) > 1:
                                next_cell = select_waypoint_cell(cur_pose, path_to_target, mapper.obs_map)
                                if next_cell is not None:
                                    decision_type = "PATH_LOS"

                        # If path following fails, use greedy neighbor selection
                        if next_cell is None:
                            next_cell, decision_type = cell_manager.pick_best_neighbor(
                                target_cell,
                                mapper.obs_map
                            )
                        
                        if next_cell is not None:
                            target_pos = next_cell.center
                            print(f"[LOCAL] {decision_type} → Cell {next_cell.index}")
                        else:
                            # Fallback: stay in place or use target directly
                            target_pos = cur_pose if target_cell is None else target_cell.center
                            print(f"[LOCAL] No valid neighbor. Fallback to target.")
                        
                        # Lock next_cell as waypoint for local navigation
                        if next_cell is not None:
                            target_pos = next_cell.center
                            
                            # Lock this new waypoint + path for graph-based distance
                            current_waypoint = target_pos
                            locked_target_cell = target_cell  # Store high-level target for visualization
                            locked_path_to_target = path_to_target  # Store path for distance calculation
                            waypoint_steps_counter = 0
                            
                            # --- SAMPLE FLOW TRAJECTORY ONCE PER CELL TRANSITION ---
                            # Build goal vector toward next cell
                            diff = target_pos - cur_pose
                            dist = np.linalg.norm(diff)
                            if dist > 1e-3:
                                goal_vec_for_sampling = (diff / dist).astype(np.float32)
                            else:
                                goal_vec_for_sampling = np.zeros(2, dtype=np.float32)
                            
                            # Sample multiple trajectories
                            flow_tensor_sample, flow_goal_tensor_sample = build_flow_input_tensor(
                                flow_obs_patch, flow_mean_patch, flow_var_patch, goal_vec_for_sampling
                            )
                            if flow_tensor_sample is not None and use_flow_planner:
                                from nbh.flow_viz_utils import sample_multiple_trajectories, select_best_trajectory

                                locked_trajectory_samples = sample_multiple_trajectories(
                                    flow_model, flow_tensor_sample, flow_goal_tensor_sample, flow_device,
                                    num_samples=50, num_steps=flow_num_steps, delta_scale=DELTA_SCALE,
                                    goal_perturb_std=0.5  # ~30° spread
                                )

                                # Select best trajectory (collision-free, in corridor, closest to goal)
                                # Pass path_to_target as "fence" constraint
                                locked_trajectory, best_idx, collision_free_count = select_best_trajectory(
                                    locked_trajectory_samples, cur_pose, padded_obs_map, goal_vec_for_sampling,
                                    path_cells=path_to_target, cell_size=CELL_SIZE
                                )

                                sampled_trajectories = locked_trajectory_samples  # For visualization
                                print(f"  → Sampled {len(locked_trajectory_samples)} trajectories, {collision_free_count} free, selected idx={best_idx}")
                            else:
                                locked_trajectory_samples = None
                                locked_trajectory = goal_vec_for_sampling * DELTA_SCALE  # Fallback: direct
                            
                            # Print graph-based distance
                            path_cells = len(path_to_target) if path_to_target else 1
                            graph_distance = path_cells * CELL_SIZE
                            print(f"New Waypoint Locked: Cell {next_cell.index}, path={path_cells} cells, dist={graph_distance:.1f}px")
                    
                    # Ensure we have a valid target_cell for downstream logic
                    if target_cell is None:
                        target_cell = locked_target_cell
                    
                    # Use next_cell for flow goal if available, otherwise fallback to target_cell
                    nav_target_cell = next_cell if next_cell is not None else target_cell
                    
                    if target_cell is not None:

                        # --- GOAL VECTOR: Direct to target ---
                        # The Flow Model should learn obstacle avoidance from local observation.
                        # We trust the model to figure out how to navigate around walls.
                        
                        diff = target_pos - cur_pose
                        dist = np.linalg.norm(diff)
                        if dist > 1e-3:
                            goal_vec_np = (diff / dist).astype(np.float32)
                        else:
                            goal_vec_np = np.zeros(2, dtype=np.float32)
                        
                        # DEBUG PRINT
                        tp_r, tp_c = target_pos
                        cp_r, cp_c = cur_pose
                        target_xy_global = np.array([tp_c - pd_size, tp_r - pd_size])
                        pose_xy_global = np.array([cp_c - pd_size, cp_r - pd_size])
                        
                        if not USE_ASTAR_FOR_LOCAL:
                            print(f"Target Cell Center (Locked): {target_xy_global}, Current Pose: {pose_xy_global}, Vector: {goal_vec_np}")
                        else:
                            print(f"Target Cell Center (Locked): {target_xy_global}, Current Pose: {pose_xy_global}")
                    
                        
                        # print(f"Graph Decision: {decision_type} -> Cell{target_cell.index}")
                    else:
                        # STUCK? Random walk
                        # print("Graph STUCK. Random walk.")
                        rand_angle = np.random.uniform(0, 2*np.pi)
                        goal_vec_np = np.array([np.sin(rand_angle), np.cos(rand_angle)], dtype=np.float32)

                    # ==============================================================
                    # LOCAL PLANNING: A* (debug mode) or Flow (production mode)
                    # ==============================================================
                    
                    if USE_ASTAR_FOR_LOCAL and target_pos is not None:
                        # ========== A* LOCAL PLANNER (for debugging high-level) ==========
                        # Use A* to plan from current pose to target cell center
                        target_pose_int = tuple(target_pos.astype(int))
                        astar_path = pyastar2d.astar_path(occ_grid_pyastar, tuple(cur_pose), target_pose_int, allow_diagonal=False)
                        
                        if astar_path is not None and len(astar_path) > 1:
                            # Use pseudo trajectory controller to get next pose
                            plan_x = astar_path[:, 0]
                            plan_y = astar_path[:, 1]
                            next_pose = sim_utils.psuedo_traj_controller(plan_x, plan_y, plan_ind_to_use=ind_to_move_per_step)
                            
                            # Compute delta for visualization
                            delta = (next_pose - cur_pose).astype(np.float32)
                            delta_pixels = (next_pose - cur_pose).astype(int)
                            
                            # Store path for visualization
                            path = astar_path
                            print(f"[A* LOCAL] Path found: {len(astar_path)} waypoints to target {target_pose_int}")
                        else:
                            # A* failed - target unreachable
                            print(f"[A* LOCAL] ⚠️ No path to target {target_pose_int}! Staying in place.")
                            next_pose = cur_pose.copy()
                            delta = np.zeros(2, dtype=np.float32)
                            delta_pixels = np.zeros(2, dtype=int)
                            path = None
                        
                        # Skip flow/repulsion/slide logic - A* handles everything
                        if astar_path is None or len(astar_path) <= 1:
                            plan_x = np.array([cur_pose[0], next_pose[0]])
                            plan_y = np.array([cur_pose[1], next_pose[1]])
                        
                    else:
                        # ========== FLOW LOCAL PLANNER (production mode) ==========
                        # Execute Flow
                        flow_tensor, flow_goal_tensor = build_flow_input_tensor(flow_obs_patch, flow_mean_patch, flow_var_patch, goal_vec_np)
                        
                        # --- OSCILLATION DETECTION (DISABLED for pure flow testing) ---
                        oscillation_override = False  # Manually disabled

                        if oscillation_override:
                            # FORCE DIRECT MOVEMENT (P-Controller)
                            delta = (goal_vec_np * DELTA_SCALE).astype(np.float32)
                            oscillation_override_counter -= 1
                            if oscillation_override_counter <= 0:
                                oscillation_override = False
                                print("Oscillation Override Deactivated.")
                        elif locked_trajectory is not None:
                            # USE LOCKED TRAJECTORY (computed once per cell transition)
                            # The locked_trajectory is already the delta to follow
                            delta = locked_trajectory.copy()
                            # sampled_trajectories already set when trajectory was locked
                        else:
                            # Fallback: compute flow per-step (shouldn't happen often)
                            delta = sample_flow_delta(flow_model, flow_tensor, flow_goal_tensor, flow_device, num_steps=flow_num_steps)
                        
                        if delta is None:
                            raise ValueError("Flow planner could not compute delta")
                    
                        # --- Safety Rails (Repulsion + Slide) for FLOW mode only ---
                        repulsive_force = np.zeros(2, dtype=np.float32)
                        
                        # Only use repulsion if NOT pure_flow mode
                        should_use_repulsion = (not collect_opts.pure_flow)

                        if should_use_repulsion and flow_obs_patch is not None:
                            window_r = 10
                            center_r = flow_obs_patch.shape[0] // 2
                            center_c = flow_obs_patch.shape[1] // 2
                            local_window = flow_obs_patch[center_r-window_r:center_r+window_r+1, center_c-window_r:center_c+window_r+1]
                            obs_rows, obs_cols = np.where(local_window == 1.0)
                            if len(obs_rows) > 0:
                                vec_r = window_r - obs_rows
                                vec_c = window_r - obs_cols
                                dists_sq = vec_r**2 + vec_c**2
                                dists_sq = np.maximum(dists_sq, 0.1)
                                rep_r = np.sum(vec_r / dists_sq)
                                rep_c = np.sum(vec_c / dists_sq)
                                repulsive_force = np.array([rep_r, rep_c])
                                rep_mag = np.linalg.norm(repulsive_force)
                                if rep_mag > 1e-3:
                                    repulsive_force = repulsive_force / rep_mag

                        alpha = 0.5
                        delta_norm = delta / (np.linalg.norm(delta) + 1e-6)
                        combined_delta = delta_norm + alpha * repulsive_force
                        combined_mag = np.linalg.norm(combined_delta)
                        if combined_mag > 1e-3:
                            step_size = float(ind_to_move_per_step)
                            delta_pixels = (combined_delta / combined_mag * step_size).astype(int)
                        else:
                            delta_pixels = np.zeros(2, dtype=int)
                    
                        # Collision Check (Slide/Squeeze)
                        def check_path_validity(start, delta, grid_gt, grid_inflated, check_inflated=True):
                            steps = max(abs(delta[0]), abs(delta[1]))
                            curr = start.copy()
                            if steps == 0: return True, curr
                            for i in range(1, steps + 1):
                                alpha_i = i / steps
                                cand = start + np.round(delta * alpha_i).astype(int)
                                if not (0 <= cand[0] < grid_gt.shape[0] and 0 <= cand[1] < grid_gt.shape[1]): return False, curr
                                if grid_gt[cand[0], cand[1]] == 1: return False, curr
                                if check_inflated:
                                    if grid_inflated[cand[0], cand[1]] == np.inf:
                                        if grid_inflated[start[0], start[1]] != np.inf: return False, curr
                                curr = cand
                            return True, curr

                        def try_rotations(start, delta, grid_gt, grid_inflated, check_inflated):
                            angles = [0, np.pi/12, -np.pi/12, np.pi/6, -np.pi/6, np.pi/4, -np.pi/4, np.pi/3, -np.pi/3, np.pi/2, -np.pi/2]
                            for ang in angles:
                                if ang == 0: alt_delta = delta
                                else:
                                    c, s = np.cos(ang), np.sin(ang)
                                    R = np.array([[c, -s], [s, c]])
                                    alt_delta = np.rint(R @ delta.astype(float)).astype(int)
                                if np.array_equal(alt_delta, np.zeros(2)): continue
                                v, p = check_path_validity(start, alt_delta, grid_gt, grid_inflated, check_inflated)
                                if v: return True, p
                            return False, start

                        success, end_pose = try_rotations(cur_pose, delta_pixels, mapper.gt_map, occ_grid_pyastar, check_inflated=True)
                        if not success:
                            success, end_pose = try_rotations(cur_pose, delta_pixels, mapper.gt_map, occ_grid_pyastar, check_inflated=False)

                        next_pose = end_pose
                        plan_x = np.array([cur_pose[0], next_pose[0]])
                        plan_y = np.array([cur_pose[1], next_pose[1]])

                else:
                    raise ValueError("Invalid local planner: {}".format(chosen_local_planner))

                if flow_logging_enabled and (t % getattr(flow_logging_cfg, "save_every", 1) == 0):
                    # Compute Goal Vector (Direction to locked frontier)
                    goal_vector = np.zeros(2, dtype=np.float32)
                    if locked_frontier_center is not None:
                        diff = locked_frontier_center - cur_pose
                        dist = np.linalg.norm(diff)
                        if dist > 1e-3:
                            goal_vector = (diff / dist).astype(np.float32)
                    
                    flow_sample = {
                        "timestamp": int(t),
                        "cur_pose": np.array(cur_pose).astype(np.int32),
                        "next_pose": np.array(next_pose).astype(np.int32),
                        "delta": (np.array(next_pose) - np.array(cur_pose)).astype(np.int32),
                        "goal_vector": goal_vector, # New Input Feature
                        "locked_frontier": np.array(locked_frontier_center).astype(np.int32) if locked_frontier_center is not None else None,
                        "frontier_region_centers": np.array(frontier_region_centers).astype(np.int32) if frontier_region_centers is not None else None,
                        "path_prefix": path[:min(25, len(path))].copy() if chosen_local_planner == 'astar' and path is not None else None,  # Save up to 25 waypoints for trajectory training
                        "obs_patch": flow_obs_patch.copy() if flow_obs_patch is not None else None,
                        "pred_mean_patch": flow_mean_patch.copy() if flow_mean_patch is not None else None,
                        "pred_var_patch": flow_var_patch.copy() if flow_var_patch is not None else None,
                        "mode": mode,
                        "mission_failed": mission_failed,
                    }
                    flow_episode_samples.append(flow_sample)
                
                if collect_opts.log_iou: # Toggle: Do real-time iou check
                    _, _, _, _, lama_pred_alltrain_viz_t = \
                            get_lama_pred_from_obs(mapper.obs_map.copy(), lama_alltrain_model, lama_map_transform, device)

                    iou = calculate_iou_kth(lama_pred_alltrain_viz_t[500:-500, 500:-500, :], 
                                            padded_gt_map[500:-500, 500:-500], 
                                            show_plt=False, 
                                            file_name=file_name)

                    # Log every 20 timesteps
                    if t % 20 == 0:
                        log_file.write(f"Time step is {t} and IoU is {iou}\n")
                        log_file.flush()
                    
                    if not iou_90_flag and iou > 0.8999:
                        log_file.write(f"IoU 90% reached : Time step is {t} and IoU is {iou}\n")
                        log_file.flush()
                        iou_90_flag = True 

                    if not iou_95_flag and iou > 0.9499:
                        log_file.write(f"IoU 95% reached : Time step is {t} and IoU is {iou}\n")
                        log_file.flush()
                        iou_95_flag = True 
                    

                    if iou_95_flag and t > (time_step // 3):
                        update_mission_status(start_time=start_exp_time, cur_step=t, mission_complete=True,
                                            fail_reason="", mission_status_save_path=mission_status_save_path)
                        print("\033[94mMission complete for {}!\033[0m".format(exp_title))
                        break
     

                # Visualize
                start_time = time.time()
                if show_plt: 
                    for a in ax.flatten():
                        a.clear()

                    # make a general map kwarg that is gray cmap
                    map_kwargs = {
                        'cmap': 'gray',
                        'vmin': 0,
                        'vmax': 1,
                    }
                    ax_gt.imshow(1-mapper.gt_map[pd_size:-pd_size,pd_size:-pd_size], **map_kwargs)
                    if pose_list is not None:
                        ax_gt.plot(pose_list[:, 1]-pd_size, pose_list[:, 0]-pd_size, c='r', alpha=0.5)
                        ax_gt.scatter(pose_list[-1, 1]-pd_size, pose_list[-1, 0]-pd_size, c='g', s=10, marker='*')
                    if mode != 'upen':
                        # if frontier cost list is all zeros, make it red
                        if frontier_cost_list is not None and np.all(frontier_cost_list == 0):
                            frontier_colors = 'r'
                        else:
                            frontier_colors = -frontier_cost_list
                            
                        if mode == 'pipe' or mode == 'pw-nbv-2d':
                            # Check the length of frontier_colors to avoid accessing out-of-bounds elements
                            if len(frontier_colors) > 0:
                                # Plot the best frontier with size 20 and its color
                                ax_gt.scatter(locked_frontier_center[1] - pd_size,
                                            locked_frontier_center[0] - pd_size,
                                            c='r', s=20, marker='x', cmap='plasma')

                            if len(frontier_colors) > 1:
                                # Plot the second-best frontier with size 10 and its color (only if available)
                                ax_gt.scatter(np.array(frontier_region_centers)[medium_ind, 1] - pd_size,
                                            np.array(frontier_region_centers)[medium_ind, 0] - pd_size,
                                            c='b', s=10, marker='x', cmap='plasma')

                            # Plot the remaining frontiers with size 5 and a neutral color (e.g., gray)
                            remaining_indices = [i for i in range(len(frontier_region_centers)) if i not in [best_ind, medium_ind]]

                            if len(remaining_indices) > 0:
                                ax_gt.scatter(np.array(frontier_region_centers)[remaining_indices, 1] - pd_size,
                                            np.array(frontier_region_centers)[remaining_indices, 0] - pd_size,
                                            c='gray', s=5, marker='x')
                        else:
                            ax_gt.scatter(np.array(frontier_region_centers)[:, 1]-pd_size, np.array(frontier_region_centers)[:, 0]-pd_size, c=frontier_colors, s=10, marker='x',cmap='plasma')
                    ax_gt.set_title('GT Map')
                    # print("3a. Visualizing GT Map took {} seconds".format(np.round(time.time() - start_time, 2)))

                    #colors_ = ["#68e371", "#D9D9D9", "#f0432b"]
                    colors_ = ["#FFFFFF", "#D9D9D9", "#000000"]
                    cmap = ListedColormap(colors_)
                    ax_obs.imshow(mapper.obs_map[pd_size:-pd_size,pd_size:-pd_size], cmap = cmap)#, **map_kwargs)

                    

                    if pose_list is not None:
                        ax_obs.plot(pose_list[:, 1]-pd_size, pose_list[:, 0]-pd_size, c='#eb4205', alpha=1.0)
                    if mode not in ['upen', 'hector', 'hectoraug'] and locked_frontier_center is not None: # UPEN and Hector do not have locked frontiers
                        ax_obs.scatter(locked_frontier_center[1]-pd_size, locked_frontier_center[0]-pd_size, c='#eb4205', s=10)
                    if mode not in ['hector', 'hectoraug']: # Hector does not have path planning
                        ax_obs.plot(plan_y-pd_size, plan_x-pd_size,c='#eb4205', linestyle=':')

                    if mode not in ['upen', 'hector', 'hectoraug']: # UPEN and Hector are not a frontier planner
                        if viz_medium_flooded_grid is not None:
                            second_most_flooded_grid = viz_medium_flooded_grid[pd_size:-pd_size,pd_size:-pd_size]
                            second_flooded_ind = np.where(second_most_flooded_grid==True)
                            #ax_obs.scatter(second_flooded_ind[1]-pad_w1,second_flooded_ind[0]-pad_h1,c='c',s=1,alpha=0.05)

                        if viz_most_flooded_grid is not None:
                            most_flooded_grid = viz_most_flooded_grid#[pd_size:-pd_size,pd_size:-pd_size]
                            flooded_ind = np.where(most_flooded_grid==True)
                            flooded_ind_colors_alpha = np.zeros((mapper.obs_map.shape[0],mapper.obs_map.shape[1],4))
                            flooded_ind_colors_alpha[flooded_ind[0],flooded_ind[1],:] = (255/255,159/255,28/255,0.3)    # orange
                            #flooded_ind_colors_alpha[flooded_ind[0],flooded_ind[1],:] = (207/255,3/255,252/255,0.25)   # purple                            
                            ###ax_obs.scatter(flooded_ind[1]-pad_w1, flooded_ind[0]-pad_h1,c="#FF9F1C",s=1,alpha=0.05)
                            if mode in ['nbv-2d', 'pw-nbv-2d']:  # Show flooded grid for nbv-2d and pw-nbv-2d modes
                                ax_obs.imshow(flooded_ind_colors_alpha[pd_size+pad_h1:-(pd_size+pad_h2),pd_size+pad_w1:-(pd_size+pad_w2)])

                    if mode == 'nbh':
                        # Visualize Gateway and Target Cell
                        if 'target_cell' in locals() and target_cell is not None:
                            # Target Cell Center (Blue Circle)
                            tc_r, tc_c = target_cell.center
                            ax_gt.scatter(tc_c - pd_size, tc_r - pd_size, c='blue', s=30, marker='o', label='Cell Center')
                            ax_obs.scatter(tc_c - pd_size, tc_r - pd_size, c='blue', s=30, marker='o')

                    # Visualize Flow Vector
                    if mode == 'nbh' and 'delta_pixels' in locals():
                        # Visualize Sensor Range (Instant View)
                        # We call get_instant_obs_at_pose just for viz
                        try:
                            instant_obs = mapper.get_instant_obs_at_pose(cur_pose)
                            vis_ind = instant_obs['vis_ind']
                            
                            # Theoretical Range Mask (Circle) - read from mapper config
                            range_pix = mapper.lidar_sim_configs['laser_range_m'] * mapper.lidar_sim_configs['pixel_per_meter']
                            y_grid, x_grid = np.ogrid[:mapper.obs_map.shape[0], :mapper.obs_map.shape[1]]
                            dist_from_pose = np.sqrt((x_grid - cur_pose[1])**2 + (y_grid - cur_pose[0])**2)
                            range_mask = dist_from_pose <= range_pix
                            
                            # Visible Mask (intersection of lidar visibility AND within range)
                            vis_mask = np.zeros_like(mapper.obs_map, dtype=bool)
                            vis_mask[vis_ind[:, 0], vis_ind[:, 1]] = True
                            
                            # Visible within range (correctly filtered)
                            visible_in_range = vis_mask & range_mask
                            
                            # Occluded Mask = Range Mask AND NOT Visible in Range
                            occluded_mask = range_mask & (~visible_in_range)
                            
                            # Get indices for plotting
                            vis_rows, vis_cols = np.where(visible_in_range)
                            occ_rows, occ_cols = np.where(occluded_mask)
                            
                            # Plot Visible within range (Cyan)
                            ax_obs.scatter(vis_cols - pd_size, vis_rows - pd_size, c='cyan', s=1, alpha=0.05, label='Visible Area')
                            
                            # Plot Occluded (Orange) - within range but blocked
                            ax_obs.scatter(occ_cols - pd_size, occ_rows - pd_size, c='orange', s=1, alpha=0.05, label='Occluded Area')
                            
                        except Exception:
                            pass

                        if not USE_ASTAR_FOR_LOCAL:
                            # Scale up for visibility
                            viz_scale = 10 
                            
                            # 1. Plot Raw Flow (Cyan) - Where the brain wants to go
                            if 'delta_norm' in locals():
                                d_raw = delta_norm * 3.0 # Scale to match step size
                                ax_obs.arrow(cur_pose[1]-pd_size, cur_pose[0]-pd_size, 
                                             d_raw[1]*viz_scale, d_raw[0]*viz_scale, 
                                             color='cyan', head_width=3, label='Flow Vec')

                            # 2. Plot Repulsion (Yellow) - Where walls push
                            if 'repulsive_force' in locals() and np.linalg.norm(repulsive_force) > 0.1:
                                d_rep = repulsive_force * 3.0
                                ax_obs.arrow(cur_pose[1]-pd_size, cur_pose[0]-pd_size, 
                                             d_rep[1]*viz_scale, d_rep[0]*viz_scale, 
                                             color='yellow', head_width=3, label='Repulsion')

                            # 3. Plot Final Result (Red/Orange) - Actual movement
                            d_row, d_col = delta_pixels[0] * viz_scale, delta_pixels[1] * viz_scale
                            ax_obs.arrow(cur_pose[1]-pd_size, cur_pose[0]-pd_size, d_col, d_row, color='red', head_width=5, label='Combined Vel')
                            
                            # Add Legend (Flow mode)
                            from matplotlib.lines import Line2D
                            legend_elements = [
                                Line2D([0], [0], marker='o', color='w', label='Target Cell', markerfacecolor='blue', markersize=8),
                                Line2D([0], [0], color='cyan', lw=2, label='Flow Vec'),
                                Line2D([0], [0], color='yellow', lw=2, label='Repulsion'),
                                Line2D([0], [0], color='red', lw=2, label='Combined Vel'),
                                Line2D([0], [0], marker='o', color='w', label='Visible', markerfacecolor='cyan', markersize=5, alpha=0.5),
                                Line2D([0], [0], marker='o', color='w', label='Occluded', markerfacecolor='orange', markersize=5, alpha=0.5),
                            ]
                            ax_obs.legend(handles=legend_elements, loc='upper right', fontsize='small')
                        else:
                            # Add Legend (A* debug mode)
                            from matplotlib.lines import Line2D
                            legend_elements = [
                                Line2D([0], [0], marker='o', color='w', label='Target Cell', markerfacecolor='blue', markersize=8),
                                Line2D([0], [0], color='#eb4205', lw=2, linestyle=':', label='A* Path'),
                                Line2D([0], [0], marker='o', color='w', label='Visible', markerfacecolor='cyan', markersize=5, alpha=0.5),
                                Line2D([0], [0], marker='o', color='w', label='Occluded', markerfacecolor='orange', markersize=5, alpha=0.5),
                            ]
                            ax_obs.legend(handles=legend_elements, loc='upper right', fontsize='small')
                        
                    ax_obs.set_title('Observed Map')

                    if pred_maputils is not None and mean_map is not None:
                        white = "#FFFFFF"
                        blue = "#0000FF"
                        orange = "#FF9F1C"
                        if mode == 'pipe':
                            colors = [white, orange]
                        else:
                            colors = [white, blue]
                        n_bins = 100
                        cmap = LinearSegmentedColormap.from_list("customgreenred", colors, N=n_bins)
                        ax_mean_map.imshow(mean_map[pd_size:-(pd_size),pd_size:-(pd_size)],cmap=cmap) #predicted map

                        #overlay observed(known) occupied cells on top of the predicted map
                        obs_occ_mask = np.zeros_like(mean_map[pd_size:-(pd_size),pd_size:-(pd_size)])
                        occupied_indices_in_obsmap = np.where(mapper.obs_map[pd_size-pad_h1:-(pd_size-pad_h2),pd_size-pad_w1:-(pd_size-pad_w2)] == 1.0) #indices where obs_map is occupied
                        obs_occ_mask[occupied_indices_in_obsmap] = 1
                        obs_occ_mask_colors = ["#000000","#000000"]
                        obs_occ_mask_cmap = LinearSegmentedColormap.from_list("mask_black",obs_occ_mask_colors,N=2)
                        obs_occ_mask_alpha = np.zeros_like(obs_occ_mask, dtype=float)
                        obs_occ_mask_alpha[obs_occ_mask==1] = 1.0
                        obs_occ_mask_alpha[obs_occ_mask==0] = 0.0
                        ax_mean_map.imshow(obs_occ_mask, cmap=obs_occ_mask_cmap, alpha=obs_occ_mask_alpha) #obs_map known occ cells -> black

                        #overlay unknown(obs_map) cells as gray tint
                        obs_unk_mask = np.zeros_like(mean_map[pd_size:-pd_size,pd_size:-pd_size])
                        unknown_indices_in_obs_map = np.where(mapper.obs_map[pd_size-pad_h1:-(pd_size-pad_h2),pd_size-pad_w1:-(pd_size-pad_w2)] == 0.5) #indices of unknown cells in obs_map
                        obs_unk_mask[unknown_indices_in_obs_map] = 1
                        grey = "#909090" #tunable grey value
                        obs_unk_mask_colors = [grey,grey]
                        obs_unk_mask_cmap = LinearSegmentedColormap.from_list("mask_grey", obs_unk_mask_colors, N=2)
                        obs_unk_mask_alpha = np.zeros_like(obs_unk_mask, dtype=float)
                        obs_unk_mask_alpha[obs_unk_mask==1] = 0.3 #tunable opacity for grey unknown area
                        obs_unk_mask_alpha[obs_unk_mask==0] = 0.0
                        ax_mean_map.imshow(obs_unk_mask, cmap=obs_unk_mask_cmap, alpha=obs_unk_mask_alpha)
                                
                        #path_color = "#417CF2" #blue
                        path_color = "#eb4205" #coral(red)
                        if pose_list is not None:
                            ax_mean_map.plot(pose_list[:, 1]-(pd_size-pad_w1), pose_list[:, 0]-(pd_size-pad_h1), c=path_color, alpha=1.0)
                            #ax_mean_map.scatter(pose_list[-1, 1]-(pd_size-pad_w1), pose_list[-1, 0]-(pd_size-pad_h1), c='g', s=10, marker='*')
                        if mode not in ['upen', 'hector', 'hectoraug'] and locked_frontier_center is not None: # UPEN and Hector do not have locked frontiers
                            ax_mean_map.scatter(locked_frontier_center[1]-(pd_size-pad_w1), locked_frontier_center[0]-(pd_size-pad_h1), c='#eb4205', s=10)
                        #ax_mean_map.scatter(cur_pose[1]-(pd_size-pad_w1), cur_pose[0]-(pd_size-pad_h1), c='r', s=5, marker='x')
                        #ax_mean_map.scatter(next_pose[1]-(pd_size-pad_w1), next_pose[0]-(pd_size-pad_h1), c='g', s=5, marker='x')
                        if mode not in ['hector', 'hectoraug']: # Hector does not have path planning
                            ax_mean_map.plot(plan_y-(pd_size-pad_w1), plan_x-(pd_size-pad_h1), c='#eb4205', linestyle=':')
                        ax_mean_map.set_title('Mean Map of Prediction Ensembles')

                        #visualize frontiers on the mean_map
                        #if frontier_cost_list is not None and np.all(frontier_cost_list == 0):
                        #    frontier_colors = 'r'
                        #else:
                        #    frontier_colors = -frontier_cost_list
                        #ax_mean_map.scatter(np.array(frontier_region_centers)[:, 1]-(pd_size-pad_w1), np.array(frontier_region_centers)[:, 0]-(pd_size-pad_h1), c=frontier_colors, s=15, marker='x',cmap='plasma')

                        if mode != 'upen':
                            if viz_most_flooded_grid is not None:
                                most_flooded_grid = viz_most_flooded_grid#[pd_size:-pd_size,pd_size:-pd_size]
                                flooded_ind = np.where(most_flooded_grid==True)
                                flooded_ind_colors_alpha = np.zeros((mean_map.shape[0],mean_map.shape[1],4))
                                if mode == 'pipe':
                                    flooded_ind_colors_alpha[flooded_ind[0],flooded_ind[1],:] = (0, 0, 1, 0.15) #blue color for visibility mask
                                else:
                                    flooded_ind_colors_alpha[flooded_ind[0],flooded_ind[1],:] = (255/255,159/255,28/255,0.3) #orange color for visibility mask
                                ax_mean_map.imshow(flooded_ind_colors_alpha[pd_size:-pd_size,pd_size:-pd_size])
                            if viz_medium_flooded_grid is not None:
                                medium_flooded_ind = np.where(viz_medium_flooded_grid == True)
                                medium_flooded_ind_colors_alpha = np.zeros((mean_map.shape[0],mean_map.shape[1],4))
                                #medium_flooded_ind_colors_alpha[medium_flooded_ind[0], medium_flooded_ind[1],:] = (154/255,230/255,72/255,0.4)
                                medium_flooded_ind_colors_alpha[medium_flooded_ind[0], medium_flooded_ind[1],:] = (207/255,3/255,252/255,0.25)
                                #ax_mean_map.imshow(medium_flooded_ind_colors_alpha[pd_size:-pd_size,pd_size:-pd_size])
                                #if medium_ind is not None:
                                #    ax_mean_map.scatter(frontier_region_centers[medium_ind,1]-(pd_size-pad_w1), frontier_region_centers[medium_ind,0]-(pd_size-pad_h1), c='#390ccc',s=10)

                    if mode in ['hector', 'hectoraug']:
                        ax_mean_map.imshow(cost_transform_map[pd_size:-pd_size,pd_size:-pd_size], cmap='gray')
                        ax_mean_map.set_title('Cost Transform Map')
                    # # Display the flood fill
                    # if viz_min_flooded_grid is not None:
                    #     ax_min_flooded_grid.imshow(viz_min_flooded_grid)
                    #     ax_min_flooded_grid.set_title('Min Frontier Val')
                    # if viz_most_flooded_grid is not None:
                    #     ax_max_flooded_grid.imshow(viz_most_flooded_grid)
                    #     ax_max_flooded_grid.set_title('Most Frontier Val')

                    # --- NEW: Cell Graph Visualization ---
                    if mode == 'nbh' and 'ax_cell_graph' in dir() and cell_manager is not None:
                        ax_cell_graph.clear()
                        visualize_cell_graph(
                            ax_cell_graph, cell_manager, padded_obs_map, mean_map,
                            current_pose=cur_pose, target_cell=target_cell, 
                            path_to_target=path_to_target, pd_size=pd_size,
                            show_cell_boundaries=True
                        )

                    plt.tight_layout()
                    # plt.show()
                    # plt.pause(0.001)
                    
                    # fig name include experiment title and time step
                    print("saving fig:", t)
                    plt.savefig(run_viz_dir + '/{}_{}.png'.format(exp_title, str(t).zfill(8)),dpi=300)
                    # print("3d. Visualizing took {} seconds".format(np.round(time.time() - start_time, 2)))
                    # import pdb; pdb.set_trace() 
                    # print("4. Visualizing took {} seconds".format(np.round(time.time() - start_time, 2)))
                    #plt.close()

                    # Save detailed graph visualization with scores (baseline-style)
                    if mode == 'nbh' and cell_manager is not None and show_plt:
                        fig_graph, ax_graph = plt.subplots(1, 1, figsize=(10, 10))
                        ax_graph.imshow(1-mapper.gt_map[pd_size:-pd_size,pd_size:-pd_size], cmap='gray', vmin=0, vmax=1)
                        visualize_cell_graph(
                            ax_graph, cell_manager, padded_gt_map, mean_map,
                            current_pose=cur_pose, target_cell=target_cell,
                            path_to_target=path_to_target, pd_size=pd_size,
                            overlay_mode=True,
                            show_scores=True,
                            start_cell=None,
                            astar_path=astar_path if 'astar_path' in locals() else None,
                            show_cell_boundaries=True
                        )
                        ax_graph.scatter(cur_pose[1] - pd_size, cur_pose[0] - pd_size, c='lime', s=30, marker='*',
                                         zorder=20, edgecolors='black', linewidths=0.5)
                        scores = [n.propagated_value for n in cell_manager.cells.values() if n.propagated_value > 0]
                        min_score = min(scores) if scores else 0.0
                        max_score = max(scores) if scores else 0.0
                        ax_graph.set_title(f'Detailed Cell Graph (Scores: {min_score:.2f}-{max_score:.2f})')
                        plt.savefig(graph_map_dir + f'/graph_{str(t).zfill(8)}.png', dpi=300)
                        plt.close(fig_graph)

                    
                    show_plt = False

                # Save obs map and pose list 
                # Save obs map 
                cv2.imwrite(global_obs_dir + '/{}.png'.format(str(t).zfill(8)), smu.convert_01_single_channel_to_0_255_3_channel(mapper.obs_map))
                # Save pose list
                np.save(odom_npy_save_path, pose_list)
                
                # # Go to next pose
                # start_time = time.time()
                cur_pose = next_pose
                if mapper.gt_map[cur_pose[0], cur_pose[1]] == 1:
                    print("Hit wall!")
                    mission_failed = True
                    update_mission_status(start_time=start_exp_time, cur_step=t, mission_complete=False, fail_reason="hit_wall", mission_status_save_path=mission_status_save_path)
                    break
                pose_list = np.concatenate([pose_list, np.atleast_2d(cur_pose)], axis=0)
                
                # Update recent_poses for Oscillation Detection
                recent_poses.append(cur_pose)
                
                # # Observation: Get instant observation and accumulate
                mapper.observe_and_accumulate_given_pose(cur_pose)
                # print("5. Accumulating obs took {} seconds".format(np.round(time.time() - start_time, 2)))

                # Save current mission status 
                update_mission_status(start_time=start_exp_time, cur_step=t, mission_complete=False, fail_reason="", mission_status_save_path=mission_status_save_path)
                # if (t % 99) == 0:
                #print("Total time for step {} is {} seconds".format(t, np.round(time.time() - start_mission_i_time, 2)))
                t += 1

            #TODO:  Save final mission status 
            if mission_failed:
                # Don't update mission status if it's already failed
                print("\033[91mMission failed for {}!\033[0m".format(exp_title))

            else:
                update_mission_status(start_time=start_exp_time, cur_step=t, mission_complete=True, fail_reason="", mission_status_save_path=mission_status_save_path)
                print("\033[94mMission complete for {}!\033[0m".format(exp_title))

            if flow_logging_enabled and flow_episode_samples:
                flow_save_name = f"{exp_title}_flow_samples.npz"
                flow_save_path = os.path.join(flow_logging_dir, flow_save_name)
                np.savez_compressed(
                    flow_save_path,
                    samples=np.array(flow_episode_samples, dtype=object),
                    meta=flow_meta
                )
                print(f"Saved {len(flow_episode_samples)} flow samples to {flow_save_path}")
    except Exception as e:
        print("\033[93mMission failed with exception for {}!\033[0m".format(exp_title))
        print(e)
        # print the exception and line number 
        print(traceback.format_exc())
        if mission_status_save_path:
            update_mission_status(start_time=start_exp_time, cur_step=t, mission_complete=False, fail_reason=str(e), mission_status_save_path=mission_status_save_path)
    
def determine_use_model(mode):
    """Determine if LAMA prediction model should be used for the given mode."""
    if mode == 'nbh':
        return True
    else:
        raise ValueError(f"Unsupported mode: {mode}. Only 'nbh' mode is supported.")


if __name__ == '__main__':
    data_collect_config_name = 'base.yaml' #customize yaml file, as needed
    today = datetime.today()
    output_subdirectory_name = str(today.year)+"{:02d}".format(today.month)+"{:02d}".format(today.day)+'_test'
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect_world_list', nargs='+', help='List of worlds to collect data from')
    parser.add_argument('--start_pose', nargs='+', help='List of start pose')
    parser.add_argument('--mode', help='Override modes_to_test with a single mode')
    parser.add_argument('--flow_checkpoint', type=str, help='Checkpoint path for flow model')
    parser.add_argument('--flow_device', type=str, default='cuda', help='Device for flow model (e.g., cuda or cpu)')
    parser.add_argument('--flow_steps', type=int, default=1, help='Number of integration steps for flow inference')
    parser.add_argument('--pure_flow', action='store_true', help='Disable all safety rails (repulsion, A* recovery) to test raw model performance')
    parser.add_argument('--local_planner', type=str, choices=['astar', 'flow'], help='Local planner: astar (debug) or flow (production)')
    parser.add_argument('--cell_size', type=int, help='Cell size in pixels (default: 25 = 2.5m)')
    args = parser.parse_args()
    
    # Auto-detect root path (nbh repo root is parent of nbh/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.dirname(script_dir)
    
    collect_opts = get_options_dict_from_yml(data_collect_config_name)
    collect_opts.root_path = root_path  # Set root_path dynamically
    if args.collect_world_list is not None:
        collect_opts.collect_world_list = args.collect_world_list
    if args.start_pose is not None: 
        start_pose = []
        for pose_elem in args.start_pose:
            start_pose.append(int(pose_elem))
        collect_opts.start_pose = start_pose
    if args.mode is not None:
        collect_opts.modes_to_test = [args.mode]
    flow_checkpoint = args.flow_checkpoint
    if flow_checkpoint is None and hasattr(collect_opts, "flow_checkpoint"):
        flow_checkpoint = collect_opts.flow_checkpoint
    
    # If still None, try to find best.pt automatically
    if flow_checkpoint is None:
        default_best_path = os.path.join(root_path, "checkpoints", "best.pt")
        if os.path.exists(default_best_path):
            flow_checkpoint = default_best_path
            print(f"No checkpoint specified, using best found at: {flow_checkpoint}")
            
    collect_opts.flow_checkpoint = flow_checkpoint
    collect_opts.flow_device = args.flow_device
    collect_opts.flow_steps = args.flow_steps
    collect_opts.pure_flow = args.pure_flow
    collect_opts.flow_crop_radius = getattr(collect_opts.flow_logging, "crop_radius", 128)
    
    # Local planner mode: CLI > yaml > default ('astar')
    if args.local_planner is not None:
        collect_opts.local_planner = args.local_planner
    elif not hasattr(collect_opts, 'local_planner'):
        collect_opts.local_planner = 'astar'
    
    # Cell size: CLI > yaml > default (25)
    if args.cell_size is not None:
        collect_opts.cell_size = args.cell_size
    elif not hasattr(collect_opts, 'cell_size'):
        collect_opts.cell_size = 25

    kth_map_folder_path = os.path.join(root_path, 'kth_test_maps')
    print(f"Looking for maps in: {kth_map_folder_path}")
    kth_map_paths = os.listdir(kth_map_folder_path)

    if collect_opts.collect_world_list is not None:
        print(f"Filtering for worlds: {collect_opts.collect_world_list}")
        kth_map_paths_collect = []
        for folder_name in kth_map_paths:
            if folder_name in collect_opts.collect_world_list:
                kth_map_paths_collect.append(folder_name)
        kth_map_paths = kth_map_paths_collect
    
    print(f"Found {len(kth_map_paths)} maps to run.")

    kth_map_folder_paths = [os.path.join(kth_map_folder_path, p) for p in kth_map_paths] * collect_opts.num_data_per_world

    # Make output_subdirectory_name if it doesn't exist 
    output_root_dir = os.path.join(root_path, collect_opts.output_folder_name, output_subdirectory_name)
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)
    
    model_list = []
    device = collect_opts.lama_device

    #ensemble of lama models (G_i; G_1, G_2, G_3 in the paper) - fine-tuned with split, smaller training sets
    if collect_opts.ensemble_folder_name is not None:
        ensemble_folder_name = collect_opts.ensemble_folder_name
        ensemble_folder_path = os.path.join(root_path, 'pretrained_models', ensemble_folder_name)
        ensemble_model_dirs = sorted(os.listdir(ensemble_folder_path))
        for ensemble_model_dir in ensemble_model_dirs:
            ensemble_model_path = os.path.join(ensemble_folder_path, ensemble_model_dir)
            model = load_lama_model(ensemble_model_path, device=collect_opts.lama_device)
            print("Loaded model: ", ensemble_model_dir)
            model_list.append(model)
    
    #setup a big lama model (G in the paper) - fine-tuned with the entire training set
    big_lama_path = os.path.join(root_path, 'pretrained_models', collect_opts.big_lama_model_folder_name)
    lama_model = load_lama_model(big_lama_path, device=collect_opts.lama_device)
    lama_map_transform = get_lama_transform(collect_opts.lama_transform_variant, collect_opts.lama_out_size)

    flow_model = None
    flow_runtime = None
    flow_device = torch.device(collect_opts.flow_device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {flow_device}")
    if collect_opts.flow_checkpoint is not None:
        try:
            flow_model = load_flow_model(collect_opts.flow_checkpoint, flow_device)
            flow_runtime = {
                "model": flow_model,
                "device": flow_device,
                "num_steps": max(1, collect_opts.flow_steps),
                "crop_radius": collect_opts.flow_crop_radius
            }
            print(f"Loaded flow model from {collect_opts.flow_checkpoint} on {flow_device}")
        except Exception as flow_exc:
            print(f"Failed to load flow model: {flow_exc}")
            flow_runtime = None

    run_exploration_args = []
    for kth_map_folder_path in kth_map_folder_paths:
        args_dict = {
            'map_folder_path': kth_map_folder_path,
            'models_list': model_list,
            'lama_model': lama_model,
            'lama_map_transform': lama_map_transform,
            'pred_vis_configs': collect_opts.pred_vis_configs,
            'lidar_sim_configs': collect_opts.lidar_sim_configs,
            'start_pose': collect_opts.start_pose,
            'modes_to_test': collect_opts.modes_to_test,
            'unknown_as_occ': collect_opts.unknown_as_occ,
            'use_distance_transform_for_planning': collect_opts.use_distance_transform_for_planning,
            'upen_config': collect_opts.upen_config,
            'flow_runtime': flow_runtime
        }
        run_exploration_args.append(args_dict)
    
    for run_exploration_arg in run_exploration_args:
        run_exploration_comparison_for_map(run_exploration_arg)
