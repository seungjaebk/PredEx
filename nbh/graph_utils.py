import numpy as np
import cv2
from collections import deque

DEBUG_GRAPH = False

class CellNode:
    def __init__(self, index, center_pixel, is_ghost=False):
        self.index = index # (row, col) tuple
        self.center = center_pixel # (x, y) numpy array [row, col]
        self.visited = False
        self.visit_count = 0
        self.parent = None # For backtracking
        self.is_blocked = False
        self.is_ghost = is_ghost
        
        # Graph properties
        self.neighbors = [] # List of neighbor indices
        
        # Diffusion Values
        self.base_value = 0.0 # Intrinsic value (Uncertainty at this spot)
        self.propagated_value = 0.0 # Value flowed from neighbors

    def __repr__(self):
        type_str = "Ghost" if self.is_ghost else "Real"
        return f"[{type_str}]Cell{self.index}(Val:{self.propagated_value:.2f})"

class CellManager:
    def __init__(
        self,
        cell_size=50,
        start_pose=None,
        valid_space_map=None,
        promotion_cfg=None,
        connectivity_cfg=None,
        debug_cfg=None,
    ):
        """
        Robot-centric cell manager for unknown map exploration.
        
        Args:
            cell_size: Size of each cell in pixels
            start_pose: Robot's starting position [row, col] - becomes centroid of cell (0, 0)
            valid_space_map: Optional valid space map (for reference only, not used for ghost creation)
        """
        self.cell_size = cell_size
        self.free_mask = None
        self.last_obs_shape = None
        self.promotion_cfg = promotion_cfg or {}
        self.connectivity_cfg = connectivity_cfg or {}
        self.debug_cfg = debug_cfg or {}
        
        # Robot's start position is the centroid of cell (0, 0)
        # This will be set on first update_graph call if not provided
        self.origin = np.array(start_pose, dtype=float) if start_pose is not None else None
        
        self.cells = {}  # Dict: (r, c) -> CellNode, indices can be NEGATIVE!
        self.current_cell = None
        self.visited_stack = []
        
        self.valid_space_map = valid_space_map

        # Visualization (dynamic, updated as needed)
        self.scent_map_vis = None  # Will be created dynamically

    def _get_cfg(self, key, default):
        return self.promotion_cfg.get(key, default)

    def _patch_mean(self, patch, default=0.0):
        if patch is None or patch.size == 0:
            return default
        return float(np.mean(patch))

    def get_cell_index(self, pixel_pose):
        """
        Convert pixel (row, col) to cell index (r, c) relative to origin.
        
        Cell (0, 0) is centered at self.origin (robot's start position).
        Indices can be NEGATIVE if robot moves "backwards" from start.
        """
        if self.origin is None:
            raise ValueError("Origin not set! Call update_graph first or provide start_pose in __init__")
        
        offset = np.array(pixel_pose) - self.origin
        # Add half cell so origin is at CENTER of cell (0, 0), not corner
        r = int(np.floor((offset[0] + self.cell_size / 2) / self.cell_size))
        c = int(np.floor((offset[1] + self.cell_size / 2) / self.cell_size))
        return (r, c)  # Can be negative!

    def get_cell_center(self, cell_idx):
        """
        Get absolute pixel coordinates of cell centroid.
        
        Args:
            cell_idx: (r, c) tuple, can be negative
            
        Returns:
            np.array([row, col]) in absolute pixel coordinates
        """
        if self.origin is None:
            raise ValueError("Origin not set!")
        r, c = cell_idx
        center_r = self.origin[0] + r * self.cell_size
        center_c = self.origin[1] + c * self.cell_size
        return np.array([center_r, center_c])

    def get_cell(self, index, create_if_missing=True, is_ghost=False):
        """Get or create a cell node."""
        if index not in self.cells:
            if not create_if_missing:
                return None
            center = self.get_cell_center(index)
            self.cells[index] = CellNode(index, center, is_ghost=is_ghost)
        return self.cells[index]

    def update_graph_structure(self, obs_map, pred_mean_map, pred_var_map=None, max_ghost_distance=2, free_mask=None):
        """
        Update cell graph based on current observations and predictions.
        
        Robot-centric: Expands from current cells, no fixed grid bounds.
        Cell indices can be negative!
        
        Args:
            obs_map: Observed map (0=free, 0.5=unknown, 1=occupied) in absolute pixel coords
            pred_mean_map: Predicted mean occupancy
            pred_var_map: Predicted variance (IMPORTANT for uncertainty!)
            max_ghost_distance: Max cell distance from observed cell for ghost expansion
        """
        map_h, map_w = obs_map.shape
        self.free_mask = free_mask
        max_ghost_distance = self._get_cfg("graph_max_ghost_distance", max_ghost_distance)
        obs_blocked_ratio = self._get_cfg("graph_obs_blocked_ratio", 0.3)
        unknown_ratio_threshold = self._get_cfg("graph_unknown_ratio_threshold", 0.5)
        centroid_blocked_threshold = self._get_cfg("graph_centroid_blocked_threshold", 0.8)
        pred_mean_free_threshold = self._get_cfg("graph_ghost_pred_mean_free_threshold", 0.4)
        pred_var_max_threshold = self._get_cfg("graph_ghost_pred_var_max_threshold", 0.3)
        
        # --- 1. UPDATE REAL NODES (Observed) ---
        # Expand from existing cells + their neighbors
        
        active_indices = list(self.cells.keys())
        
        # Get indices of REAL (observed) cells only
        real_cell_indices = set()
        for idx, cell in self.cells.items():
            if not cell.is_ghost and not cell.is_blocked:
                real_cell_indices.add(idx)
        
        # Expand only from REAL cells (not ghosts!) with distance limit
        # No bounds checking on indices - they can be negative!
        potential_indices = set(active_indices)
        frontier = list(real_cell_indices)
        distances = {idx: 0 for idx in real_cell_indices}
        
        for distance in range(1, max_ghost_distance + 1):
            new_frontier = []
            for idx in frontier:
                r, c = idx
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nr, nc = r+dr, c+dc
                    # No bounds check on cell indices! But we check pixel bounds when accessing map
                    if (nr, nc) not in distances:
                        distances[(nr, nc)] = distance
                        potential_indices.add((nr, nc))
                        new_frontier.append((nr, nc))
            frontier = new_frontier
        
        for idx in potential_indices:
            # Get absolute pixel coordinates for this cell
            cell_center = self.get_cell_center(idx)
            half = self.cell_size // 2
            
            # Calculate pixel bounds (in absolute coordinates)
            y_start = int(cell_center[0] - half)
            y_end = int(cell_center[0] + half)
            x_start = int(cell_center[1] - half)
            x_end = int(cell_center[1] + half)
            
            # Clip to map bounds (obs_map has finite size)
            y_start_clip = max(0, y_start)
            y_end_clip = min(map_h, y_end)
            x_start_clip = max(0, x_start)
            x_end_clip = min(map_w, x_end)
            
            # Skip if cell is completely outside map
            if y_start_clip >= y_end_clip or x_start_clip >= x_end_clip:
                continue
            
            # NOTE: We do NOT use valid_space_map here because it contains GT info!
            # Ghost cells should only be based on obs_map (observed) and pred_mean_map (predicted)
            # The robot shouldn't "know" about invalid space before observing it.
            
            # Use clipped bounds for map access
            patch_obs = obs_map[y_start_clip:y_end_clip, x_start_clip:x_end_clip]
            patch_pred = pred_mean_map[y_start_clip:y_end_clip, x_start_clip:x_end_clip] if pred_mean_map is not None else None
            
            if patch_obs.size == 0: continue

            # Centroid lookup (absolute pixel coords)
            centroid_r = int(cell_center[0])
            centroid_c = int(cell_center[1])
            if not (0 <= centroid_r < map_h and 0 <= centroid_c < map_w):
                continue

            centroid_obs_val = obs_map[centroid_r, centroid_c]

            # Check Obstacle status
            is_obs_blocked = np.mean(patch_obs == 1) > obs_blocked_ratio  # If >30% occupied
            if centroid_obs_val >= centroid_blocked_threshold:
                is_obs_blocked = True
            
            # Check Unknown status (patch-based)
            unknown_ratio = float(np.mean(patch_obs == 0.5))
            is_unknown = (unknown_ratio > unknown_ratio_threshold)
            
            # Check Prediction status (Ghost)
            # If unknown but Predicted Free, it's a Ghost Candidate
            pred_mean_patch = self._patch_mean(patch_pred, default=1.0)
            is_pred_free = pred_mean_patch < pred_mean_free_threshold

            # Prediction variance at centroid (for debug and gating)
            pred_var_center = None
            if pred_var_map is not None:
                var_h, var_w = pred_var_map.shape
                if 0 <= centroid_r < var_h and 0 <= centroid_c < var_w:
                    pred_var_center = float(pred_var_map[centroid_r, centroid_c])

            # Debug: inspect specific cells near start
            debug_indices = {(0, -1), (1, 0)}
            if idx in debug_indices:
                pred_mean_str = f"{pred_mean_patch:.3f}" if pred_mean_map is not None else "NA"
                pred_var_str = f"{pred_var_center:.3f}" if pred_var_center is not None else "NA"
                ghost_distance = distances.get(idx, float('inf'))
            if DEBUG_GRAPH:
                print(
                    f"[DEBUG][GHOST_CAND] idx={idx} "
                    f"unknown_ratio={unknown_ratio:.3f} centroid_obs={centroid_obs_val:.2f} "
                    f"pred_mean={pred_mean_str} pred_var={pred_var_str} "
                    f"is_obs_blocked={is_obs_blocked} ghost_distance={ghost_distance}"
                )
            
            # Node Creation / Update Logic
            if is_obs_blocked:
                # Mark blocked
                node = self.get_cell(idx)
                node.is_blocked = True
                node.base_value = 0.0
            elif not is_unknown:
                # Real Free Space
                node = self.get_cell(idx, is_ghost=False)
                node.is_blocked = False
                node.is_ghost = False
                node.base_value = 0.0 # Visited/Known = Boring (unless we add semantic bonus later)
            elif is_unknown and is_pred_free:
                # GHOST NODE - but only if uncertainty is LOW enough
                # High variance = uncertain = don't trust the prediction
                cell_variance = 1.0  # Default high (don't expand)
                var_patch = None
                if pred_var_map is not None:
                    var_patch = pred_var_map[y_start_clip:y_end_clip, x_start_clip:x_end_clip]
                cell_variance = self._patch_mean(var_patch, default=1.0)
                
                # Only create ghost if:
                # 1. Variance is below threshold (model is confident it's free)
                # 2. Distance from real cell is reasonable (still need some limit)
                ghost_distance = distances.get(idx, float('inf'))
                if cell_variance < pred_var_max_threshold and ghost_distance <= max_ghost_distance:
                    node = self.get_cell(idx, is_ghost=True)
                    node.is_blocked = False
                    node.base_value = cell_variance  # Store actual variance
            else:
                # Unknown and Predicted Blocked/Unknown -> Don't add to graph (Wall in Fog)
                pass

        # --- 2. CONNECT NEIGHBORS (with wall checking!) ---
        for idx, node in self.cells.items():
            if node.is_blocked: continue
            r, c = idx
            node.neighbors = []
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r+dr, c+dc
                if (nr, nc) in self.cells:
                    neighbor = self.cells[(nr, nc)]
                    if not neighbor.is_blocked:
                        involves_ghost = node.is_ghost or neighbor.is_ghost
                        if not involves_ghost and self.free_mask is not None:
                            # Real-real edges: use portal check on inflated free space
                            if self._check_portal_clear(idx, (nr, nc), self.free_mask) and self._check_path_clear(idx, (nr, nc), obs_map, pred_mean_map, involves_ghost=False):
                                node.neighbors.append(neighbor)
                        else:
                            # Ghost-involved edges: keep strict line-of-sight check
                            if self._check_path_clear(idx, (nr, nc), obs_map, pred_mean_map, involves_ghost):
                                node.neighbors.append(neighbor)
    
    def _check_path_clear(self, node_idx, neighbor_idx, obs_map, pred_mean_map=None, involves_ghost=False):
        """
        Check if robot can pass between two adjacent cells.
        
        SIMPLE STRICT CHECK: Block if ANY wall pixel on centroid line.
        Uses robot-centric cell centers (absolute pixel coordinates).
        
        Returns True if passage is possible, False if blocked.
        """
        # Get absolute pixel coordinates of cell centroids
        centroid1 = self.get_cell_center(node_idx)
        centroid2 = self.get_cell_center(neighbor_idx)
        
        # Sample points along the line between centroids
        num_samples = self.cell_size
        for i in range(num_samples + 1):
            t = i / num_samples
            r = int(centroid1[0] * (1 - t) + centroid2[0] * t)
            c = int(centroid1[1] * (1 - t) + centroid2[1] * t)
            
            # Check if point is within map bounds
            if 0 <= r < obs_map.shape[0] and 0 <= c < obs_map.shape[1]:
                obs_val = obs_map[r, c]
                
                # Block on observed wall
                if obs_val >= 0.8:
                    return False
                
                # For ghost cells, also check prediction
                if involves_ghost and pred_mean_map is not None:
                    if 0 <= r < pred_mean_map.shape[0] and 0 <= c < pred_mean_map.shape[1]:
                        pred_val = pred_mean_map[r, c]
                        if pred_val > 0.5:
                            return False
        
        # Centroid line is clear
        return True

    def _check_portal_clear(self, node_idx, neighbor_idx, free_mask, thickness=2):
        """
        Check if there exists a traversable "portal" across the shared boundary
        between two adjacent real cells using a thin strip of free space.
        """
        if free_mask is None:
            return False

        h, w = free_mask.shape
        thickness = max(1, int(thickness))

        a_center = self.get_cell_center(node_idx).astype(int)
        b_center = self.get_cell_center(neighbor_idx).astype(int)
        half = self.cell_size // 2

        a_r0 = max(0, int(a_center[0] - half))
        a_r1 = min(h, int(a_center[0] + half))
        a_c0 = max(0, int(a_center[1] - half))
        a_c1 = min(w, int(a_center[1] + half))

        b_r0 = max(0, int(b_center[0] - half))
        b_r1 = min(h, int(b_center[0] + half))
        b_c0 = max(0, int(b_center[1] - half))
        b_c1 = min(w, int(b_center[1] + half))

        dr = neighbor_idx[0] - node_idx[0]
        dc = neighbor_idx[1] - node_idx[1]

        if dr == 0 and dc == 1:
            # neighbor to the right
            a_strip = free_mask[a_r0:a_r1, max(a_c1 - thickness, a_c0):a_c1]
            b_strip = free_mask[b_r0:b_r1, b_c0:min(b_c0 + thickness, b_c1)]
            min_len = min(a_strip.shape[0], b_strip.shape[0])
            if min_len <= 0:
                return False
            a_strip = a_strip[:min_len, :]
            b_strip = b_strip[:min_len, :]
            return np.any(np.any(a_strip, axis=1) & np.any(b_strip, axis=1))

        if dr == 0 and dc == -1:
            # neighbor to the left
            a_strip = free_mask[a_r0:a_r1, a_c0:min(a_c0 + thickness, a_c1)]
            b_strip = free_mask[b_r0:b_r1, max(b_c1 - thickness, b_c0):b_c1]
            min_len = min(a_strip.shape[0], b_strip.shape[0])
            if min_len <= 0:
                return False
            a_strip = a_strip[:min_len, :]
            b_strip = b_strip[:min_len, :]
            return np.any(np.any(a_strip, axis=1) & np.any(b_strip, axis=1))

        if dr == 1 and dc == 0:
            # neighbor below
            a_strip = free_mask[max(a_r1 - thickness, a_r0):a_r1, a_c0:a_c1]
            b_strip = free_mask[b_r0:min(b_r0 + thickness, b_r1), b_c0:b_c1]
            min_len = min(a_strip.shape[1], b_strip.shape[1])
            if min_len <= 0:
                return False
            a_strip = a_strip[:, :min_len]
            b_strip = b_strip[:, :min_len]
            return np.any(np.any(a_strip, axis=0) & np.any(b_strip, axis=0))

        if dr == -1 and dc == 0:
            # neighbor above
            a_strip = free_mask[a_r0:min(a_r0 + thickness, a_r1), a_c0:a_c1]
            b_strip = free_mask[max(b_r1 - thickness, b_r0):b_r1, b_c0:b_c1]
            min_len = min(a_strip.shape[1], b_strip.shape[1])
            if min_len <= 0:
                return False
            a_strip = a_strip[:, :min_len]
            b_strip = b_strip[:, :min_len]
            return np.any(np.any(a_strip, axis=0) & np.any(b_strip, axis=0))

        return False

    def _align_pred_map(self, pred_map, obs_shape, fill_value=1.0):
        if pred_map is None or obs_shape is None:
            return pred_map

        pred_h, pred_w = pred_map.shape
        obs_h, obs_w = obs_shape
        if pred_h == obs_h and pred_w == obs_w:
            return pred_map

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

        aligned[dst_top:dst_top + copy_h, dst_left:dst_left + copy_w] = pred_map[
            src_top:src_top + copy_h, src_left:src_left + copy_w
        ]
        return aligned
    
    def diffuse_scent(self, pred_var_map):
        """
        Propagate values (Heat Diffusion) across the graph.
        """
        if pred_var_map is None:
            return
            
        map_h, map_w = pred_var_map.shape
        
        # 1. Assign Base Values (Heat Sources)
        # Ghosts get heat from Prediction Variance
        for node in self.cells.values():
            if node.is_blocked:
                node.base_value = 0.0
                node.propagated_value = 0.0
                continue
                
            if node.is_ghost:
                half = self.cell_size // 2
                r0 = max(0, int(node.center[0] - half))
                r1 = min(map_h, int(node.center[0] + half))
                c0 = max(0, int(node.center[1] - half))
                c1 = min(map_w, int(node.center[1] + half))
                patch = pred_var_map[r0:r1, c0:c1]
                node.base_value = self._patch_mean(patch, default=0.0)
            
            # Reset propagation to base
            node.propagated_value = node.base_value

        # 2. Iterative Diffusion (Bellman Update)
        # V_new = V_base + gamma * max(Neighbors)
        # Using 'max' (Gradient) instead of 'mean' (Diffusion) creates a stronger path.
        gamma = self._get_cfg("graph_diffuse_gamma", 0.95)  # Decay factor
        iterations = int(self._get_cfg("graph_diffuse_iterations", 50))  # Enough to cover the map
        
        for _ in range(iterations):
            # Create copy of values to update synchronously (or async is fine too)
            updates = {}
            for idx, node in self.cells.items():
                if node.is_blocked: continue
                
                # Find best neighbor value
                max_neighbor_val = 0.0
                for neighbor in node.neighbors:
                    if neighbor.propagated_value > max_neighbor_val:
                        max_neighbor_val = neighbor.propagated_value
                
                # Update: My value is my intrinsic worth + decayed promise of neighbors
                updates[idx] = node.base_value + (gamma * max_neighbor_val)
            
            # Apply updates
            for idx, val in updates.items():
                self.cells[idx].propagated_value = val
        
        # Note: scent_map_vis is no longer used (indices can be negative now)

    def update_graph(self, robot_pose, obs_map, pred_mean_map=None, pred_var_map=None, inflated_occ_grid=None):
        """Main update loop called by agent."""
        # 0. Set origin on first call (robot's start = cell (0,0) centroid)
        if self.origin is None:
            self.origin = np.array(robot_pose, dtype=float)
            print(f"[GRAPH] Origin set to robot start: {self.origin}")
        self.last_obs_shape = obs_map.shape
        
        # 1. Update structure (Real + Ghost nodes)
        free_mask = None
        if inflated_occ_grid is not None:
            free_mask = np.isfinite(inflated_occ_grid)
        self.update_graph_structure(obs_map, pred_mean_map, pred_var_map=pred_var_map, free_mask=free_mask)
        
        # 2. Update current position
        curr_idx = self.get_cell_index(robot_pose)
        curr_node = self.get_cell(curr_idx)  # Ensure current node exists
        
        # Update parent/stack for backtracking
        if self.current_cell is not None and self.current_cell.index != curr_idx:
            if curr_node.parent is None and curr_node != self.current_cell.parent:
                curr_node.parent = self.current_cell
                self.visited_stack.append(self.current_cell)
            elif curr_node == self.current_cell.parent:
                if self.visited_stack: self.visited_stack.pop()
        
        self.current_cell = curr_node
        if not curr_node.visited:
            curr_node.visited = True
            curr_node.visit_count = 1
        else:
            curr_node.visit_count += 1

        if pred_var_map is not None and self._get_cfg("graph_diffuse_on_update", True):
            self.diffuse_scent(pred_var_map)
        
        # DEBUG: Print current cell info
        if DEBUG_GRAPH:
            print(f"[DEBUG] Current cell: idx={curr_node.index}, centroid={curr_node.center}, "
                  f"obs_map.shape={obs_map.shape}, robot_pose={robot_pose}")

    def get_best_target_cell(self, pred_var_map, obs_map):
        """Select next cell based on Diffused Scent."""
        # 1. Run Diffusion
        self.diffuse_scent(pred_var_map)
        
        if self.current_cell is None: return None, "INIT"
        
        # 2. Score Neighbors
        best_score = -1e9
        best_node = None
        
        # Look at graph neighbors
        valid_moves = []
        
        for neighbor in self.current_cell.neighbors:
            if neighbor.is_blocked: continue
            
            # --- CENTER VALIDITY CHECK ---
            # A cell might not be "blocked" overall, but its CENTER could be on/near a wall.
            # Check if the center area is actually in free space.
            center_r, center_c = int(neighbor.center[0]), int(neighbor.center[1])
            
            # Check a small patch around center for walls (more strict near doors)
            buffer = 3  # Check 7x7 area (±3 pixels)
            r_min = max(0, center_r - buffer)
            r_max = min(obs_map.shape[0], center_r + buffer + 1)
            c_min = max(0, center_c - buffer)
            c_max = min(obs_map.shape[1], center_c + buffer + 1)
            
            center_patch = obs_map[r_min:r_max, c_min:c_max]
            if center_patch.size > 0:
                wall_ratio = np.mean(center_patch == 1.0)
                if wall_ratio > 0.15:  # stricter: skip centers near walls
                    # print(f"Skipping cell {neighbor.index}: center near wall ({wall_ratio*100:.0f}%)")
                    continue
            
            # Score = Propagated Scent - Visit Penalty
            # We want to follow the high values
            score = neighbor.propagated_value
            
            # Penalty for re-visiting (Anti-Loop)
            if neighbor.visited:
                # If it's just a transit node, penalty is small.
                # If it's a dead end we stuck in, penalty grows.
                score /= (1.0 + 0.5 * neighbor.visit_count)
            
            valid_moves.append((score, neighbor))
            
            if score > best_score:
                best_score = score
                best_node = neighbor
        
        # 3. Decision
        # If best option is terrible (low value), Backtrack
        # Threshold depends on scale of variance. 
        # If base_value is ~100 (sum of variance), propagated is high.
        # If 0, then 0.
        
        if best_node is not None and best_score > 1.0:
            return best_node, f"FLOW(Val={best_score:.1f})"
        else:
            # Backtrack
            if self.current_cell.parent:
                return self.current_cell.parent, "BACKTRACK"
            elif self.visited_stack:
                return self.visited_stack[-1], "STACK_JUMP"
            else:
                # Stuck
                return None, "STUCK"

    # ========================================================================
    # HYBRID NAVIGATION: Goal Gradient + Local Greedy
    # ========================================================================
    
    def propagate_goal_scent(self, target_cell, decay=0.9, iterations=50):
        """
        Propagate goal gradient from target cell through the graph.
        Called ONCE when target changes (O(cells × iterations)).
        
        Result: Each cell gets a goal_scent value proportional to 
        how "reachable" the target is from that cell via the graph.
        """
        if target_cell is None:
            return
        
        # Initialize all cells to 0 goal_scent
        for cell in self.cells.values():
            cell.goal_scent = 0.0
        
        # Target cell gets max value
        target_cell.goal_scent = 1.0
        
        # Bellman-like propagation from target
        for _ in range(iterations):
            updates = {}
            for idx, cell in self.cells.items():
                if cell.is_blocked:
                    continue
                
                # Find best neighbor's goal_scent
                max_neighbor_scent = 0.0
                for neighbor in cell.neighbors:
                    if neighbor.goal_scent > max_neighbor_scent:
                        max_neighbor_scent = neighbor.goal_scent
                
                # My goal_scent = my intrinsic (0 unless target) + decayed neighbor
                if cell.index == target_cell.index:
                    updates[idx] = 1.0  # Target stays at 1.0
                else:
                    updates[idx] = decay * max_neighbor_scent
            
            # Apply updates
            for idx, val in updates.items():
                self.cells[idx].goal_scent = val
    
    def pick_best_neighbor(self, target_cell, obs_map, alpha=0.01):
        """
        Local greedy selection: O(4) per step.
        
        Score = propagated_value (uncertainty) + alpha × goal_scent (tie-breaker)
        
        Args:
            target_cell: The high-level target cell
            obs_map: Current observation map (for center validity check)
            alpha: Weight for goal_scent tie-breaker (small value)
        
        Returns:
            best_neighbor: Next cell to navigate toward
            reason: String describing decision
        """
        if self.current_cell is None:
            return None, "NO_CURRENT_CELL"
        
        best_score = -1e9
        best_neighbor = None
        
        for neighbor in self.current_cell.neighbors:
            if neighbor.is_blocked:
                continue
            
            # --- CENTER VALIDITY CHECK ---
            center_r, center_c = int(neighbor.center[0]), int(neighbor.center[1])
            buffer = 3
            r_min = max(0, center_r - buffer)
            r_max = min(obs_map.shape[0], center_r + buffer + 1)
            c_min = max(0, center_c - buffer)
            c_max = min(obs_map.shape[1], center_c + buffer + 1)
            
            center_patch = obs_map[r_min:r_max, c_min:c_max]
            if center_patch.size > 0:
                wall_ratio = np.mean(center_patch == 1.0)
                if wall_ratio > 0.15:
                    continue
            
            # PRIMARY: Uncertainty (propagated_value from diffuse_scent)
            score = neighbor.propagated_value
            
            # TIE-BREAKER: Goal direction (goal_scent from propagate_goal_scent)
            goal_scent = getattr(neighbor, 'goal_scent', 0.0)
            score += alpha * goal_scent * 100  # Scale to be meaningful tie-breaker
            
            # Visit penalty (anti-loop)
            if neighbor.visited:
                score /= (1.0 + 0.5 * neighbor.visit_count)
            
            if score > best_score:
                best_score = score
                best_neighbor = neighbor
        
        if best_neighbor is not None:
            return best_neighbor, f"LOCAL(score={best_score:.2f})"
        else:
            # Fallback: backtrack
            if self.current_cell.parent:
                return self.current_cell.parent, "BACKTRACK"
            elif self.visited_stack:
                return self.visited_stack[-1], "STACK_JUMP"
            else:
                return None, "STUCK"
    
    def get_next_path_cell(self, path_to_target):
        """
        Get the next cell from the BFS-computed path.
        
        This ensures the robot follows the planned path to target,
        rather than relying on greedy neighbor selection.
        
        Args:
            path_to_target: List of CellNodes from current to target
            
        Returns:
            next_cell: Next cell in path (or None if path invalid)
            reason: Description string
        """
        if self.current_cell is None:
            return None, "NO_CURRENT_CELL"
        
        if path_to_target is None or len(path_to_target) < 2:
            return None, "NO_PATH"
        
        # Find current cell in path
        for i, cell in enumerate(path_to_target):
            if cell.index == self.current_cell.index:
                # Return next cell in path
                if i + 1 < len(path_to_target):
                    return path_to_target[i + 1], f"PATH_FOLLOW({i+1}/{len(path_to_target)})"
                else:
                    return None, "AT_TARGET"
        
        # Current cell not in path - path may be stale
        return None, "OFF_PATH"
    
    def find_exploration_target(self, pred_var_map, current_cell=None):
        """
        Find the best exploration target (highest uncertainty/value cell).
        Called when current target is reached or invalid.
        
        Considers ALL cells (both ghost and real) based on their propagated_value.
        A real cell may be selected if it leads to unexplored areas with high uncertainty.
        
        IMPORTANT: Only considers cells that are REACHABLE from current_cell
        (i.e., in the same connected component of the graph).
        
        Args:
            pred_var_map: Prediction variance map for diffusion
            current_cell: Current cell the robot is in (required for reachability check)
        
        Returns:
            target_cell: Best reachable cell to explore (ghost or real), or None if none found
        """
        best_cell = None
        best_value = -1e9
        
        # Run diffusion first to get propagated values
        pred_var_map = self._align_pred_map(pred_var_map, self.last_obs_shape)
        self.diffuse_scent(pred_var_map)
        
        # Find all cells reachable from current_cell using BFS
        reachable_cells = set()
        if current_cell is not None:
            from collections import deque
            queue = deque([current_cell])
            visited = {current_cell.index}
            
            while queue:
                cell = queue.popleft()
                reachable_cells.add(cell.index)
                for neighbor in cell.neighbors:
                    if neighbor.index not in visited and not neighbor.is_blocked:
                        visited.add(neighbor.index)
                        queue.append(neighbor)
        
        # Find best cell (ghost OR real) among REACHABLE cells based on propagated_value
        for cell in self.cells.values():
            if cell.is_blocked:
                continue
            
            # Skip if cell is not reachable from current position
            if current_cell is not None and cell.index not in reachable_cells:
                continue
            
            # Skip current cell itself
            if current_cell is not None and cell.index == current_cell.index:
                continue
                
            # Select cell with highest propagated value (uncertainty flow)
            if cell.propagated_value > best_value:
                best_value = cell.propagated_value
                best_cell = cell
        
        if best_cell is None and current_cell is not None:
            print(f"[HIGH-LEVEL] No reachable target cells! Reachable: {len(reachable_cells)} cells")
        elif best_cell is not None:
            cell_type = "Ghost" if best_cell.is_ghost else "Real"
            print(f"[HIGH-LEVEL] Selected {cell_type} cell {best_cell.index} with value {best_value:.4f}")
        
        return best_cell

    def find_path_to_target(self, start_cell, target_cell):
        """
        Find path from start_cell to target_cell using BFS.
        
        Returns:
            path: list of CellNodes from start to target (inclusive)
                  or None if no path found
        """
        if start_cell is None or target_cell is None:
            print(f"  [PATH] Cannot find path: start={start_cell}, target={target_cell}")
            return None
        
        if start_cell.index == target_cell.index:
            return [start_cell]
        
        from collections import deque
        
        # BFS
        queue = deque([(start_cell, [start_cell])])
        visited = {start_cell.index}
        
        while queue:
            current, path = queue.popleft()
            
            for neighbor in current.neighbors:
                if neighbor.index in visited:
                    continue
                if neighbor.is_blocked:
                    continue
                
                new_path = path + [neighbor]
                
                if neighbor.index == target_cell.index:
                    # Path found! Print the path
                    path_str = " → ".join([f"{c.index}" for c in new_path])
                    print(f"  [PATH] Found: {path_str} ({len(new_path)} cells)")
                    return new_path
                
                visited.add(neighbor.index)
                queue.append((neighbor, new_path))
        
        # No path found - debug why
        print(f"  [PATH] NO PATH from {start_cell.index} to {target_cell.index}!")
        print(f"         Start neighbors: {[n.index for n in start_cell.neighbors]}")
        print(f"         Target neighbors: {[n.index for n in target_cell.neighbors]}")
        print(f"         Visited {len(visited)} cells, target not reachable")
        return None  # No path found
