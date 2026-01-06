import os
import numpy as np
import cv2
import pyastar2d
from collections import deque

DEBUG_GRAPH = False

def parse_debug_edge_samples(raw_samples):
    if not raw_samples:
        return set()
    samples = set()
    for entry in raw_samples:
        if not entry or len(entry) != 2:
            continue
        a, b = entry
        if a is None or b is None or len(a) != 2 or len(b) != 2:
            continue
        key = tuple(sorted([tuple(a), tuple(b)]))
        samples.add(key)
    return samples

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
        
        # Origin retained for backward compatibility; absolute indexing is used.
        self.origin = np.array(start_pose, dtype=float) if start_pose is not None else None
        
        self.cells = {}  # Dict: (r, c) -> CellNode, indices can be NEGATIVE!
        self.current_cell = None
        self.visited_stack = []
        
        self.valid_space_map = valid_space_map

        # Visualization (dynamic, updated as needed)
        self.scent_map_vis = None  # Will be created dynamically

        self._mini_astar_cache = {}
        self._rr_edge_cache = {}
        self._mini_astar_step = 0
        self._obs_update_id = 0
        self._pred_update_id = 0
        self._grid_update_id = 0
        self._last_obs_map = None
        self._last_pred_map = None
        self._last_inflated_sig = None
        self.edge_costs = {}
        self._p_occ_map = None
        self._dijkstra_cache = None  # (start_idx, dist, prev)
        self.debug_edge_samples = set()
        self.los_viz_dir = None
        self._los_viz_counter = 0

    def _get_cfg(self, key, default):
        return self.promotion_cfg.get(key, default)

    def _patch_mean(self, patch, default=0.0):
        if patch is None or patch.size == 0:
            return default
        return float(np.mean(patch))

    def _map_change_ratio(self, prev, curr):
        if prev is None:
            return 1.0
        if prev.shape != curr.shape:
            return 1.0
        return float(np.mean(prev != curr))

    def _inflated_signature(self, grid):
        if grid is None:
            return None
        finite = np.isfinite(grid)
        return (grid.shape, int(np.count_nonzero(finite)))

    def _grid_policy_hash(self):
        return hash((
            self.connectivity_cfg.get("graph_unknown_as_occ", True),
            self.connectivity_cfg.get("graph_pred_wall_threshold", 0.7),
            self.connectivity_cfg.get("graph_pred_hard_wall_threshold", 0.95),
            self.connectivity_cfg.get("graph_dilate_diam", 3),
            self._obs_update_id,
            self._pred_update_id,
            self._grid_update_id,
        ))

    def _edge_key(self, node_idx, neighbor_idx):
        return tuple(sorted([node_idx, neighbor_idx]))

    def _collect_line_samples(self, node_idx, neighbor_idx, obs_map):
        centroid1 = self.get_cell_center(node_idx)
        centroid2 = self.get_cell_center(neighbor_idx)
        num_samples = self.cell_size
        samples = []

        for i in range(num_samples + 1):
            t = i / num_samples
            r = int(centroid1[0] * (1 - t) + centroid2[0] * t)
            c = int(centroid1[1] * (1 - t) + centroid2[1] * t)
            if 0 <= r < obs_map.shape[0] and 0 <= c < obs_map.shape[1]:
                samples.append((r, c, float(obs_map[r, c])))
            else:
                samples.append((r, c, None))
        return samples

    def _trace_los_line(self, centroid1, centroid2, obs_map, offset_r=0, offset_c=0):
        num_samples = self.cell_size
        points = []
        has_sample = False
        blocked = False

        for i in range(num_samples + 1):
            t = i / num_samples
            r = int(centroid1[0] * (1 - t) + centroid2[0] * t) + offset_r
            c = int(centroid1[1] * (1 - t) + centroid2[1] * t) + offset_c
            if 0 <= r < obs_map.shape[0] and 0 <= c < obs_map.shape[1]:
                has_sample = True
                val = float(obs_map[r, c])
                points.append((r, c, val))
                if val >= 0.8:
                    blocked = True

        return points, (has_sample and not blocked)

    def _samples_blocked(self, samples, obs_map):
        if not samples:
            return True
        h, w = obs_map.shape
        for r, c in samples:
            if not (0 <= r < h and 0 <= c < w):
                return True
            if obs_map[r, c] >= 0.8:
                return True
        return False

    def _get_clear_los_samples(self, node_idx, neighbor_idx, obs_map):
        centroid1 = self.get_cell_center(node_idx)
        centroid2 = self.get_cell_center(neighbor_idx)

        points, clear = self._trace_los_line(centroid1, centroid2, obs_map)
        if clear:
            return [(r, c) for r, c, _ in points]

        dr = neighbor_idx[0] - node_idx[0]
        dc = neighbor_idx[1] - node_idx[1]
        if dr == 0 and dc != 0:
            offsets = [(-1, 0), (1, 0)]
        elif dc == 0 and dr != 0:
            offsets = [(0, -1), (0, 1)]
        else:
            return []

        for off_r, off_c in offsets:
            points, clear = self._trace_los_line(
                centroid1, centroid2, obs_map, offset_r=off_r, offset_c=off_c
            )
            if clear:
                return [(r, c) for r, c, _ in points]
        return []

    def _save_los_debug_plot(self, node_idx, neighbor_idx, obs_map, line_results):
        if not self.los_viz_dir:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return

        centroid1 = self.get_cell_center(node_idx)
        centroid2 = self.get_cell_center(neighbor_idx)
        rows = [centroid1[0], centroid2[0]]
        cols = [centroid1[1], centroid2[1]]
        for line in line_results:
            for r, c, _ in line["points"]:
                rows.append(r)
                cols.append(c)

        margin = int(self.cell_size * 2)
        r_min = max(int(min(rows) - margin), 0)
        r_max = min(int(max(rows) + margin), obs_map.shape[0] - 1)
        c_min = max(int(min(cols) - margin), 0)
        c_max = min(int(max(cols) + margin), obs_map.shape[1] - 1)

        crop = obs_map[r_min:r_max + 1, c_min:c_max + 1]
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(crop, cmap="gray", vmin=0, vmax=1)

        for line in line_results:
            points = line["points"]
            if not points:
                continue
            color = "blue" if line["clear"] else "red"
            xs = [p[1] - c_min for p in points]
            ys = [p[0] - r_min for p in points]
            ax.plot(xs, ys, color=color, linewidth=2)

        ax.scatter(
            [centroid1[1] - c_min, centroid2[1] - c_min],
            [centroid1[0] - r_min, centroid2[0] - r_min],
            c="cyan",
            s=30,
            marker="o",
            edgecolors="black",
            linewidths=0.5,
        )
        ax.set_title(f"LOS {node_idx} <-> {neighbor_idx}")
        ax.axis("off")

        filename = (
            f"los_{node_idx[0]}_{node_idx[1]}_"
            f"{neighbor_idx[0]}_{neighbor_idx[1]}_"
            f"{self._los_viz_counter:04d}.png"
        )
        fig.savefig(os.path.join(self.los_viz_dir, filename), dpi=200)
        plt.close(fig)
        self._los_viz_counter += 1

    def get_cell_index(self, pixel_pose):
        """
        Convert pixel (row, col) to absolute cell index (r, c).

        Cell (0, 0) corresponds to the top-left of the map.
        """
        r = int(np.floor(pixel_pose[0] / self.cell_size))
        c = int(np.floor(pixel_pose[1] / self.cell_size))
        return (r, c)

    def get_cell_center(self, cell_idx):
        """
        Get absolute pixel coordinates of cell centroid.
        
        Args:
            cell_idx: (r, c) tuple in absolute grid coordinates
            
        Returns:
            np.array([row, col]) in absolute pixel coordinates
        """
        r, c = cell_idx
        center_r = (r + 0.5) * self.cell_size
        center_c = (c + 0.5) * self.cell_size
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

        self._p_occ_map = self._compute_p_occ(obs_map, pred_mean_map, pred_var_map)
        self.edge_costs = {}

        potential_count = 0
        processed_count = 0
        skipped_outside = 0
        
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
            potential_count += 1
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
                skipped_outside += 1
                continue
            processed_count += 1
            
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
        edge_stats = {
            "RR": {"attempt": 0, "ok": 0},
            "RG": {"attempt": 0, "ok": 0},
            "GG": {"attempt": 0, "ok": 0},
        }

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
                        if not involves_ghost:
                            edge_type = "RR"
                            ok, _info = self._check_rr_edge(
                                idx, (nr, nc), obs_map, pred_mean_map, self.free_mask
                            )
                        else:
                            edge_type = "RG" if (node.is_ghost != neighbor.is_ghost) else "GG"
                            ok, _info = self._check_ghost_edge(
                                idx, (nr, nc), obs_map, pred_mean_map, edge_type
                            )

                        edge_stats[edge_type]["attempt"] += 1
                        if ok:
                            edge_stats[edge_type]["ok"] += 1

                        self._log_edge_debug(
                            edge_type,
                            _info["portal"],
                            _info["los"],
                            _info["mini"],
                            _info["cache"],
                            _info["ttl_left"],
                        )
                        if ok:
                            node.neighbors.append(neighbor)
                            edge_key = self._edge_key(idx, (nr, nc))
                            if edge_key not in self.edge_costs:
                                self.edge_costs[edge_key] = self._edge_risk_cost(
                                    idx, (nr, nc), self._p_occ_map
                                )

        if self.debug_cfg.get("graph_debug_risk_stats", False):
            total_cells = len(self.cells)
            real_cells = sum(
                1 for cell in self.cells.values() if (not cell.is_ghost and not cell.is_blocked)
            )
            ghost_cells = sum(
                1 for cell in self.cells.values() if (cell.is_ghost and not cell.is_blocked)
            )
            blocked_cells = sum(1 for cell in self.cells.values() if cell.is_blocked)
            if total_cells:
                rows = [idx[0] for idx in self.cells.keys()]
                cols = [idx[1] for idx in self.cells.keys()]
                idx_range = f"r=[{min(rows)},{max(rows)}], c=[{min(cols)},{max(cols)}]"
            else:
                idx_range = "n/a"

            print(
                "[GRAPH][STATS] "
                f"cells total={total_cells} real={real_cells} ghost={ghost_cells} blocked={blocked_cells} "
                f"idx_range={idx_range}"
            )
            print(
                "[GRAPH][STATS] "
                f"potential={potential_count} processed={processed_count} skipped_outside={skipped_outside} "
                f"max_ghost_distance={max_ghost_distance}"
            )
            print(
                "[GRAPH][EDGES] "
                f"RR {edge_stats['RR']['ok']}/{edge_stats['RR']['attempt']} "
                f"RG {edge_stats['RG']['ok']}/{edge_stats['RG']['attempt']} "
                f"GG {edge_stats['GG']['ok']}/{edge_stats['GG']['attempt']}"
            )

            if self._p_occ_map is not None:
                p_occ = self._p_occ_map
                print(
                    "[GRAPH][POCC] "
                    f"min={float(np.min(p_occ)):.3f} mean={float(np.mean(p_occ)):.3f} "
                    f"max={float(np.max(p_occ)):.3f}"
                )
                unknown_ratio = float(np.mean(obs_map == 0.5))
                print(f"[GRAPH][POCC] unknown_ratio={unknown_ratio:.3f}")

            if self.edge_costs:
                edge_vals = np.fromiter(self.edge_costs.values(), dtype=float)
                zero_ratio = float(np.mean(edge_vals <= 1e-6))
                print(
                    "[GRAPH][COST] "
                    f"min={float(np.min(edge_vals)):.6f} mean={float(np.mean(edge_vals)):.6f} "
                    f"max={float(np.max(edge_vals)):.6f} zero_ratio={zero_ratio:.3f}"
                )
                stats = self._edge_strip_stats(pred_mean_map, self._p_occ_map)
                pred_vals = stats["pred_max"]
                if pred_vals.size:
                    hard_thr = float(
                        self.connectivity_cfg.get("graph_pred_hard_wall_threshold", 0.95)
                    )
                    hard_ratio = float(np.mean(pred_vals >= hard_thr))
                    print(
                        "[GRAPH][EDGE_PRED_MAX] "
                        f"min={float(np.min(pred_vals)):.3f} mean={float(np.mean(pred_vals)):.3f} "
                        f"max={float(np.max(pred_vals)):.3f} hard_ratio={hard_ratio:.3f}"
                    )
                pocc_vals = stats["pocc_max"]
                if pocc_vals.size:
                    print(
                        "[GRAPH][EDGE_POCC_MAX] "
                        f"min={float(np.min(pocc_vals)):.3f} mean={float(np.mean(pocc_vals)):.3f} "
                        f"max={float(np.max(pocc_vals)):.3f}"
                    )
    
    def _build_cost_grid(self, obs_map, pred_mean_map=None, edge_type="RR"):
        unknown_as_occ = self.connectivity_cfg.get("graph_unknown_as_occ", True)
        pred_wall_threshold = self.connectivity_cfg.get("graph_pred_wall_threshold", 0.7)

        cost = np.ones_like(obs_map, dtype=np.float32)
        cost[obs_map >= 0.8] = np.inf
        if unknown_as_occ:
            cost[obs_map == 0.5] = np.inf

        if edge_type in ("RG", "GG") and pred_mean_map is not None:
            cost[pred_mean_map > pred_wall_threshold] = np.inf
        return cost

    def _run_mini_astar(self, node_idx, neighbor_idx, cost_grid, edge_type):
        cache_ttl = self.connectivity_cfg.get("graph_mini_astar_ttl_steps", 10)
        policy_hash = self._grid_policy_hash()
        cache_key = (node_idx, neighbor_idx, edge_type, policy_hash)

        entry = self._mini_astar_cache.get(cache_key)
        if entry is not None:
            ok, path_len, reason, last_step = entry
            ttl_left = cache_ttl - (self._mini_astar_step - last_step)
            if ttl_left > 0:
                return ok, path_len, reason, "hit", ttl_left

        start = tuple(self.get_cell_center(node_idx).astype(int))
        goal = tuple(self.get_cell_center(neighbor_idx).astype(int))
        path = pyastar2d.astar_path(cost_grid, start, goal, allow_diagonal=False)
        if path is None:
            result = (False, 0, "no_path")
        else:
            result = (True, int(path.shape[0]), "ok")

        self._mini_astar_cache[cache_key] = (*result, self._mini_astar_step)
        return result[0], result[1], result[2], "miss", cache_ttl

    def _run_mini_astar_local_rr(self, node_idx, neighbor_idx, obs_map):
        cache_ttl = self.connectivity_cfg.get("graph_mini_astar_ttl_steps", 10)
        policy_hash = self._grid_policy_hash()
        cache_key = (node_idx, neighbor_idx, "RR", "local", policy_hash)

        entry = self._mini_astar_cache.get(cache_key)
        if entry is not None:
            ok, path_len, reason, last_step = entry
            ttl_left = cache_ttl - (self._mini_astar_step - last_step)
            if ttl_left > 0:
                return ok, path_len, reason, "hit", ttl_left, None

        c1 = self.get_cell_center(node_idx)
        c2 = self.get_cell_center(neighbor_idx)

        half_cell = self.cell_size // 2
        half_patch = 2 * half_cell
        min_r = int(min(c1[0], c2[0]) - half_patch)
        max_r = int(max(c1[0], c2[0]) + half_patch)
        min_c = int(min(c1[1], c2[1]) - half_patch)
        max_c = int(max(c1[1], c2[1]) + half_patch)

        map_h, map_w = obs_map.shape
        min_r = max(0, min_r)
        max_r = min(map_h, max_r)
        min_c = max(0, min_c)
        max_c = min(map_w, max_c)

        if min_r >= max_r or min_c >= max_c:
            result = (False, 0, "empty_patch")
            self._mini_astar_cache[cache_key] = (*result, self._mini_astar_step)
            return result[0], result[1], result[2], "miss", cache_ttl, None

        local_patch = obs_map[min_r:max_r, min_c:max_c]
        start_local = (int(c1[0] - min_r), int(c1[1] - min_c))
        end_local = (int(c2[0] - min_r), int(c2[1] - min_c))

        if not (0 <= start_local[0] < local_patch.shape[0] and 0 <= start_local[1] < local_patch.shape[1]):
            result = (False, 0, "out_of_bounds")
            self._mini_astar_cache[cache_key] = (*result, self._mini_astar_step)
            return result[0], result[1], result[2], "miss", cache_ttl, None
        if not (0 <= end_local[0] < local_patch.shape[0] and 0 <= end_local[1] < local_patch.shape[1]):
            result = (False, 0, "out_of_bounds")
            self._mini_astar_cache[cache_key] = (*result, self._mini_astar_step)
            return result[0], result[1], result[2], "miss", cache_ttl, None

        cost = self._build_cost_grid(local_patch, None, edge_type="RR")
        path = pyastar2d.astar_path(cost, start_local, end_local, allow_diagonal=False)
        if path is None:
            result = (False, 0, "no_path", None)
        else:
            global_path = [(int(p[0] + min_r), int(p[1] + min_c)) for p in path]
            result = (True, int(path.shape[0]), "ok", global_path)

        self._mini_astar_cache[cache_key] = (*result[:3], self._mini_astar_step)
        return result[0], result[1], result[2], "miss", cache_ttl, result[3]

    def _boundary_wall_ratio(self, obs_map, node_idx, neighbor_idx):
        centroid1 = self.get_cell_center(node_idx)
        centroid2 = self.get_cell_center(neighbor_idx)
        num_samples = self.cell_size
        wall_hits = 0
        valid_samples = 0

        for i in range(num_samples + 1):
            t = i / num_samples
            r = int(centroid1[0] * (1 - t) + centroid2[0] * t)
            c = int(centroid1[1] * (1 - t) + centroid2[1] * t)
            if 0 <= r < obs_map.shape[0] and 0 <= c < obs_map.shape[1]:
                valid_samples += 1
                if obs_map[r, c] >= 0.8:
                    wall_hits += 1

        if valid_samples == 0:
            return 1.0
        return float(wall_hits) / float(valid_samples)

    def _should_fallback_when_portal_false(self, obs_map, node_idx, neighbor_idx):
        max_ratio = self.connectivity_cfg.get("graph_portal_fallback_max_obs_ratio", 0.2)
        return self._boundary_wall_ratio(obs_map, node_idx, neighbor_idx) <= max_ratio

    def _check_rr_edge(self, node_idx, neighbor_idx, obs_map, pred_mean_map, free_mask):
        edge_key = self._edge_key(node_idx, neighbor_idx)
        if edge_key in self.debug_edge_samples:
            samples = self._collect_line_samples(node_idx, neighbor_idx, obs_map)
            print(f"[RR DEBUG] edge={edge_key} line_samples={samples}")
        cached = self._rr_edge_cache.get(edge_key)
        if cached is not None:
            if cached.get("blocked"):
                return False, {
                    "portal": None,
                    "los": False,
                    "mini": None,
                    "cache": "stale",
                    "ttl_left": None,
                }
            if self._samples_blocked(cached["samples"], obs_map):
                cached["blocked"] = True
                return False, {
                    "portal": None,
                    "los": False,
                    "mini": None,
                    "cache": "stale",
                    "ttl_left": None,
                }
            return True, {
                "portal": None,
                "los": None,
                "mini": None,
                "cache": "hit",
                "ttl_left": None,
            }
        portal_ok = False
        if free_mask is not None:
            thickness = self.connectivity_cfg.get("graph_portal_thickness", 2)
            portal_ok = self._check_portal_clear(node_idx, neighbor_idx, free_mask, thickness=thickness)

        los_ok = self._check_path_clear(node_idx, neighbor_idx, obs_map)
        if portal_ok and los_ok:
            samples = self._get_clear_los_samples(node_idx, neighbor_idx, obs_map)
            if samples:
                self._rr_edge_cache[edge_key] = {"samples": samples, "blocked": False}
            return True, {
                "portal": True,
                "los": True,
                "mini": None,
                "cache": "miss",
                "ttl_left": None,
            }

        if portal_ok and not los_ok:
            ok, path_len, reason, cache_status, ttl_left, path_samples = self._run_mini_astar_local_rr(
                node_idx, neighbor_idx, obs_map
            )
            if ok and path_samples:
                self._rr_edge_cache[edge_key] = {"samples": path_samples, "blocked": False}
            return ok, {
                "portal": True,
                "los": False,
                "mini": (ok, path_len, reason),
                "cache": "miss" if cache_status == "miss" else cache_status,
                "ttl_left": ttl_left,
            }

        if not portal_ok and self._should_fallback_when_portal_false(obs_map, node_idx, neighbor_idx):
            ok, path_len, reason, cache_status, ttl_left, path_samples = self._run_mini_astar_local_rr(
                node_idx, neighbor_idx, obs_map
            )
            if ok and path_samples:
                self._rr_edge_cache[edge_key] = {"samples": path_samples, "blocked": False}
            return ok, {
                "portal": False,
                "los": False,
                "mini": (ok, path_len, reason),
                "cache": "miss" if cache_status == "miss" else cache_status,
                "ttl_left": ttl_left,
            }

        return False, {
            "portal": False,
            "los": False,
            "mini": None,
            "cache": "skip",
            "ttl_left": None,
        }

    def _check_ghost_edge(self, node_idx, neighbor_idx, obs_map, pred_mean_map, edge_type):
        obs_blocked = self._edge_strip_any_above(node_idx, neighbor_idx, obs_map, 0.8)
        if obs_blocked:
            return False, {
                "portal": None,
                "los": False,
                "mini": None,
                "cache": "skip",
                "ttl_left": None,
            }

        pred_hard = float(self.connectivity_cfg.get("graph_pred_hard_wall_threshold", 0.95))
        pred_mean_map = self._align_pred_map(pred_mean_map, obs_map.shape, fill_value=1.0)
        pred_blocked = self._edge_strip_any_above(node_idx, neighbor_idx, pred_mean_map, pred_hard)
        if pred_blocked:
            return False, {
                "portal": None,
                "los": False,
                "mini": None,
                "cache": "skip",
                "ttl_left": None,
            }

        return True, {
            "portal": None,
            "los": True,
            "mini": None,
            "cache": "skip",
            "ttl_left": None,
        }

    def _log_edge_debug(self, edge_type, portal, los, mini, cache_status, ttl_left):
        if not self.debug_cfg.get("graph_debug_edges", False):
            return
        mini_str = "none"
        if mini is not None:
            ok, path_len, reason = mini
            mini_str = f"hit={ok} path_len={path_len} fail_reason={reason}"
        ttl_str = "NA" if ttl_left is None else ttl_left
        print(
            f"edge_type={edge_type} portal={portal} los={los} "
            f"miniA*({mini_str}) cache={cache_status} ttl_left={ttl_str}"
        )

    def _check_path_clear(self, node_idx, neighbor_idx, obs_map, pred_mean_map=None, involves_ghost=False):
        """
        Check if robot can pass between two adjacent cells.
        
        SIMPLE STRICT CHECK: Block if ANY wall pixel on centroid line.
        Uses robot-centric cell centers (absolute pixel coordinates).
        
        Returns True if passage is possible, False if blocked.
        """
        centroid1 = self.get_cell_center(node_idx)
        centroid2 = self.get_cell_center(neighbor_idx)
        edge_key = self._edge_key(node_idx, neighbor_idx)
        want_viz = (
            self.debug_cfg.get("graph_debug_los_viz", False)
            and edge_key in self.debug_edge_samples
            and self.los_viz_dir
        )
        line_results = []
        points, clear = self._trace_los_line(centroid1, centroid2, obs_map)
        line_results.append({"offset": (0, 0), "points": points, "clear": clear})
        if clear and not want_viz:
            return True

        dr = neighbor_idx[0] - node_idx[0]
        dc = neighbor_idx[1] - node_idx[1]
        if dr == 0 and dc != 0:
            offsets = [(-1, 0), (1, 0)]
        elif dc == 0 and dr != 0:
            offsets = [(0, -1), (0, 1)]
        else:
            if want_viz:
                self._save_los_debug_plot(node_idx, neighbor_idx, obs_map, line_results)
            return False
        overall_ok = clear
        for off_r, off_c in offsets:
            points, clear = self._trace_los_line(
                centroid1, centroid2, obs_map, offset_r=off_r, offset_c=off_c
            )
            line_results.append({"offset": (off_r, off_c), "points": points, "clear": clear})
            if clear and not want_viz:
                return True
            overall_ok = overall_ok or clear

        if want_viz:
            self._save_los_debug_plot(node_idx, neighbor_idx, obs_map, line_results)

        return overall_ok

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

    def _compute_p_occ(self, obs_map, pred_mean_map, pred_var_map, eps=1e-4, k=2.0, sigma_max=0.5):
        p_occ = np.zeros_like(obs_map, dtype=np.float32)

        observed_free = (obs_map == 0.0)
        observed_occ = (obs_map == 1.0)
        unknown = (obs_map == 0.5)

        p_occ[observed_free] = 0.0
        p_occ[observed_occ] = 1.0

        if pred_mean_map is None or pred_var_map is None:
            p_occ[unknown] = 0.5
            return p_occ

        pred_mean_map = self._align_pred_map(pred_mean_map, obs_map.shape, fill_value=1.0)
        pred_var_map = self._align_pred_map(pred_var_map, obs_map.shape, fill_value=sigma_max * sigma_max)

        m = np.clip(pred_mean_map, eps, 1.0 - eps)
        sigma = np.sqrt(np.clip(pred_var_map, 0.0, sigma_max * sigma_max))
        s = np.exp(-k * sigma * sigma)

        logit = np.log(m / (1.0 - m))
        m_cal = 1.0 / (1.0 + np.exp(-s * logit))

        p_occ[unknown] = m_cal[unknown]
        return p_occ

    def _edge_risk_cost(self, node_idx, neighbor_idx, p_occ_map):
        if p_occ_map is None:
            return 1.0

        strip_div = max(1, int(self.connectivity_cfg.get("graph_edge_risk_strip_divisor", 6)))
        eps = float(self.connectivity_cfg.get("graph_edge_risk_eps", 1e-4))
        thickness = max(1, int(self.cell_size // strip_div))

        h, w = p_occ_map.shape
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

        total = 0.0
        count = 0

        if dr == 0 and dc == 1:
            a_strip = p_occ_map[a_r0:a_r1, max(a_c1 - thickness, a_c0):a_c1]
            b_strip = p_occ_map[b_r0:b_r1, b_c0:min(b_c0 + thickness, b_c1)]
            if a_strip.size:
                total += float(np.sum(a_strip))
                count += int(a_strip.size)
            if b_strip.size:
                total += float(np.sum(b_strip))
                count += int(b_strip.size)
        elif dr == 0 and dc == -1:
            a_strip = p_occ_map[a_r0:a_r1, a_c0:min(a_c0 + thickness, a_c1)]
            b_strip = p_occ_map[b_r0:b_r1, max(b_c1 - thickness, b_c0):b_c1]
            if a_strip.size:
                total += float(np.sum(a_strip))
                count += int(a_strip.size)
            if b_strip.size:
                total += float(np.sum(b_strip))
                count += int(b_strip.size)
        elif dr == 1 and dc == 0:
            a_strip = p_occ_map[max(a_r1 - thickness, a_r0):a_r1, a_c0:a_c1]
            b_strip = p_occ_map[b_r0:min(b_r0 + thickness, b_r1), b_c0:b_c1]
            if a_strip.size:
                total += float(np.sum(a_strip))
                count += int(a_strip.size)
            if b_strip.size:
                total += float(np.sum(b_strip))
                count += int(b_strip.size)
        elif dr == -1 and dc == 0:
            a_strip = p_occ_map[a_r0:min(a_r0 + thickness, a_r1), a_c0:a_c1]
            b_strip = p_occ_map[max(b_r1 - thickness, b_r0):b_r1, b_c0:b_c1]
            if a_strip.size:
                total += float(np.sum(a_strip))
                count += int(a_strip.size)
            if b_strip.size:
                total += float(np.sum(b_strip))
                count += int(b_strip.size)

        if count == 0:
            return float(-np.log(max(eps, 1e-12)))

        mean_occ = total / float(count)
        mean_occ = float(np.clip(mean_occ, 0.0, 1.0))
        p_free = max(1.0 - mean_occ, eps)
        return float(-np.log(p_free))

    def _edge_strip_max(self, node_idx, neighbor_idx, grid):
        if grid is None:
            return None

        strip_div = max(1, int(self.connectivity_cfg.get("graph_edge_risk_strip_divisor", 6)))
        thickness = max(1, int(self.cell_size // strip_div))

        h, w = grid.shape
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

        max_val = None

        if dr == 0 and dc == 1:
            a_strip = grid[a_r0:a_r1, max(a_c1 - thickness, a_c0):a_c1]
            b_strip = grid[b_r0:b_r1, b_c0:min(b_c0 + thickness, b_c1)]
            if a_strip.size:
                max_val = float(np.max(a_strip))
            if b_strip.size:
                max_val = float(np.max(b_strip)) if max_val is None else max(max_val, float(np.max(b_strip)))
        elif dr == 0 and dc == -1:
            a_strip = grid[a_r0:a_r1, a_c0:min(a_c0 + thickness, a_c1)]
            b_strip = grid[b_r0:b_r1, max(b_c1 - thickness, b_c0):b_c1]
            if a_strip.size:
                max_val = float(np.max(a_strip))
            if b_strip.size:
                max_val = float(np.max(b_strip)) if max_val is None else max(max_val, float(np.max(b_strip)))
        elif dr == 1 and dc == 0:
            a_strip = grid[max(a_r1 - thickness, a_r0):a_r1, a_c0:a_c1]
            b_strip = grid[b_r0:min(b_r0 + thickness, b_r1), b_c0:b_c1]
            if a_strip.size:
                max_val = float(np.max(a_strip))
            if b_strip.size:
                max_val = float(np.max(b_strip)) if max_val is None else max(max_val, float(np.max(b_strip)))
        elif dr == -1 and dc == 0:
            a_strip = grid[a_r0:min(a_r0 + thickness, a_r1), a_c0:a_c1]
            b_strip = grid[max(b_r1 - thickness, b_r0):b_r1, b_c0:b_c1]
            if a_strip.size:
                max_val = float(np.max(a_strip))
            if b_strip.size:
                max_val = float(np.max(b_strip)) if max_val is None else max(max_val, float(np.max(b_strip)))

        return max_val

    def _edge_strip_any_above(self, node_idx, neighbor_idx, grid, threshold):
        max_val = self._edge_strip_max(node_idx, neighbor_idx, grid)
        if max_val is None:
            return False
        return bool(max_val >= threshold)

    def _edge_strip_stats(self, pred_mean_map, p_occ_map):
        stats = {
            "pred_max": np.array([], dtype=float),
            "pocc_max": np.array([], dtype=float),
        }
        if not self.edge_costs:
            return stats

        obs_shape = None
        if p_occ_map is not None:
            obs_shape = p_occ_map.shape
        elif pred_mean_map is not None:
            obs_shape = pred_mean_map.shape
        if obs_shape is None:
            return stats

        if pred_mean_map is not None:
            pred_mean_map = self._align_pred_map(pred_mean_map, obs_shape, fill_value=1.0)
        if p_occ_map is not None:
            p_occ_map = self._align_pred_map(p_occ_map, obs_shape, fill_value=1.0)

        pred_vals = []
        pocc_vals = []
        for edge_key in self.edge_costs:
            node_idx, neighbor_idx = edge_key
            if pred_mean_map is not None:
                pred_max = self._edge_strip_max(node_idx, neighbor_idx, pred_mean_map)
                if pred_max is not None:
                    pred_vals.append(pred_max)
            if p_occ_map is not None:
                pocc_max = self._edge_strip_max(node_idx, neighbor_idx, p_occ_map)
                if pocc_max is not None:
                    pocc_vals.append(pocc_max)

        if pred_vals:
            stats["pred_max"] = np.array(pred_vals, dtype=float)
        if pocc_vals:
            stats["pocc_max"] = np.array(pocc_vals, dtype=float)

        return stats

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
        
        # Note: scent_map_vis is no longer used (absolute indexing now)

    def update_graph_light(self, robot_pose):
        """Update only current cell/visit state without rebuilding the graph."""
        if self.origin is None:
            self.origin = np.array(robot_pose, dtype=float)
        self._mini_astar_step += 1

        # Update current position
        curr_idx = self.get_cell_index(robot_pose)
        curr_node = self.get_cell(curr_idx)

        # Update parent/stack for backtracking
        if self.current_cell is not None and self.current_cell.index != curr_idx:
            if curr_node.parent is None and curr_node != self.current_cell.parent:
                curr_node.parent = self.current_cell
                self.visited_stack.append(self.current_cell)
            elif curr_node == self.current_cell.parent:
                if self.visited_stack:
                    self.visited_stack.pop()

        self.current_cell = curr_node
        if not curr_node.visited:
            curr_node.visited = True
            curr_node.visit_count = 1
        else:
            curr_node.visit_count += 1

    def update_graph(self, robot_pose, obs_map, pred_mean_map=None, pred_var_map=None, inflated_occ_grid=None):
        """Main update loop called by agent."""
        # Legacy: origin retained for compatibility, not used for indexing.
        if self.origin is None:
            self.origin = np.array(robot_pose, dtype=float)
        self.last_obs_shape = obs_map.shape
        self._mini_astar_step += 1

        change_threshold = self.connectivity_cfg.get("graph_cache_change_ratio_threshold", 0.01)
        obs_change = self._map_change_ratio(self._last_obs_map, obs_map)
        if obs_change > change_threshold:
            self._obs_update_id += 1
            self._mini_astar_cache.clear()
        self._last_obs_map = obs_map.copy()

        if pred_mean_map is not None:
            pred_change = self._map_change_ratio(self._last_pred_map, pred_mean_map)
            if pred_change > change_threshold:
                self._pred_update_id += 1
                self._mini_astar_cache.clear()
            self._last_pred_map = pred_mean_map.copy()
        else:
            self._last_pred_map = None

        inflated_sig = self._inflated_signature(inflated_occ_grid)
        if inflated_sig != self._last_inflated_sig:
            self._grid_update_id += 1
            self._mini_astar_cache.clear()
        self._last_inflated_sig = inflated_sig
        self._dijkstra_cache = None
        
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
        Get the next cell from the risk path.
        
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
        best_score = -1e9
        
        # Run diffusion first to get propagated values
        pred_var_map = self._align_pred_map(pred_var_map, self.last_obs_shape)
        self.diffuse_scent(pred_var_map)
        
        risk_lambda = float(self._get_cfg("graph_target_risk_lambda", 0.5))
        dist = None
        prev = None
        if current_cell is not None:
            dist, prev = self._run_dijkstra(current_cell.index)
            self._dijkstra_cache = (current_cell.index, dist, prev)

        # Find best cell (ghost OR real) among reachable cells based on score
        for cell in self.cells.values():
            if cell.is_blocked:
                continue
            
            # Skip current cell itself
            if current_cell is not None and cell.index == current_cell.index:
                continue

            if current_cell is not None:
                if dist is None:
                    continue
                cost = dist.get(cell.index)
                if cost is None:
                    continue
                score = cell.propagated_value - risk_lambda * cost
            else:
                score = cell.propagated_value
                
            # Select cell with highest score
            if score > best_score:
                best_score = score
                best_cell = cell
        
        if best_cell is None and current_cell is not None:
            reachable_count = len(dist) if dist is not None else 0
            print(f"[HIGH-LEVEL] No reachable target cells! Reachable: {reachable_count} cells")
        elif best_cell is not None:
            cell_type = "Ghost" if best_cell.is_ghost else "Real"
            cost_str = ""
            if dist is not None and best_cell.index in dist:
                cost_str = f", cost={dist[best_cell.index]:.3f}"
            print(
                f"[HIGH-LEVEL] Selected {cell_type} cell {best_cell.index} "
                f"score={best_score:.4f}{cost_str}"
            )
        
        return best_cell

    def _run_dijkstra(self, start_idx):
        import heapq

        dist = {start_idx: 0.0}
        prev = {}
        heap = [(0.0, start_idx)]
        visited = set()

        while heap:
            curr_cost, curr_idx = heapq.heappop(heap)
            if curr_idx in visited:
                continue
            visited.add(curr_idx)

            curr_node = self.cells.get(curr_idx)
            if curr_node is None:
                continue

            for neighbor in curr_node.neighbors:
                if neighbor.is_blocked:
                    continue
                edge_key = self._edge_key(curr_idx, neighbor.index)
                edge_cost = self.edge_costs.get(edge_key, 1.0)
                new_cost = curr_cost + edge_cost
                if new_cost < dist.get(neighbor.index, float("inf")):
                    dist[neighbor.index] = new_cost
                    prev[neighbor.index] = curr_idx
                    heapq.heappush(heap, (new_cost, neighbor.index))

        return dist, prev

    def find_path_to_target(self, start_cell, target_cell):
        """
        Find path from start_cell to target_cell using Dijkstra (risk cost).
        
        Returns:
            path: list of CellNodes from start to target (inclusive)
                  or None if no path found
        """
        if start_cell is None or target_cell is None:
            print(f"  [PATH] Cannot find path: start={start_cell}, target={target_cell}")
            return None

        cached = self._dijkstra_cache
        if cached is not None:
            cache_start, dist, prev = cached
            if cache_start == start_cell.index and target_cell.index in dist:
                path_indices = [target_cell.index]
                while path_indices[-1] != start_cell.index:
                    next_idx = prev.get(path_indices[-1])
                    if next_idx is None:
                        break
                    path_indices.append(next_idx)
                if path_indices[-1] == start_cell.index:
                    path_indices.reverse()
                    path = [self.cells[idx] for idx in path_indices]
                    path_str = " → ".join([f"{c.index}" for c in path])
                    print(
                        f"  [PATH] Found (cached): {path_str} "
                        f"({len(path)} cells, cost={dist[target_cell.index]:.3f})"
                    )
                    return path
        
        if start_cell.index == target_cell.index:
            return [start_cell]
        
        dist, prev = self._run_dijkstra(start_cell.index)

        if target_cell.index not in dist:
            print(f"  [PATH] NO PATH from {start_cell.index} to {target_cell.index}!")
            print(f"         Start neighbors: {[n.index for n in start_cell.neighbors]}")
            print(f"         Target neighbors: {[n.index for n in target_cell.neighbors]}")
            print(f"         Visited {len(dist)} cells, target not reachable")
            return None

        path_indices = [target_cell.index]
        while path_indices[-1] != start_cell.index:
            path_indices.append(prev[path_indices[-1]])
        path_indices.reverse()
        path = [self.cells[idx] for idx in path_indices]

        path_str = " → ".join([f"{c.index}" for c in path])
        print(f"  [PATH] Found: {path_str} ({len(path)} cells, cost={dist[target_cell.index]:.3f})")
        return path
