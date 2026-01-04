# Graph Stable Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement graph_stable with config injection, unified connectivity, safe mini-A* caching, and debug gating, while keeping pytest -q green after each step.

**Architecture:** `explore.py` reads Hydra options and passes compact config dicts into `CellManager`. `graph_utils.py` uses those dicts for ghost promotion, connectivity, and caching without reading Hydra directly. Prediction is only used for ghost creation and mini-A* cost grids; obs_map alone drives hard blockers.

**Tech Stack:** Python, numpy, pyastar2d, pytest, Hydra/OmegaConf.

---

### Task 1: Config injection + patch-based ghost promotion + diffusion base value

**Files:**
- Modify: `configs/base.yaml:27-45`
- Modify: `nbh/explore.py:210-250`
- Modify: `nbh/graph_utils.py:28-260`
- Modify: `nbh/graph_utils.py:427-479`
- Test: `tests/test_graph_ghost_promotion.py`

**Step 1: Write the failing test**

Create `tests/test_graph_ghost_promotion.py`:

```python
import numpy as np

from nbh.graph_utils import CellManager


def _make_manager(graph_cfg=None):
    cfg = graph_cfg or {}
    return CellManager(
        cell_size=4,
        start_pose=np.array([2, 2]),
        promotion_cfg=cfg,
        connectivity_cfg={},
        debug_cfg={},
    )


def _build_maps(pred_mean_val=0.2, pred_var_val=0.1):
    obs = np.full((8, 8), 0.5, dtype=np.float32)
    obs[0:4, 0:4] = 0.0  # make cell (0,0) observed free

    pred_mean = np.full((8, 8), pred_mean_val, dtype=np.float32)
    pred_var = np.full((8, 8), pred_var_val, dtype=np.float32)
    return obs, pred_mean, pred_var


def test_ghost_promotion_uses_patch_thresholds():
    obs, pred_mean, pred_var = _build_maps(pred_mean_val=0.2, pred_var_val=0.1)
    manager = _make_manager({
        "graph_max_ghost_distance": 1,
        "graph_ghost_pred_mean_free_threshold": 0.4,
        "graph_ghost_pred_var_max_threshold": 0.3,
        "graph_obs_blocked_ratio": 0.3,
        "graph_unknown_ratio_threshold": 0.5,
        "graph_centroid_blocked_threshold": 0.8,
    })

    manager.update_graph(
        robot_pose=np.array([2, 2]),
        obs_map=obs,
        pred_mean_map=pred_mean,
        pred_var_map=pred_var,
        inflated_occ_grid=None,
    )

    ghost = manager.cells.get((0, 1))
    assert ghost is not None
    assert ghost.is_ghost is True


def test_ghost_promotion_respects_var_threshold():
    obs, pred_mean, pred_var = _build_maps(pred_mean_val=0.2, pred_var_val=0.5)
    manager = _make_manager({
        "graph_max_ghost_distance": 1,
        "graph_ghost_pred_mean_free_threshold": 0.4,
        "graph_ghost_pred_var_max_threshold": 0.3,
        "graph_obs_blocked_ratio": 0.3,
        "graph_unknown_ratio_threshold": 0.5,
        "graph_centroid_blocked_threshold": 0.8,
    })

    manager.update_graph(
        robot_pose=np.array([2, 2]),
        obs_map=obs,
        pred_mean_map=pred_mean,
        pred_var_map=pred_var,
        inflated_occ_grid=None,
    )

    ghost = manager.cells.get((0, 1))
    assert ghost is None
```

**Step 2: Run test to verify it fails**

Run:

```bash
bash -lc "source /home/seungjab/anaconda3/etc/profile.d/conda.sh && CONDA_NO_PLUGINS=true conda activate nbh && pytest -q tests/test_graph_ghost_promotion.py"
```

Expected: FAIL because `CellManager` does not accept config dicts and ghost promotion is still centroid-based.

**Step 3: Write minimal implementation**

1) Update `configs/base.yaml` to include graph config keys under `nbh:` (flat `graph_*` keys):

```yaml
nbh:
  graph_max_ghost_distance: 2
  graph_obs_blocked_ratio: 0.3
  graph_unknown_ratio_threshold: 0.5
  graph_centroid_blocked_threshold: 0.8
  graph_ghost_pred_mean_free_threshold: 0.4
  graph_ghost_pred_var_max_threshold: 0.3
  graph_diffuse_gamma: 0.95
  graph_diffuse_iterations: 50
  graph_diffuse_on_update: false
```

2) Update `nbh/explore.py` to build config dicts and pass them to `CellManager`:

```python
nbh_cfg = getattr(collect_opts, "nbh", {})

promotion_cfg = {
    "graph_max_ghost_distance": getattr(nbh_cfg, "graph_max_ghost_distance", 2),
    "graph_obs_blocked_ratio": getattr(nbh_cfg, "graph_obs_blocked_ratio", 0.3),
    "graph_unknown_ratio_threshold": getattr(nbh_cfg, "graph_unknown_ratio_threshold", 0.5),
    "graph_centroid_blocked_threshold": getattr(nbh_cfg, "graph_centroid_blocked_threshold", 0.8),
    "graph_ghost_pred_mean_free_threshold": getattr(nbh_cfg, "graph_ghost_pred_mean_free_threshold", 0.4),
    "graph_ghost_pred_var_max_threshold": getattr(nbh_cfg, "graph_ghost_pred_var_max_threshold", 0.3),
    "graph_diffuse_gamma": getattr(nbh_cfg, "graph_diffuse_gamma", 0.95),
    "graph_diffuse_iterations": getattr(nbh_cfg, "graph_diffuse_iterations", 50),
    "graph_diffuse_on_update": getattr(nbh_cfg, "graph_diffuse_on_update", False),
}

connectivity_cfg = {}
debug_cfg = {}

cell_manager = CellManager(
    cell_size=CELL_SIZE_CONFIG,
    start_pose=start_pose,
    valid_space_map=validspace_map,
    promotion_cfg=promotion_cfg,
    connectivity_cfg=connectivity_cfg,
    debug_cfg=debug_cfg,
)
```

3) Update `nbh/graph_utils.py`:

- Accept config dicts in `CellManager.__init__` and store defaults.
- Update ghost promotion to use patch mean thresholds (pred_mean/pred_var).
- Update `diffuse_scent` to use patch mean of pred_var (not centroid).

```python
class CellManager:
    def __init__(self, cell_size=50, start_pose=None, valid_space_map=None,
                 promotion_cfg=None, connectivity_cfg=None, debug_cfg=None):
        self.promotion_cfg = promotion_cfg or {}
        self.connectivity_cfg = connectivity_cfg or {}
        self.debug_cfg = debug_cfg or {}
        # ... keep existing init code

    def _get_cfg(self, key, default):
        return self.promotion_cfg.get(key, default)

    def _patch_mean(self, patch, default=0.0):
        if patch is None or patch.size == 0:
            return default
        return float(np.mean(patch))
```

Ghost promotion core (within `update_graph_structure`):

```python
obs_blocked_ratio = self._get_cfg("graph_obs_blocked_ratio", 0.3)
unknown_ratio_threshold = self._get_cfg("graph_unknown_ratio_threshold", 0.5)
centroid_blocked_threshold = self._get_cfg("graph_centroid_blocked_threshold", 0.8)
max_ghost_distance = self._get_cfg("graph_max_ghost_distance", max_ghost_distance)

is_obs_blocked = np.mean(patch_obs == 1) > obs_blocked_ratio
if centroid_obs_val >= centroid_blocked_threshold:
    is_obs_blocked = True

unknown_ratio = float(np.mean(patch_obs == 0.5))
is_unknown = (unknown_ratio > unknown_ratio_threshold) and (centroid_obs_val == 0.5)

pred_mean_free_threshold = self._get_cfg("graph_ghost_pred_mean_free_threshold", 0.4)
pred_var_max_threshold = self._get_cfg("graph_ghost_pred_var_max_threshold", 0.3)

pred_mean_patch = self._patch_mean(patch_pred, default=1.0)
var_patch = None
if pred_var_map is not None:
    var_patch = pred_var_map[y_start_clip:y_end_clip, x_start_clip:x_end_clip]

pred_var_patch_mean = self._patch_mean(var_patch, default=1.0)

is_pred_free = pred_mean_patch < pred_mean_free_threshold

if is_unknown and is_pred_free:
    ghost_distance = distances.get(idx, float('inf'))
    if pred_var_patch_mean < pred_var_max_threshold and ghost_distance <= max_ghost_distance:
        node = self.get_cell(idx, is_ghost=True)
        node.is_blocked = False
        node.base_value = pred_var_patch_mean
```

Diffusion base value (in `diffuse_scent`):

```python
if node.is_ghost:
    half = self.cell_size // 2
    r0 = max(0, int(node.center[0] - half))
    r1 = min(map_h, int(node.center[0] + half))
    c0 = max(0, int(node.center[1] - half))
    c1 = min(map_w, int(node.center[1] + half))
    patch = pred_var_map[r0:r1, c0:c1]
    node.base_value = self._patch_mean(patch, default=0.0)
```

**Step 4: Run full test suite**

```bash
bash -lc "source /home/seungjab/anaconda3/etc/profile.d/conda.sh && CONDA_NO_PLUGINS=true conda activate nbh && pytest -q"
```

Expected: PASS.

**Step 5: Commit**

```bash
git add configs/base.yaml nbh/explore.py nbh/graph_utils.py tests/test_graph_ghost_promotion.py
git commit -m "feat: inject graph config and patch-based ghost promotion"
```

---

### Task 2: Connectivity refactor (RR portal-first + RG/GG mini-A*)

**Files:**
- Modify: `nbh/graph_utils.py:261-392`
- Modify: `nbh/graph_utils.py:281-317`
- Test: `tests/test_graph_connectivity_policy.py`

**Step 1: Write the failing tests**

Create `tests/test_graph_connectivity_policy.py`:

```python
import numpy as np

from nbh.graph_utils import CellManager


def _make_manager(connectivity_cfg=None):
    return CellManager(
        cell_size=4,
        start_pose=np.array([2, 2]),
        promotion_cfg={
            "graph_max_ghost_distance": 1,
            "graph_obs_blocked_ratio": 0.3,
            "graph_unknown_ratio_threshold": 0.5,
            "graph_centroid_blocked_threshold": 0.8,
            "graph_ghost_pred_mean_free_threshold": 0.4,
            "graph_ghost_pred_var_max_threshold": 0.3,
        },
        connectivity_cfg=connectivity_cfg or {},
        debug_cfg={},
    )


def _free_grid(shape, wall_col=None):
    obs = np.zeros(shape, dtype=np.float32)
    if wall_col is not None:
        obs[:, wall_col] = 1.0
    return obs


def test_rr_thin_wall_blocks_edge():
    obs = _free_grid((8, 8), wall_col=4)
    pred_mean = np.zeros_like(obs)
    pred_var = np.zeros_like(obs)
    inflated = np.where(obs >= 0.8, np.inf, 1.0)

    manager = _make_manager({
        "graph_portal_fallback_max_obs_ratio": 0.2,
        "graph_pred_wall_threshold": 0.7,
        "graph_unknown_as_occ": True,
        "graph_dilate_diam": 3,
    })

    manager.update_graph(
        robot_pose=np.array([2, 2]),
        obs_map=obs,
        pred_mean_map=pred_mean,
        pred_var_map=pred_var,
        inflated_occ_grid=inflated,
    )

    node = manager.cells.get((0, 0))
    assert node is not None
    assert not any(n.index == (0, 1) for n in node.neighbors)


def test_portal_false_negative_falls_back_to_mini_astar():
    obs = _free_grid((8, 8), wall_col=None)
    pred_mean = np.zeros_like(obs)
    pred_var = np.zeros_like(obs)
    inflated = np.full_like(obs, np.inf)  # portal will be False

    manager = _make_manager({
        "graph_portal_fallback_max_obs_ratio": 0.2,
        "graph_pred_wall_threshold": 0.7,
        "graph_unknown_as_occ": True,
        "graph_dilate_diam": 3,
    })

    manager.update_graph(
        robot_pose=np.array([2, 2]),
        obs_map=obs,
        pred_mean_map=pred_mean,
        pred_var_map=pred_var,
        inflated_occ_grid=inflated,
    )

    node = manager.cells.get((0, 0))
    assert node is not None
    assert any(n.index == (0, 1) for n in node.neighbors)
```

**Step 2: Run tests to verify failure**

```bash
bash -lc "source /home/seungjab/anaconda3/etc/profile.d/conda.sh && CONDA_NO_PLUGINS=true conda activate nbh && pytest -q tests/test_graph_connectivity_policy.py"
```

Expected: FAIL because RR/RG/GG logic still uses LOS and no mini-A* fallback for portal False.

**Step 3: Write minimal implementation**

In `nbh/graph_utils.py`:

1) Remove pred_mean checks from `_check_path_clear` so it uses obs_map only.

2) Add a cost-grid builder and mini-A* function:

```python
import pyastar2d

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

    def _run_mini_astar(self, node_idx, neighbor_idx, cost_grid):
        start = tuple(self.get_cell_center(node_idx).astype(int))
        goal = tuple(self.get_cell_center(neighbor_idx).astype(int))
        path = pyastar2d.astar_path(cost_grid, start, goal, allow_diagonal=False)
        if path is None:
            return False, 0, "no_path"
        return True, int(path.shape[0]), "ok"
```

3) Add RR/RG/GG decision helpers:

```python
    def _should_fallback_when_portal_false(self, obs_map, node_idx, neighbor_idx):
        # compute wall ratio in boundary strip
        # return True if ratio <= graph_portal_fallback_max_obs_ratio
```

```python
    def _check_rr_edge(self, node_idx, neighbor_idx, obs_map, pred_mean_map, free_mask):
        portal_ok = self._check_portal_clear(node_idx, neighbor_idx, free_mask, thickness=self.connectivity_cfg.get("graph_portal_thickness", 2))
        los_ok = self._check_path_clear(node_idx, neighbor_idx, obs_map)
        if portal_ok and los_ok:
            return True, {"portal": True, "los": True, "mini": None}
        if portal_ok and not los_ok:
            cost = self._build_cost_grid(obs_map, pred_mean_map, edge_type="RR")
            ok, path_len, reason = self._run_mini_astar(node_idx, neighbor_idx, cost)
            return ok, {"portal": True, "los": False, "mini": (ok, path_len, reason)}
        if not portal_ok and self._should_fallback_when_portal_false(obs_map, node_idx, neighbor_idx):
            cost = self._build_cost_grid(obs_map, pred_mean_map, edge_type="RR")
            ok, path_len, reason = self._run_mini_astar(node_idx, neighbor_idx, cost)
            return ok, {"portal": False, "los": False, "mini": (ok, path_len, reason)}
        return False, {"portal": False, "los": False, "mini": None}
```

```python
    def _check_ghost_edge(self, node_idx, neighbor_idx, obs_map, pred_mean_map, edge_type):
        cost = self._build_cost_grid(obs_map, pred_mean_map, edge_type=edge_type)
        ok, path_len, reason = self._run_mini_astar(node_idx, neighbor_idx, cost)
        return ok, {"portal": None, "los": None, "mini": (ok, path_len, reason)}
```

4) Update the neighbor loop to use the helpers and remove LOS-only logic for ghosts:

```python
if not involves_ghost:
    ok, info = self._check_rr_edge(idx, (nr, nc), obs_map, pred_mean_map, self.free_mask)
else:
    edge_type = "RG" if (node.is_ghost != neighbor.is_ghost) else "GG"
    ok, info = self._check_ghost_edge(idx, (nr, nc), obs_map, pred_mean_map, edge_type)
if ok:
    node.neighbors.append(neighbor)
```

**Step 4: Run full test suite**

```bash
bash -lc "source /home/seungjab/anaconda3/etc/profile.d/conda.sh && CONDA_NO_PLUGINS=true conda activate nbh && pytest -q"
```

Expected: PASS.

**Step 5: Commit**

```bash
git add nbh/graph_utils.py tests/test_graph_connectivity_policy.py
git commit -m "feat: unify connectivity with portal-first and mini-astar"
```

---

### Task 3: Mini-A* cache + invalidation on map changes

**Files:**
- Modify: `nbh/graph_utils.py:28-120`
- Modify: `nbh/graph_utils.py:483-520`
- Modify: `nbh/graph_utils.py:220-280`
- Test: `tests/test_graph_cache_invalidation.py`

**Step 1: Write the failing test**

Create `tests/test_graph_cache_invalidation.py`:

```python
import numpy as np

from nbh.graph_utils import CellManager


def _make_manager(connectivity_cfg=None):
    return CellManager(
        cell_size=4,
        start_pose=np.array([2, 2]),
        promotion_cfg={
            "graph_max_ghost_distance": 1,
            "graph_obs_blocked_ratio": 0.3,
            "graph_unknown_ratio_threshold": 0.5,
            "graph_centroid_blocked_threshold": 0.8,
            "graph_ghost_pred_mean_free_threshold": 0.4,
            "graph_ghost_pred_var_max_threshold": 0.3,
        },
        connectivity_cfg=connectivity_cfg or {},
        debug_cfg={},
    )


def test_cache_invalidates_on_obs_change():
    obs = np.zeros((8, 8), dtype=np.float32)
    pred_mean = np.zeros_like(obs)
    pred_var = np.zeros_like(obs)
    inflated = np.where(obs >= 0.8, np.inf, 1.0)

    manager = _make_manager({
        "graph_mini_astar_ttl_steps": 100,
        "graph_cache_change_ratio_threshold": 0.0,
        "graph_pred_wall_threshold": 0.7,
        "graph_unknown_as_occ": True,
        "graph_dilate_diam": 3,
    })

    manager.update_graph(np.array([2, 2]), obs, pred_mean, pred_var, inflated)
    node = manager.cells.get((0, 0))
    assert any(n.index == (0, 1) for n in node.neighbors)

    obs2 = obs.copy()
    obs2[:, 4] = 1.0
    inflated2 = np.where(obs2 >= 0.8, np.inf, 1.0)

    manager.update_graph(np.array([2, 2]), obs2, pred_mean, pred_var, inflated2)
    node2 = manager.cells.get((0, 0))
    assert not any(n.index == (0, 1) for n in node2.neighbors)
```

**Step 2: Run test to verify failure**

```bash
bash -lc "source /home/seungjab/anaconda3/etc/profile.d/conda.sh && CONDA_NO_PLUGINS=true conda activate nbh && pytest -q tests/test_graph_cache_invalidation.py"
```

Expected: FAIL because cache does not invalidate on obs change.

**Step 3: Write minimal implementation**

In `nbh/graph_utils.py`:

1) Add cache state in `__init__`:

```python
self._mini_astar_cache = {}
self._mini_astar_step = 0
self._obs_update_id = 0
self._pred_update_id = 0
self._grid_update_id = 0
self._last_obs_map = None
self._last_pred_map = None
self._last_inflated_sig = None
```

2) Add helpers:

```python
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
        self.connectivity_cfg.get("graph_dilate_diam", 3),
        self._obs_update_id,
        self._pred_update_id,
        self._grid_update_id,
    ))
```

3) In `update_graph`, before `update_graph_structure`, bump versions and invalidate cache:

```python
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

inflated_sig = self._inflated_signature(inflated_occ_grid)
if inflated_sig != self._last_inflated_sig:
    self._grid_update_id += 1
    self._mini_astar_cache.clear()
self._last_inflated_sig = inflated_sig
```

4) Update `_run_mini_astar` to use cache key and TTL:

```python
cache_ttl = self.connectivity_cfg.get("graph_mini_astar_ttl_steps", 10)
policy_hash = self._grid_policy_hash()
cache_key = (node_idx, neighbor_idx, policy_hash)

entry = self._mini_astar_cache.get(cache_key)
if entry is not None:
    ok, path_len, reason, last_step = entry
    ttl_left = cache_ttl - (self._mini_astar_step - last_step)
    if ttl_left > 0:
        return ok, path_len, reason

# compute fresh result
ok, path_len, reason = ...
self._mini_astar_cache[cache_key] = (ok, path_len, reason, self._mini_astar_step)
return ok, path_len, reason
```

**Step 4: Run full test suite**

```bash
bash -lc "source /home/seungjab/anaconda3/etc/profile.d/conda.sh && CONDA_NO_PLUGINS=true conda activate nbh && pytest -q"
```

Expected: PASS.

**Step 5: Commit**

```bash
git add nbh/graph_utils.py tests/test_graph_cache_invalidation.py
git commit -m "feat: add mini-astar cache with map invalidation"
```

---

### Task 4: Debug logging gates + target logs

**Files:**
- Modify: `configs/base.yaml:27-45`
- Modify: `nbh/explore.py:528-670`
- Modify: `nbh/graph_utils.py:261-360`
- Test: `tests/test_graph_debug_flags.py`

**Step 1: Write the failing test**

Create `tests/test_graph_debug_flags.py`:

```python
import numpy as np

from nbh.graph_utils import CellManager


def test_debug_edges_toggle(capsys):
    obs = np.zeros((8, 8), dtype=np.float32)
    pred_mean = np.zeros_like(obs)
    pred_var = np.zeros_like(obs)
    inflated = np.where(obs >= 0.8, np.inf, 1.0)

    manager = CellManager(
        cell_size=4,
        start_pose=np.array([2, 2]),
        promotion_cfg={
            "graph_max_ghost_distance": 1,
            "graph_obs_blocked_ratio": 0.3,
            "graph_unknown_ratio_threshold": 0.5,
            "graph_centroid_blocked_threshold": 0.8,
            "graph_ghost_pred_mean_free_threshold": 0.4,
            "graph_ghost_pred_var_max_threshold": 0.3,
        },
        connectivity_cfg={
            "graph_pred_wall_threshold": 0.7,
            "graph_unknown_as_occ": True,
            "graph_dilate_diam": 3,
        },
        debug_cfg={"graph_debug_edges": True},
    )

    manager.update_graph(np.array([2, 2]), obs, pred_mean, pred_var, inflated)
    out = capsys.readouterr().out
    assert "edge_type" in out

    manager2 = CellManager(
        cell_size=4,
        start_pose=np.array([2, 2]),
        promotion_cfg=manager.promotion_cfg,
        connectivity_cfg=manager.connectivity_cfg,
        debug_cfg={"graph_debug_edges": False},
    )
    manager2.update_graph(np.array([2, 2]), obs, pred_mean, pred_var, inflated)
    out2 = capsys.readouterr().out
    assert "edge_type" not in out2
```

**Step 2: Run test to verify failure**

```bash
bash -lc "source /home/seungjab/anaconda3/etc/profile.d/conda.sh && CONDA_NO_PLUGINS=true conda activate nbh && pytest -q tests/test_graph_debug_flags.py"
```

Expected: FAIL because debug logging is not implemented or not gated.

**Step 3: Write minimal implementation**

1) Add debug flags to `configs/base.yaml`:

```yaml
nbh:
  graph_debug_edges: false
  graph_debug_targets: false
  graph_debug_belief: false
  graph_debug_propagation_stats: false
```

2) In `nbh/explore.py`, read debug flags from config and gate prints:

```python
debug_edges = getattr(nbh_cfg, "graph_debug_edges", False)
debug_targets = getattr(nbh_cfg, "graph_debug_targets", False)
debug_belief = getattr(nbh_cfg, "graph_debug_belief", False)
debug_prop = getattr(nbh_cfg, "graph_debug_propagation_stats", False)

debug_cfg = {
    "graph_debug_edges": debug_edges,
    "graph_debug_targets": debug_targets,
    "graph_debug_belief": debug_belief,
    "graph_debug_propagation_stats": debug_prop,
}
```

Replace `DEBUG_PROPAGATION_STATS` usage:

```python
if debug_cfg.get("graph_debug_propagation_stats") and cell_manager.cells:
    ...
```

Gate target selection logs:

```python
if debug_cfg.get("graph_debug_targets"):
    print(f"[HIGH-LEVEL] Reselecting target (reason: {reasons_str})")
```

3) In `nbh/graph_utils.py`, add a debug log helper and call it in connectivity decisions:

```python
def _log_edge_debug(self, edge_type, portal, los, mini, cache_status, ttl_left):
    if not self.debug_cfg.get("graph_debug_edges", False):
        return
    mini_str = "none"
    if mini is not None:
        ok, path_len, reason = mini
        mini_str = f"hit={ok} path_len={path_len} fail_reason={reason}"
    print(
        f"edge_type={edge_type} portal={portal} los={los} "
        f"miniA*({mini_str}) cache={cache_status} ttl_left={ttl_left}"
    )
```

Call `_log_edge_debug` after each edge decision in RR/RG/GG paths.

**Step 4: Run full test suite**

```bash
bash -lc "source /home/seungjab/anaconda3/etc/profile.d/conda.sh && CONDA_NO_PLUGINS=true conda activate nbh && pytest -q"
```

Expected: PASS.

**Step 5: Commit**

```bash
git add configs/base.yaml nbh/explore.py nbh/graph_utils.py tests/test_graph_debug_flags.py
git commit -m "feat: gate debug logging for edges and targets"
```

---

## Execution Notes
- Always run tests via conda: `source /home/seungjab/anaconda3/etc/profile.d/conda.sh && CONDA_NO_PLUGINS=true conda activate nbh`.
- Keep each task small and keep pytest -q green after each task.
- If a test in a later task requires config keys added earlier, add defaults first to avoid breaking prior steps.
