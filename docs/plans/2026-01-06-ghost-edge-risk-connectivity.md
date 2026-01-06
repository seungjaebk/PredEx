# Ghost Edge Risk Connectivity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Option C ghost edge connectivity (hard-block only for observed walls or very high predicted walls) and cache RR connectivity, invalidating only when new observed walls block it.

**Architecture:** Ghost edges stop using mini-A* and are connected unless blocked by observed walls or a high pred_mean strip threshold. RR edges keep current portal/LOS/mini-A* logic but cache positive edges; cached edges are invalidated only when newly observed walls block the stored clearance samples.

**Tech Stack:** Python, numpy, pyastar2d, pytest

### Task 1: Add failing tests for ghost edge policy C and RR cache invalidation

**Files:**
- Create: `tests/test_graph_ghost_edge_policy.py`
- Modify: `tests/test_graph_connectivity_policy.py` (optional, only if needed for RR cache behavior)

**Step 1: Write the failing test (ghost edge allows below hard threshold)**

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


def _base_maps():
    obs = np.full((8, 8), 0.5, dtype=np.float32)
    obs[0:4, 0:4] = 0.0  # seed real cell at (0, 0)
    pred_mean = np.full_like(obs, 0.05)
    pred_var = np.full_like(obs, 0.1)
    return obs, pred_mean, pred_var


def test_ghost_edge_allows_high_pred_below_hard_threshold():
    obs, pred_mean, pred_var = _base_maps()
    # put moderately high pred on the boundary strip, but below hard threshold
    pred_mean[0:4, 3] = 0.8
    pred_mean[0:4, 4] = 0.8

    manager = _make_manager({
        "graph_unknown_as_occ": True,
        "graph_pred_wall_threshold": 0.7,
        "graph_pred_hard_wall_threshold": 0.95,
    })

    manager.update_graph(np.array([2, 2]), obs, pred_mean, pred_var, None)
    manager.update_graph(np.array([2, 2]), obs, pred_mean, pred_var, None)

    real = manager.cells.get((0, 0))
    ghost = manager.cells.get((0, 1))
    assert real is not None
    assert ghost is not None
    assert any(n.index == ghost.index for n in real.neighbors)
```

**Step 2: Write the failing test (ghost edge blocks above hard threshold)**

```python

def test_ghost_edge_blocks_when_pred_exceeds_hard_threshold():
    obs, pred_mean, pred_var = _base_maps()
    pred_mean[0:4, 3] = 0.97
    pred_mean[0:4, 4] = 0.97

    manager = _make_manager({
        "graph_unknown_as_occ": True,
        "graph_pred_wall_threshold": 0.7,
        "graph_pred_hard_wall_threshold": 0.95,
    })

    manager.update_graph(np.array([2, 2]), obs, pred_mean, pred_var, None)
    manager.update_graph(np.array([2, 2]), obs, pred_mean, pred_var, None)

    real = manager.cells.get((0, 0))
    ghost = manager.cells.get((0, 1))
    assert real is not None
    assert ghost is not None
    assert not any(n.index == ghost.index for n in real.neighbors)
```

**Step 3: Write the failing test (RR cache invalidates on new wall)**

```python

def test_rr_edge_invalidates_on_new_wall():
    obs = np.zeros((8, 8), dtype=np.float32)
    pred_mean = np.zeros_like(obs)
    pred_var = np.zeros_like(obs)

    manager = _make_manager({
        "graph_unknown_as_occ": True,
        "graph_pred_wall_threshold": 0.7,
    })

    manager.update_graph(np.array([2, 2]), obs, pred_mean, pred_var, None)
    manager.update_graph(np.array([2, 2]), obs, pred_mean, pred_var, None)

    real = manager.cells.get((0, 0))
    assert real is not None
    assert any(n.index == (0, 1) for n in real.neighbors)

    obs2 = obs.copy()
    obs2[2, 4] = 1.0  # wall on the line between (0,0) and (0,1)

    manager.update_graph(np.array([2, 2]), obs2, pred_mean, pred_var, None)
    manager.update_graph(np.array([2, 2]), obs2, pred_mean, pred_var, None)

    real2 = manager.cells.get((0, 0))
    assert real2 is not None
    assert not any(n.index == (0, 1) for n in real2.neighbors)
```

**Step 4: Run tests to confirm they fail**

Run:
- `pytest tests/test_graph_ghost_edge_policy.py -v`

Expected: FAIL (ghost edges still use binary mini-A*)

### Task 2: Implement ghost edge Option C and RR cache

**Files:**
- Modify: `nbh/graph_utils.py`
- Modify: `nbh/explore.py`
- Modify: `configs/base.yaml`

**Step 1: Add new config param in base.yaml with inline comments**

Example:
```yaml
  # Connectivity / collision
  graph_unknown_as_occ: true           # treat unknown as occupied for RR mini-A*
  graph_pred_wall_threshold: 0.7       # legacy hard-block for predicted walls (RR/legacy)
  graph_pred_hard_wall_threshold: 0.95 # hard-block ghost edges only when pred is very high
  graph_mini_astar_ttl_steps: 10       # cache TTL for mini-A* checks
```

**Step 2: Pass new param through explore.py**

```python
"graph_pred_hard_wall_threshold": nbh_cfg.get("graph_pred_hard_wall_threshold", 0.95),
```

**Step 3: Implement ghost edge check (no mini-A*)**

- Add a helper to compute edge strip samples for obs_map/pred_mean.
- In `_check_ghost_edge`, replace mini-A* with:
  - hard-block if any `obs_map >= 0.8` in strip.
  - hard-block if any `pred_mean >= graph_pred_hard_wall_threshold` in strip.
  - otherwise return ok and let risk cost drive path selection.

**Step 4: Add RR cache with wall invalidation**

- Add `self._rr_edge_cache = {}` in `__init__`.
- Update `_check_rr_edge` to:
  - return cached OK if stored clearance samples are still wall-free.
  - invalidate cached OK if new walls intersect samples.
  - on first OK, store samples (LOS lines and/or mini-A* path points).
- Keep negative edges uncached (or cache only OK edges).

### Task 3: Run tests and ensure green

Run:
- `pytest tests/test_graph_ghost_edge_policy.py -v`
- `pytest tests/test_graph_connectivity_policy.py -v`
- `pytest tests/test_graph_cache_invalidation.py -v`

Expected: PASS

### Task 4: Commit

```bash
git add configs/base.yaml nbh/graph_utils.py nbh/explore.py tests/test_graph_ghost_edge_policy.py

git commit -m "feat: add hard-threshold ghost edges and cache RR connectivity"
```
