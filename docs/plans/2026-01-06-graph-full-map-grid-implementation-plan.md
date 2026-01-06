# Graph Full-Map Grid Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Switch the cell graph to absolute indexing and full-map grid scanning (with a policy toggle), and align Observed Map visualization to the padded prediction map size.

**Architecture:** Add a `graph_grid_policy` config to switch between `frontier_expand` and `full_map`. Update `CellManager` to use absolute indices and full-map scanning when requested, while retaining current connectivity logic. Align run_viz Observed Map to padded map size using shared padding offsets.

**Tech Stack:** Python, NumPy, Matplotlib

---

### Task 1: Absolute indexing for cells

**Files:**
- Modify: `nbh/graph_utils.py`
- Test: `tests/test_graph_absolute_indexing.py`

**Step 1: Write the failing test**

Create `tests/test_graph_absolute_indexing.py`:
```python
import numpy as np
from nbh.graph_utils import CellManager


def test_get_cell_index_absolute():
    cm = CellManager(cell_size=10)
    assert cm.get_cell_index((0, 0)) == (0, 0)
    assert cm.get_cell_index((9, 9)) == (0, 0)
    assert cm.get_cell_index((10, 0)) == (1, 0)
    assert cm.get_cell_index((0, 10)) == (0, 1)


def test_get_cell_center_absolute():
    cm = CellManager(cell_size=10)
    assert np.allclose(cm.get_cell_center((0, 0)), [5, 5])
    assert np.allclose(cm.get_cell_center((1, 2)), [15, 25])
```

**Step 2: Run test to verify it fails**

Run:
```
eval "$(conda shell.posix activate nbh)" && PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_graph_absolute_indexing.py -v
```
Expected: FAIL with current origin-based indexing.

**Step 3: Write minimal implementation**

Update `nbh/graph_utils.py`:
- `get_cell_index`: compute `r = floor(row / cell_size)`, `c = floor(col / cell_size)`; remove origin requirement.
- `get_cell_center`: return `[(r + 0.5) * cell_size, (c + 0.5) * cell_size]`.
- Update docstrings/comments to reflect absolute indexing.
- Keep `origin` field but note it is unused (or remove origin checks where they gate execution).

**Step 4: Run test to verify it passes**

Run:
```
eval "$(conda shell.posix activate nbh)" && PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_graph_absolute_indexing.py -v
```
Expected: PASS

**Step 5: Commit**

```
git add nbh/graph_utils.py tests/test_graph_absolute_indexing.py
git commit -m "feat: use absolute indexing for cell grid"
```

---

### Task 2: Add full-map grid policy

**Files:**
- Modify: `configs/base.yaml`
- Modify: `nbh/explore.py`
- Modify: `nbh/graph_utils.py`
- Test: `tests/test_graph_grid_policy.py`

**Step 1: Write the failing test**

Create `tests/test_graph_grid_policy.py`:
```python
import numpy as np
from nbh.graph_utils import CellManager


def test_full_map_grid_creates_all_real_cells():
    obs_map = np.zeros((20, 20), dtype=np.float32)
    pred_mean = np.zeros_like(obs_map)
    pred_var = np.zeros_like(obs_map)

    cm = CellManager(
        cell_size=10,
        promotion_cfg={"graph_grid_policy": "full_map"},
    )
    cm.update_graph(
        np.array([0, 0]),
        obs_map,
        pred_mean,
        pred_var_map=pred_var,
    )

    assert len(cm.cells) == 4
    assert all((not c.is_ghost) and (not c.is_blocked) for c in cm.cells.values())
```

**Step 2: Run test to verify it fails**

Run:
```
eval "$(conda shell.posix activate nbh)" && PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_graph_grid_policy.py -v
```
Expected: FAIL (no full-map mode).

**Step 3: Write minimal implementation**

Update `configs/base.yaml` (under `nbh:`):
```
graph_grid_policy: full_map  # full_map or frontier_expand
```

Update `nbh/explore.py`:
- Add `graph_grid_policy` to `promotion_cfg`.

Update `nbh/graph_utils.py`:
- Read `graph_grid_policy` via `_get_cfg`.
- In `update_graph_structure`, if `full_map`:
  - Compute `rows = ceil(H / cell_size)`, `cols = ceil(W / cell_size)`.
  - Set `potential_indices` to all `(r, c)` in range.
  - Skip ghost-distance gating (or only apply when policy is `frontier_expand`).
- In `frontier_expand` mode:
  - Keep existing expansion, but clamp indices to map bounds so they stay non-negative and within the grid.

**Step 4: Run test to verify it passes**

Run:
```
eval "$(conda shell.posix activate nbh)" && PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_graph_grid_policy.py -v
```
Expected: PASS

**Step 5: Commit**

```
git add configs/base.yaml nbh/explore.py nbh/graph_utils.py tests/test_graph_grid_policy.py
git commit -m "feat: add full-map grid policy for graph expansion"
```

---

### Task 3: Align Observed Map visualization to padded map size

**Files:**
- Modify: `nbh/viz_utils.py`
- Modify: `nbh/explore.py`
- Test: `tests/test_viz_pad_offsets.py`

**Step 1: Write the failing test**

Create `tests/test_viz_pad_offsets.py`:
```python
from nbh.viz_utils import compute_pad_offsets


def test_compute_pad_offsets_16_divisor():
    pad_h1, pad_h2, pad_w1, pad_w2 = compute_pad_offsets(1325, 1428, divisor=16)
    assert (pad_h1, pad_h2) == (1, 2)
    assert (pad_w1, pad_w2) == (6, 6)
```

**Step 2: Run test to verify it fails**

Run:
```
eval "$(conda shell.posix activate nbh)" && PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_viz_pad_offsets.py -v
```
Expected: FAIL (helper missing).

**Step 3: Write minimal implementation**

Update `nbh/viz_utils.py`:
- Add `compute_pad_offsets(height, width, divisor=16)` that mirrors existing pad logic.

Update `nbh/explore.py`:
- Replace inline pad_h1/pad_h2/pad_w1/pad_w2 calculation with `compute_pad_offsets`.
- In run_viz Observed Map panel, use `padded_obs_map` (if available) as the image source.
- Apply `pad_h1/pad_w1` offsets consistently when subtracting `pd_size` for obs-panel overlays.

**Step 4: Run test to verify it passes**

Run:
```
eval "$(conda shell.posix activate nbh)" && PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_viz_pad_offsets.py -v
```
Expected: PASS

**Step 5: Commit**

```
git add nbh/viz_utils.py nbh/explore.py tests/test_viz_pad_offsets.py
git commit -m "feat: align observed map viz to padded size"
```

---

Plan complete and saved to `docs/plans/2026-01-06-graph-full-map-grid-implementation-plan.md`. Two execution options:

1. Subagent-Driven (this session) – I dispatch fresh subagent per task, review between tasks.
2. Parallel Session (separate) – open a new session and run with executing-plans.

Which approach?
