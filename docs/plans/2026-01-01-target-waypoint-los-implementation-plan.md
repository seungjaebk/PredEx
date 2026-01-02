# Target Waypoint LOS Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update exploration flow to refresh propagated values each timestep and pick the farthest line-of-sight waypoint on the BFS path without forcing every intermediate centroid.

**Architecture:** Keep high-level target locking unchanged. Refresh graph diffusion every timestep using the latest prediction variance. When selecting a new waypoint, walk the BFS path and choose the farthest cell whose centroid is visible from the current pose; fall back to existing path/greedy logic if LOS fails.

**Tech Stack:** Python, numpy, existing nbh graph/flow utilities.

### Task 1: Add LOS waypoint selection helper

**Files:**
- Modify: `nbh/explore.py`
- Test: `tests/test_waypoint_los.py`

**Step 1: Write the failing test**

```python
import numpy as np
from nbh.explore import pick_farthest_los_cell


def test_pick_farthest_los_cell_skips_blocked_cells():
    obs = np.zeros((10, 10), dtype=np.float32)
    obs[4, 4:8] = 1.0  # wall line
    path_cells = [
        type("Cell", (), {"center": np.array([2, 2])})(),
        type("Cell", (), {"center": np.array([3, 3])})(),
        type("Cell", (), {"center": np.array([5, 5])})(),  # blocked by wall
        type("Cell", (), {"center": np.array([7, 7])})(),
    ]
    cur_pose = np.array([2, 2])
    chosen = pick_farthest_los_cell(cur_pose, path_cells, obs)
    assert np.allclose(chosen.center, np.array([3, 3]))
```

**Step 2: Run test to verify it fails**

Run:
```
source /home/seungjab/anaconda3/etc/profile.d/conda.sh && CONDA_NO_PLUGINS=true conda activate nbh && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_waypoint_los.py::test_pick_farthest_los_cell_skips_blocked_cells -v
```
Expected: FAIL (import error / function missing)

**Step 3: Write minimal implementation**

```python
def pick_farthest_los_cell(cur_pose, path_cells, obs_map):
    last_visible = None
    for cell in path_cells:
        if check_line_of_sight(cur_pose, cell.center, obs_map):
            last_visible = cell
        else:
            break
    return last_visible
```

**Step 4: Run test to verify it passes**

Run:
```
source /home/seungjab/anaconda3/etc/profile.d/conda.sh && CONDA_NO_PLUGINS=true conda activate nbh && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_waypoint_los.py::test_pick_farthest_los_cell_skips_blocked_cells -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_waypoint_los.py nbh/explore.py
git commit -m "feat: add farthest-LOS waypoint helper"
```

### Task 2: Refresh propagated values each timestep

**Files:**
- Modify: `nbh/explore.py`
- Test: `tests/test_graph_diffuse.py`

**Step 1: Write the failing test**

```python
import numpy as np
from nbh.graph_utils import CellManager


def test_diffuse_scent_updates_each_step():
    cm = CellManager(cell_size=5)
    obs = np.zeros((20, 20), dtype=np.float32)
    pred_var = np.zeros_like(obs)
    pred_var[10, 10] = 1.0

    cm.update_graph(np.array([10, 10]), obs, pred_mean_map=None, pred_var_map=pred_var)
    cm.diffuse_scent(pred_var)
    first = [n.propagated_value for n in cm.cells.values()]

    pred_var[10, 10] = 0.5
    cm.update_graph(np.array([10, 10]), obs, pred_mean_map=None, pred_var_map=pred_var)
    cm.diffuse_scent(pred_var)
    second = [n.propagated_value for n in cm.cells.values()]

    assert first != second
```

**Step 2: Run test to verify it fails**

Run:
```
source /home/seungjab/anaconda3/etc/profile.d/conda.sh && CONDA_NO_PLUGINS=true conda activate nbh && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_graph_diffuse.py::test_diffuse_scent_updates_each_step -v
```
Expected: FAIL (update not wired in explore loop)

**Step 3: Write minimal implementation**

```python
# After cell_manager.update_graph(...)
if unpadded_pred_var is not None:
    cell_manager.diffuse_scent(unpadded_pred_var)
```

**Step 4: Run test to verify it passes**

Run:
```
source /home/seungjab/anaconda3/etc/profile.d/conda.sh && CONDA_NO_PLUGINS=true conda activate nbh && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_graph_diffuse.py::test_diffuse_scent_updates_each_step -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_graph_diffuse.py nbh/explore.py
git commit -m "feat: refresh graph diffusion each step"
```

### Task 3: Integrate farthest-LOS waypoint selection into flow loop

**Files:**
- Modify: `nbh/explore.py`
- Test: `tests/test_waypoint_los.py`

**Step 1: Write the failing test**

```python
import numpy as np
from nbh.explore import pick_farthest_los_cell


def test_farthest_los_prefers_los_over_next_cell():
    obs = np.zeros((10, 10), dtype=np.float32)
    path_cells = [
        type("Cell", (), {"center": np.array([2, 2])})(),
        type("Cell", (), {"center": np.array([3, 3])})(),
        type("Cell", (), {"center": np.array([4, 4])})(),
    ]
    cur_pose = np.array([2, 2])
    chosen = pick_farthest_los_cell(cur_pose, path_cells, obs)
    assert np.allclose(chosen.center, np.array([4, 4]))
```

**Step 2: Run test to verify it fails**

Run:
```
source /home/seungjab/anaconda3/etc/profile.d/conda.sh && CONDA_NO_PLUGINS=true conda activate nbh && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_waypoint_los.py::test_farthest_los_prefers_los_over_next_cell -v
```
Expected: FAIL (helper not used in main loop)

**Step 3: Write minimal implementation**

```python
# When current_waypoint is None and path_to_target exists:
los_cell = pick_farthest_los_cell(cur_pose, path_to_target, mapper.obs_map)
if los_cell is not None:
    next_cell = los_cell
```

**Step 4: Run test to verify it passes**

Run:
```
source /home/seungjab/anaconda3/etc/profile.d/conda.sh && CONDA_NO_PLUGINS=true conda activate nbh && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_waypoint_los.py::test_farthest_los_prefers_los_over_next_cell -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_waypoint_los.py nbh/explore.py
git commit -m "feat: use farthest-LOS waypoint when locking"
```

Plan complete and saved to `docs/plans/2026-01-01-target-waypoint-los-implementation-plan.md`. Two execution options:

1. Subagent-Driven (this session) - I dispatch fresh subagent per task, review between tasks, fast iteration
2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints

Which approach?
