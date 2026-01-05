# Graph Update Mode Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add configurable graph update modes with a lightweight update path and clear documentation.

**Architecture:** Introduce a small policy helper to validate/resolve graph update modes, add a lightweight update method in `CellManager`, and route update calls in `explore.py` based on the selected mode. Keep mode behavior explicit with YAML and code comments.

**Tech Stack:** Python 3.11, pytest, Hydra/OmegaConf, YAML

### Task 1: Add graph update mode helpers (config validation + policy)

**Files:**
- Modify: `nbh/exploration_config.py`
- Test: `tests/test_graph_update_mode.py`

**Step 1: Write the failing test**

```python
import pytest
from nbh.exploration_config import (
    get_graph_update_mode,
    should_run_full_update,
    should_run_light_update,
)


def test_get_graph_update_mode_default():
    assert get_graph_update_mode({}) == "target_change"


def test_get_graph_update_mode_invalid():
    with pytest.raises(ValueError):
        get_graph_update_mode({"graph_update_mode": "nope"})


def test_should_run_full_update_full():
    assert should_run_full_update("full", has_graph=False, need_new_target=False)


def test_should_run_full_update_target_change():
    assert should_run_full_update("target_change", has_graph=True, need_new_target=True)
    assert not should_run_full_update("target_change", has_graph=True, need_new_target=False)


def test_should_run_full_update_light_only():
    assert should_run_full_update("light_only", has_graph=False, need_new_target=False)
    assert not should_run_full_update("light_only", has_graph=True, need_new_target=True)


def test_should_run_light_update():
    assert should_run_light_update("target_change")
    assert should_run_light_update("light_only")
    assert not should_run_light_update("full")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_graph_update_mode.py -v`  
Expected: FAIL with import error or missing functions.

**Step 3: Write minimal implementation**

```python
GRAPH_UPDATE_MODE_DEFAULT = "target_change"
GRAPH_UPDATE_MODES = {"full", "target_change", "light_only"}


def get_graph_update_mode(nbh_cfg):
    mode = getattr(nbh_cfg, "get", lambda k, d=None: None)("graph_update_mode", None)
    if mode is None:
        mode = GRAPH_UPDATE_MODE_DEFAULT
    if mode not in GRAPH_UPDATE_MODES:
        raise ValueError(f"Invalid graph_update_mode: {mode}")
    return mode


def should_run_full_update(mode, has_graph, need_new_target):
    if mode == "full":
        return True
    if mode == "target_change":
        return bool(need_new_target)
    if mode == "light_only":
        return not bool(has_graph)
    raise ValueError(f"Invalid graph_update_mode: {mode}")


def should_run_light_update(mode):
    if mode not in GRAPH_UPDATE_MODES:
        raise ValueError(f"Invalid graph_update_mode: {mode}")
    return mode in {"target_change", "light_only"}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_graph_update_mode.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_graph_update_mode.py nbh/exploration_config.py
git commit -m "feat: add graph update mode policy helpers"
```

### Task 2: Add lightweight graph update

**Files:**
- Modify: `nbh/graph_utils.py`
- Test: `tests/test_graph_light_update.py`

**Step 1: Write the failing test**

```python
import numpy as np
from nbh.graph_utils import CellManager


def test_update_graph_light_updates_current_cell_and_visits():
    cm = CellManager(cell_size=10)

    cm.update_graph_light(np.array([0, 0], dtype=float))
    first = cm.current_cell
    assert first is not None
    assert first.index == (0, 0)
    assert first.visited
    assert first.visit_count == 1

    cm.update_graph_light(np.array([0, 0], dtype=float))
    assert cm.current_cell is first
    assert first.visit_count == 2

    cm.update_graph_light(np.array([10, 0], dtype=float))
    second = cm.current_cell
    assert second.index == (1, 0)
    assert second.parent is first
    assert cm.visited_stack[-1] is first
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_graph_light_update.py -v`  
Expected: FAIL with attribute error (missing `update_graph_light`).

**Step 3: Write minimal implementation**

```python
def update_graph_light(self, robot_pose):
    if self.origin is None:
        self.origin = np.array(robot_pose, dtype=float)
    self._mini_astar_step += 1

    curr_idx = self.get_cell_index(robot_pose)
    curr_node = self.get_cell(curr_idx)

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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_graph_light_update.py -v`  
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_graph_light_update.py nbh/graph_utils.py
git commit -m "feat: add lightweight graph update"
```

### Task 3: Wire update mode into runtime + config docs

**Files:**
- Modify: `nbh/explore.py`
- Modify: `configs/base.yaml`

**Step 1: Add config docs**

In `configs/base.yaml` under `nbh`, add `graph_update_mode` with a comment block that explains:
- `full`: full update every step
- `target_change`: light update every step, full update only when reselecting target
- `light_only`: one-time full init, then light updates only

**Step 2: Route update calls in explore**

Use `get_graph_update_mode`, `should_run_full_update`, and `should_run_light_update` to decide:
- Whether to call `update_graph` each step
- Whether to call `update_graph_light` each step
- Whether to call `update_graph` when `need_new_high_level_target` is true

**Step 3: Run full suite**

Run: `pytest`  
Expected: PASS

**Step 4: Commit**

```bash
git add nbh/explore.py configs/base.yaml
git commit -m "feat: add graph update mode wiring"
```

