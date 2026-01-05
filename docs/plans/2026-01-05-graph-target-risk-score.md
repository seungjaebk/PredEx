# Graph Target Risk Scoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add risk-aware target selection using `Score = V - lambda * C` with single-source Dijkstra, cache reuse for path reconstruction, and a configurable lambda.

**Architecture:** Keep `diffuse_scent` as the source of `V` (`propagated_value`). At target-reselection time, run a single-source Dijkstra from `current_cell` to get `C` for all reachable cells, compute scores, and cache `dist/prev` for reuse by `find_path_to_target`. Invalidate the cache on full graph updates.

**Tech Stack:** Python, pytest

---

### Task 1: Add failing test for risk-aware target selection

**Files:**
- Create: `tests/test_graph_target_risk_score.py`

**Step 1: Write the failing test**

```python
import numpy as np

from nbh.graph_utils import CellManager


def test_target_selection_penalizes_risk_cost():
    promotion_cfg = {
        "graph_diffuse_gamma": 0.0,
        "graph_diffuse_iterations": 0,
        "graph_target_risk_lambda": 1.0,
    }
    cm = CellManager(
        cell_size=2,
        start_pose=np.array([0.0, 0.0]),
        promotion_cfg=promotion_cfg,
    )

    start = cm.get_cell((0, 0), is_ghost=False)
    ghost_low = cm.get_cell((0, 1), is_ghost=True)
    ghost_high = cm.get_cell((0, 2), is_ghost=True)

    start.neighbors = [ghost_low]
    ghost_low.neighbors = [start, ghost_high]
    ghost_high.neighbors = [ghost_low]

    cm.edge_costs[cm._edge_key(start.index, ghost_low.index)] = 0.1
    cm.edge_costs[cm._edge_key(ghost_low.index, ghost_high.index)] = 5.0

    pred_var = np.zeros((4, 6), dtype=np.float32)
    pred_var[0, 1:3] = 0.2
    pred_var[0, 3:5] = 0.8

    target = cm.find_exploration_target(pred_var, current_cell=start)
    assert target.index == ghost_low.index
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_graph_target_risk_score.py::test_target_selection_penalizes_risk_cost -v`  
Expected: FAIL because the current selection uses only `propagated_value` and will choose `ghost_high`.

---

### Task 2: Implement risk-aware scoring + Dijkstra cache reuse

**Files:**
- Modify: `nbh/graph_utils.py`
- Modify: `nbh/explore.py`
- Modify: `configs/base.yaml`
- Test: `tests/test_graph_target_risk_score.py`

**Step 1: Add a reusable Dijkstra helper + cache**

Add a cache field in `CellManager.__init__`:

```python
        self.edge_costs = {}
        self._p_occ_map = None
        self._dijkstra_cache = None  # (start_idx, dist, prev)
```

Clear the cache in `update_graph` after map-change handling:

```python
        self._last_inflated_sig = inflated_sig
        self._dijkstra_cache = None
```

Add a helper method near `find_path_to_target`:

```python
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
```

**Step 2: Use Dijkstra in `find_exploration_target`**

Replace the BFS reachability with Dijkstra:

```python
        risk_lambda = float(self._get_cfg("graph_target_risk_lambda", 0.5))
        dist = None
        prev = None
        if current_cell is not None:
            dist, prev = self._run_dijkstra(current_cell.index)
            self._dijkstra_cache = (current_cell.index, dist, prev)

        best_score = -1e9
        best_cell = None

        for cell in self.cells.values():
            if cell.is_blocked:
                continue
            if current_cell is not None:
                if cell.index == current_cell.index:
                    continue
                cost = dist.get(cell.index) if dist is not None else None
                if cost is None:
                    continue
                score = cell.propagated_value - risk_lambda * cost
            else:
                score = cell.propagated_value

            if score > best_score:
                best_score = score
                best_cell = cell
```

Update the log line to include score and (when available) cost:

```python
            cost_str = f", cost={dist[best_cell.index]:.3f}" if dist is not None else ""
            print(f"[HIGH-LEVEL] Selected {cell_type} cell {best_cell.index} "
                  f"score={best_score:.4f}{cost_str}")
```

**Step 3: Reuse the cached Dijkstra in `find_path_to_target`**

At the top of `find_path_to_target`, after the `start/target` null checks:

```python
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
                    print(f"  [PATH] Found (cached): {path_str} "
                          f"({len(path)} cells, cost={dist[target_cell.index]:.3f})")
                    return path
```

**Step 4: Wire config into `explore.py` and `base.yaml`**

In `nbh/explore.py` `promotion_cfg`:

```python
            "graph_target_risk_lambda": nbh_cfg.get("graph_target_risk_lambda", 0.5),
```

In `configs/base.yaml` under `nbh:`:

```yaml
  # Target scoring
  graph_target_risk_lambda: 0.5
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/test_graph_target_risk_score.py::test_target_selection_penalizes_risk_cost -v`  
Expected: PASS

**Step 6: Commit**

```bash
git add tests/test_graph_target_risk_score.py nbh/graph_utils.py nbh/explore.py configs/base.yaml
git commit -m "feat: add risk-aware target scoring with cached Dijkstra"
```

---

Plan complete and saved to `docs/plans/2026-01-05-graph-target-risk-score.md`. Two execution options:

1. Subagent-Driven (this session) - I dispatch fresh subagent per task, review between tasks, fast iteration  
2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints

Which approach?
