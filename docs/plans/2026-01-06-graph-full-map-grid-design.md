# Graph Full-Map Grid Design

**Goal:** Represent the cell graph over the full known map extent (obs_map size) with absolute indexing, removing frontier-based expansion and origin-relative indices. Keep Unknown cells out of the graph; only Real and Ghost nodes are created.

## Decisions
- **Indexing:** Absolute grid indexing. Cell (0, 0) is the map’s top-left. `r = floor(row / cell_size)`, `c = floor(col / cell_size)`. Indices are non-negative.
- **Grid policy:** Add `graph_grid_policy` with values:
  - `full_map`: scan the entire map grid each full update.
  - `frontier_expand`: keep existing `graph_max_ghost_distance` expansion.
- **Unknown handling:** Unknown cells are not nodes. Ghost nodes are created only where unknown + low pred_mean + low pred_var.
- **Map alignment:** `pred_mean_map`/`pred_var_map` are aligned to `obs_map` size with `align_pred_map`. Graph logic always uses obs_map as the canonical size.

## Data Flow
1. Compute grid dimensions from `obs_map.shape` and `cell_size`.
2. For each cell in the grid, compute patch bounds (clipped to map).
3. Determine node type in order:
   - Blocked: obs blocked ratio or centroid blocked threshold.
   - Real: unknown_ratio <= threshold.
   - Ghost: unknown + pred_mean free + pred_var below threshold.
   - Unknown: skip node creation.
4. Connect neighbors with existing RR/RG/GG checks and edge risk costs.

## Visualization Alignment
- In run_viz, draw the Observed Map using `padded_obs_map` so its display region matches the Mean Map (which is padded to 16x divisor). This removes the size mismatch between obs_map and pred_mean_map in figures.

## Risks / Performance
- Full-grid scanning is more expensive than frontier expansion. Keep `frontier_expand` for comparison, and consider caching per-cell patch bounds if needed.

## Testing
- Unit: absolute `get_cell_index` and `get_cell_center`.
- Unit: `full_map` grid range covers map bounds without negative indices.
- Regression: `frontier_expand` behavior unchanged.
- Visual sanity: Observed Map and Mean Map panels share the same display size.
