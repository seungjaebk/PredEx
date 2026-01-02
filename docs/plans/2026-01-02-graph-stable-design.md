# Graph Stable Design

Date: 2026-01-02

## Summary
Create a new branch `graph_stable` based on `graph_fixed` that combines the best stability elements from main/baseline/graph_fixed/graph_prob. The implementation minimizes refactors, keeps existing structures (CellManager, Mapper, p_occ pipeline), and centralizes configuration in `configs/base.yaml` under `nbh.*`.

## Goals
- Reduce oscillation and false disconnects by unifying connectivity and prediction use.
- Keep hard blockers strictly from `obs_map` (prediction never blocks).
- Apply prediction only to ghost creation/maintenance and mini-A* cost grids.
- Add robust mini-A* caching with explicit invalidation on map changes.
- Gate all debug output behind config flags.

## Non-Goals
- Large refactors or architectural changes.
- Changing cell indexing (no `np.round` baseline style).
- New external dependencies.

## Invariants
- Hard blockers (blocked/unknown classification) use `obs_map` only.
- Prediction is used only for: (a) ghost creation/maintenance, (b) mini-A* cost grid.
- Ghost-related decisions are patch-based or mini-A* based (avoid centroid LOS drift).
- Cached mini-A* results must be invalidated when the map changes.

## Config Layout (base.yaml)
All new parameters live under `nbh.*` and are injected from `explore.py`.

Example structure:
- `nbh.graph.*` (connectivity, cache, portal policy)
- `nbh.ghost.*` (ghost creation thresholds)
- `nbh.debug.*` (debug flags)

## Architecture and Data Flow
- `explore.py` reads Hydra options via `get_options_dict_from_yml` and builds compact dicts:
  - `promotion_cfg` (ghost creation thresholds)
  - `connectivity_cfg` (portal, LOS/mini-A* policy, cache TTL)
  - `debug_cfg` (edge/target/belief/progression logs)
- `CellManager` receives these dicts and does not access Hydra directly.
- `align_pred_map` remains in `explore.py` (graph_fixed behavior).

## Connectivity Rules
- RR (real-real): portal-first; if portal True and LOS False, run mini-A* fallback.
  - If portal False, default skip, but allow conditional mini-A* fallback to reduce portal false negatives.
- RG/GG (ghost involved): use mini-A* only. No LOS gating.
- Prediction wall threshold is applied only in mini-A* cost grid generation.

## Mini-A* Cache Strategy
- Cache key: `(cell_a, cell_b, edge_type, grid_policy_hash)`.
- `grid_policy_hash` includes:
  - `unknown_as_occ`
  - `pred_wall_threshold`
  - `dilate_diam`
  - `obs_update_id` (or grid checksum/version)
  - `pred_update_id` (if available)
- Invalidation:
  - TTL expiry
  - Inflated grid regenerated or version changes
  - `obs_map` change ratio > delta threshold

## Debug Logging (Config-Gated)
- `debug.edges` prints per edge creation:
  - `edge_type(RR/RG/GG)`, `portal(T/F)`, `LOS(T/F)`, `miniA*(hit/miss, path_len, fail_reason)`, `cache(hit/miss)`, `ttl_left`
- `debug.targets` prints target reselect reasons and unreachable cause.
- No always-on debug prints.

## Test Plan (pytest)
Add at least three targeted tests:
1) Thin wall: patch mean looks free but a blocking strip exists. Expect edge rejection.
2) Portal false negative: `portal=False` but mini-A* finds a path. Expect fallback allows edge.
3) Cache stale: edge cached True, then obs update blocks it. Expect invalidation -> edge False.

## Delivery Plan (Small PR Steps)
1) Config injection: base.yaml + option plumbing. Run `pytest -q`.
2) Connectivity refactor: RR/RG/GG rules + cost grid unification. Run `pytest -q`.
3) Cache/TTL + map invalidation: add policy hash and versioning. Run `pytest -q`.
4) Debug logs + tests: gate logs, add tests. Run `pytest -q`.

## Risks and Mitigations
- Semantics drift: unify ghost checks via patch metrics and mini-A* cost grid only.
- Portal false negatives: conditional mini-A* fallback when portal fails.
- Cache stale: policy hash + map change detection; TTL is secondary.
