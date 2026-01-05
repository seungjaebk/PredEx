# Graph Update Mode Design

## Goal
Introduce a configurable graph update policy that lets us trade off compute cost vs. graph freshness. The system should support three modes: `full` (always rebuild), `target_change` (rebuild only when choosing a new target), and `light_only` (initialize once, then only update lightweight state). The behavior must be explicit in configuration and easy to reason about from code comments.

## Approach
We split the current monolithic `update_graph` call into two levels. The existing `update_graph` remains the full update: it refreshes graph structure, risk/connectivity, cache invalidation, and current cell state. We add `update_graph_light`, which only updates robot-centric state (origin initialization, current cell selection, visit counts, parent/stack bookkeeping, and step counter) without touching structure or caches. In `explore.py`, we read `graph_update_mode` from `nbh` config and route calls accordingly. `full` always calls `update_graph`; `target_change` always calls `update_graph_light` and calls `update_graph` only when the high-level target is being reselected; `light_only` calls `update_graph` once on first use and then only `update_graph_light` thereafter. This keeps the core behavior intact while providing clear switches for compute load.

## Data Flow
Observation and prediction maps are still updated every step. The graph structure update only happens on the chosen trigger. The `target_change` mode aligns with the existing high-level target selection logic: when the target is missing/blocked/reached/unreachable, we refresh the graph and then pick a new target. This ensures the costly operations are tied to the decision point that actually needs them. `light_only` explicitly chooses stale structure for maximum speed; it is intended for experiments where deterministic, stable targets are more important than immediate graph accuracy.

## Error Handling and Validation
We validate the mode string against a small allowed set to fail fast on misconfiguration. Default is `target_change`, which is the safest “balanced” option. Unit tests cover the parsing/validation helper and the lightweight update’s behavior (current cell tracking, visit counts, parent/stack changes). We avoid heavy integration tests for `explore.py` to keep the suite fast and avoid model dependencies.

