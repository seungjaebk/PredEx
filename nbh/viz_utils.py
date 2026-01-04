def should_save_graph_map(enabled):
    """Return True when graph map saving is enabled."""
    return bool(enabled)


def should_save_viz(step_idx, save_every_steps, save_on_waypoint_reached, waypoint_reached):
    if save_every_steps is None:
        save_every_steps = 0
    try:
        save_every_steps = int(save_every_steps)
    except (TypeError, ValueError):
        save_every_steps = 0
    periodic_ok = save_every_steps > 0 and (step_idx % save_every_steps == 0)
    return bool(waypoint_reached and save_on_waypoint_reached) or periodic_ok
