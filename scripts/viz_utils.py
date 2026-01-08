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


def compute_pad_offsets(height, width, divisor=16):
    if divisor <= 0:
        raise ValueError("divisor must be positive")

    pad_h = height % divisor
    if pad_h == 0:
        pad_h1 = 0
        pad_h2 = 0
    else:
        pad_total_h = divisor - pad_h
        pad_h1 = pad_total_h // 2
        pad_h2 = pad_total_h - pad_h1

    pad_w = width % divisor
    if pad_w == 0:
        pad_w1 = 0
        pad_w2 = 0
    else:
        pad_total_w = divisor - pad_w
        pad_w1 = pad_total_w // 2
        pad_w2 = pad_total_w - pad_w1

    return pad_h1, pad_h2, pad_w1, pad_w2
