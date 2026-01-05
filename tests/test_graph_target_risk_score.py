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
