import numpy as np

from nbh.graph_utils import CellManager


def test_edge_strip_stats_pred_and_pocc_max():
    cm = CellManager(cell_size=10, start_pose=np.array([10, 10]))
    cm.connectivity_cfg["graph_edge_risk_strip_divisor"] = 2

    pred = np.zeros((30, 30), dtype=np.float32)
    p_occ = np.zeros((30, 30), dtype=np.float32)

    pred[5:15, 10:15] = 0.9
    p_occ[5:15, 15:20] = 0.7

    cm.edge_costs = {((0, 0), (0, 1)): 0.1}
    stats = cm._edge_strip_stats(pred, p_occ)

    assert stats["pred_max"].size == 1
    assert stats["pocc_max"].size == 1
    assert np.isclose(stats["pred_max"][0], 0.9)
    assert np.isclose(stats["pocc_max"][0], 0.7)
