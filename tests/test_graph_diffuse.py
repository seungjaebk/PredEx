import numpy as np
from nbh.graph_utils import CellManager


def test_update_graph_refreshes_propagated_values():
    cm = CellManager(cell_size=5)
    obs = np.full((20, 20), 0.5, dtype=np.float32)
    pred_mean = np.zeros_like(obs)
    pred_var = np.full_like(obs, 0.2)
    pose = np.array([10, 10])

    # First call sets origin/current cell.
    cm.update_graph(pose, obs, pred_mean_map=pred_mean, pred_var_map=pred_var)
    # Second call expands neighbors.
    cm.update_graph(pose, obs, pred_mean_map=pred_mean, pred_var_map=pred_var)

    propagated_values = [n.propagated_value for n in cm.cells.values()]
    assert any(v > 0 for v in propagated_values)
