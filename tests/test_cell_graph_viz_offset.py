import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from nbh.graph_utils import CellManager
from nbh.flow_viz_utils import visualize_cell_graph


def test_visualize_cell_graph_applies_display_offset():
    obs_map = np.zeros((10, 10), dtype=np.float32)

    cm = CellManager(
        cell_size=5,
        promotion_cfg={"graph_grid_policy": "full_map"},
    )
    cm.update_graph(
        np.array([0, 0]),
        obs_map,
        obs_map,
        pred_var_map=obs_map,
    )

    fig, ax = plt.subplots()
    display_offset = np.array([2.0, 3.0], dtype=float)

    visualize_cell_graph(
        ax,
        cm,
        obs_map=None,
        pred_mean_map=None,
        overlay_mode=True,
        show_edges=False,
        pd_size=0,
        display_offset=display_offset,
    )

    offsets = ax.collections[0].get_offsets()
    expected = []
    for cell in cm.cells.values():
        center = cell.center + display_offset
        expected.append([center[1], center[0]])

    assert len(offsets) == len(expected)
    assert all(any(np.allclose(o, e) for o in offsets) for e in expected)

    plt.close(fig)
