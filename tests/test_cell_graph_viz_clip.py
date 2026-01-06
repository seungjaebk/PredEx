import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from nbh.graph_utils import CellManager
from nbh.flow_viz_utils import visualize_cell_graph


def test_visualize_cell_graph_clips_to_display_shape():
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
    display_offset = np.array([5.0, 5.0], dtype=float)
    display_shape = (10, 10)

    visualize_cell_graph(
        ax,
        cm,
        obs_map=None,
        pred_mean_map=None,
        overlay_mode=True,
        show_edges=False,
        pd_size=0,
        display_offset=display_offset,
        display_shape=display_shape,
    )

    offsets = ax.collections[0].get_offsets()
    assert len(offsets) < len(cm.cells)
    assert all(
        (0 <= point[0] < display_shape[1]) and (0 <= point[1] < display_shape[0])
        for point in offsets
    )

    plt.close(fig)
