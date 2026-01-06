import numpy as np

from nbh.graph_utils import CellManager


def test_get_cell_index_absolute():
    cm = CellManager(cell_size=10)
    assert cm.get_cell_index((0, 0)) == (0, 0)
    assert cm.get_cell_index((9, 9)) == (0, 0)
    assert cm.get_cell_index((10, 0)) == (1, 0)
    assert cm.get_cell_index((0, 10)) == (0, 1)


def test_get_cell_center_absolute():
    cm = CellManager(cell_size=10)
    assert np.allclose(cm.get_cell_center((0, 0)), [5, 5])
    assert np.allclose(cm.get_cell_center((1, 2)), [15, 25])
