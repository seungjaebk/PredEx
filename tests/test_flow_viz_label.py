from nbh.flow_viz_utils import get_graph_path_label


def test_graph_path_label_is_risk_path():
    assert get_graph_path_label() == "Risk Path"
