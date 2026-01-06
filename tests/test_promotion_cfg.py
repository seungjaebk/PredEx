from nbh.exploration_config import build_promotion_cfg


def test_build_promotion_cfg_includes_grid_policy():
    promotion_cfg = build_promotion_cfg({"graph_grid_policy": "full_map"})
    assert promotion_cfg["graph_grid_policy"] == "full_map"
