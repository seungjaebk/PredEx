from omegaconf import OmegaConf


def test_base_config_has_graph_grid_policy():
    cfg = OmegaConf.load("configs/base.yaml")
    assert "graph_grid_policy" in cfg.nbh
