import pytest

from scripts.config_utils import (
    get_graph_update_mode,
    should_run_full_update,
    should_run_light_update,
)


def test_get_graph_update_mode_default():
    assert get_graph_update_mode({}) == "target_change"


def test_get_graph_update_mode_override():
    assert get_graph_update_mode({"graph_update_mode": "full"}) == "full"


def test_get_graph_update_mode_invalid():
    with pytest.raises(ValueError):
        get_graph_update_mode({"graph_update_mode": "nope"})


def test_should_run_full_update_full():
    assert should_run_full_update("full", has_graph=False, need_new_target=False)
    assert should_run_full_update("full", has_graph=True, need_new_target=True)


def test_should_run_full_update_target_change():
    assert should_run_full_update("target_change", has_graph=True, need_new_target=True)
    assert not should_run_full_update("target_change", has_graph=True, need_new_target=False)


def test_should_run_full_update_light_only():
    assert should_run_full_update("light_only", has_graph=False, need_new_target=False)
    assert not should_run_full_update("light_only", has_graph=True, need_new_target=True)


def test_should_run_full_update_invalid():
    with pytest.raises(ValueError):
        should_run_full_update("nope", has_graph=False, need_new_target=False)


def test_should_run_light_update():
    assert should_run_light_update("target_change")
    assert should_run_light_update("light_only")
    assert not should_run_light_update("full")


def test_should_run_light_update_invalid():
    with pytest.raises(ValueError):
        should_run_light_update("nope")
