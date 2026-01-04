from nbh.viz_utils import should_save_viz


def test_should_save_viz_periodic_only():
    assert should_save_viz(0, 10, False, False)
    assert not should_save_viz(5, 10, False, False)
    assert should_save_viz(10, 10, False, False)


def test_should_save_viz_waypoint_only():
    assert should_save_viz(5, 0, True, True)
    assert not should_save_viz(5, 0, True, False)


def test_should_save_viz_combined():
    assert should_save_viz(5, 10, True, True)
    assert not should_save_viz(5, 10, True, False)
