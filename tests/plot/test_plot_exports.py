import pytest


def test_plot_exports():
    pytest.importorskip("matplotlib")
    import infty.plot as plot_api

    assert hasattr(plot_api, "visualize_landscape")
    assert hasattr(plot_api, "visualize_loss_landscape")
    assert hasattr(plot_api, "visualize_esd")
    assert hasattr(plot_api, "visualize_conflicts")
    assert hasattr(plot_api, "visualize_trajectory")
