from pathlib import Path

from infty.plot import (
    CUSTOM_PLOTS_DIR,
    DEFAULT_CONFLICTS_DIR,
    DEFAULT_ESD_DIR,
    DEFAULT_LANDSCAPE_DIR,
    DEFAULT_TRAJECTORY_DIR,
    EXPERIMENTS_ROOT,
    MINIMAL_TRAJECTORY_DIR,
    PLOT_ROOT,
    REPO_ROOT,
)


def test_plot_hierarchy_constants_match_expected_layout():
    assert REPO_ROOT.name != "src"
    assert DEFAULT_CONFLICTS_DIR.relative_to(PLOT_ROOT) == Path("diagnostics/conflicts")
    assert DEFAULT_ESD_DIR.relative_to(PLOT_ROOT) == Path("diagnostics/esd")
    assert DEFAULT_LANDSCAPE_DIR.relative_to(PLOT_ROOT) == Path("diagnostics/landscape")
    assert DEFAULT_TRAJECTORY_DIR.relative_to(PLOT_ROOT) == Path("diagnostics/trajectory")
    assert MINIMAL_TRAJECTORY_DIR.relative_to(PLOT_ROOT) == Path("examples/infty_minimal/trajectory")
    assert EXPERIMENTS_ROOT.relative_to(PLOT_ROOT) == Path("experiments")
    assert CUSTOM_PLOTS_DIR.relative_to(PLOT_ROOT) == Path("custom")
