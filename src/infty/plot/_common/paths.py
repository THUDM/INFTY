"""Canonical plot output locations for INFTY runtime artifacts.

The plot tree is organized by producer first, then by plot type:

workdirs/plots/
    diagnostics/
        conflicts/
        esd/
        landscape/
        trajectory/
    examples/
        infty_minimal/
            trajectory/
    pilot/
        analyses/
    experiments/
    custom/
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
PLOT_ROOT = REPO_ROOT / "workdirs" / "plots"


def plot_dir(*parts):
    return PLOT_ROOT.joinpath(*parts)


def ensure_parent_dir(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


DIAGNOSTICS_ROOT = plot_dir("diagnostics")
DEFAULT_CONFLICTS_DIR = plot_dir("diagnostics", "conflicts")
DEFAULT_ESD_DIR = plot_dir("diagnostics", "esd")
DEFAULT_LANDSCAPE_DIR = plot_dir("diagnostics", "landscape")
DEFAULT_TRAJECTORY_DIR = plot_dir("diagnostics", "trajectory")

EXAMPLES_ROOT = plot_dir("examples")
MINIMAL_TRAJECTORY_DIR = plot_dir("examples", "infty_minimal", "trajectory")

PILOT_ROOT = plot_dir("pilot")
PILOT_ANALYSIS_DIR = plot_dir("pilot", "analyses")

EXPERIMENTS_ROOT = plot_dir("experiments")
CUSTOM_PLOTS_DIR = plot_dir("custom")

DEFAULT_PLOT_DIRS = {
    "conflicts": DEFAULT_CONFLICTS_DIR,
    "esd": DEFAULT_ESD_DIR,
    "landscape": DEFAULT_LANDSCAPE_DIR,
    "trajectory": DEFAULT_TRAJECTORY_DIR,
}


__all__ = [
    "REPO_ROOT",
    "PLOT_ROOT",
    "plot_dir",
    "ensure_parent_dir",
    "DIAGNOSTICS_ROOT",
    "DEFAULT_CONFLICTS_DIR",
    "DEFAULT_ESD_DIR",
    "DEFAULT_LANDSCAPE_DIR",
    "DEFAULT_TRAJECTORY_DIR",
    "EXAMPLES_ROOT",
    "MINIMAL_TRAJECTORY_DIR",
    "PILOT_ROOT",
    "PILOT_ANALYSIS_DIR",
    "EXPERIMENTS_ROOT",
    "CUSTOM_PLOTS_DIR",
    "DEFAULT_PLOT_DIRS",
]
