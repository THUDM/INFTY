from .render import plot_contour
from .solvers import SOLVER_MAP, run_trajectory
from .toy_problem import DEFAULT_INIT, DEFAULT_PROBLEM


def visualize_trajectory(optimizer_name, init=None, n_iter=2000, lr=0.1, output_dir=None, grid_size=500):
    if init is None:
        init = DEFAULT_INIT

    if optimizer_name not in SOLVER_MAP:
        raise AssertionError(f"Unsupported optimizer_name: {optimizer_name}")

    trajectory = run_trajectory(DEFAULT_PROBLEM, optimizer_name, lr, init, n_iter)
    plot_contour(DEFAULT_PROBLEM, init, trajectory, optimizer_name, output_dir=output_dir, grid_size=grid_size)
    return trajectory
