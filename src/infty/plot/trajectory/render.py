from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from matplotlib.lines import Line2D
from tqdm import tqdm

from .._common.paths import DEFAULT_TRAJECTORY_DIR, ensure_parent_dir


def plot_contour(problem, init, traj, trainer, output_dir=None, save_path=None, plotbar=False, grid_size=500):
    if output_dir is None:
        output_dir = save_path
    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_TRAJECTORY_DIR
    output_dir = output_dir.expanduser().resolve()

    n = grid_size
    xl = 11
    x = np.linspace(-xl, xl, n)
    y = np.linspace(-xl, xl, n)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.01, right=0.99)

    xs = torch.Tensor(np.transpose(np.array([list(X.flat), list(Y.flat)]))).double()
    ys = problem.batch_forward(xs)

    yy = -8.3552
    yv = ys.mean(1)

    plt.plot(init[0], init[1], marker="o", markersize=10, zorder=20, color="k")
    plt.plot(0, yy, marker="*", markersize=15, zorder=5, color="k")
    plt.plot(7, yy, marker="None")
    for offset in [-0.2, 0, 0.2]:
        plt.gca().add_line(Line2D([7 - offset, 7 + offset], [yy - offset, yy + offset], color="r", linewidth=5, zorder=5))
        plt.gca().add_line(Line2D([7 + offset, 7 - offset], [yy - offset, yy + offset], color="r", linewidth=5, zorder=5))
    plt.plot(-7, yy, marker="None")
    for offset in [-0.2, 0, 0.2]:
        plt.gca().add_line(Line2D([-7 - offset, -7 + offset], [yy - offset, yy + offset], color="b", linewidth=5, zorder=5))
        plt.gca().add_line(Line2D([-7 + offset, -7 - offset], [yy - offset, yy + offset], color="b", linewidth=5, zorder=5))

    contour = plt.contour(X, Y, yv.view(n, n), cmap=cm.viridis, linewidths=4.0)

    fontsize = 36
    if traj is not None:
        for points in tqdm(traj):
            ax.scatter(points[0], points[1], color=cm.viridis(points.shape[0]), s=6, zorder=10)

    if plotbar:
        colorbar = fig.colorbar(contour, ticks=[-15, -10, -5, 0, 5])
        colorbar.ax.tick_params(labelsize=fontsize)

    plt.xticks([-10, -5, 0, 5, 10], fontsize=fontsize, fontfamily="serif")
    plt.yticks([-10, -5, 0, 5, 10], fontsize=fontsize, fontfamily="serif")

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("serif")
    ax.tick_params(axis="both", labelsize=fontsize, which="major", direction="out", length=6, width=2)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(2)

    plt.tight_layout()
    figure_path = output_dir / f"traj_{trainer}.pdf"
    ensure_parent_dir(figure_path)
    plt.savefig(figure_path)
    plt.close(fig)
    return str(figure_path)
