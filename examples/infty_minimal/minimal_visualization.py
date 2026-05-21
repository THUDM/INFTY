"""Minimal trajectory visualization example for INFTY.

Run from repository root:
    python examples/infty_minimal/minimal_visualization.py
"""

import matplotlib
matplotlib.use("Agg")

from infty.plot import MINIMAL_TRAJECTORY_DIR, visualize_trajectory


def main():
    trajectory = visualize_trajectory(
        optimizer_name="adam",
        n_iter=2000,
        lr=0.1,
        output_dir=MINIMAL_TRAJECTORY_DIR,
        grid_size=120,
    )
    print(f"trajectory length: {len(trajectory)}")
    print(f"plot saved under {MINIMAL_TRAJECTORY_DIR}")


if __name__ == "__main__":
    main()
