from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "optimizer_name",
    [
        "sgd",
        "adam",
        "adamw",
        "forward_grad",
        "zo_sgd",
        "zo_sgd_q4",
        "zo_sgd_sign",
        "zo_sgd_conserve",
        "zo_adam",
        "zo_adam_q4",
        "zo_adam_sign",
        "zo_adam_conserve",
    ],
)
def test_visualize_trajectory_writes_into_output_dir(tmp_path, optimizer_name):
    pytest.importorskip("matplotlib")
    from infty.plot import visualize_trajectory

    trajectory = visualize_trajectory(optimizer_name, n_iter=3, output_dir=tmp_path, grid_size=30)
    assert trajectory.shape[0] == 3
    assert any(Path(tmp_path).rglob(f"traj_{optimizer_name}.pdf"))
