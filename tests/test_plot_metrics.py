from pathlib import Path

import numpy as np
import pytest
import torch

from infty.plot.esd import summarize_esd
from infty.plot.landscape import summarize_loss_surface


def test_summarize_loss_surface_on_parabola():
    lams_alpha = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    lams_beta = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    alpha_grid, beta_grid = np.meshgrid(lams_alpha, lams_beta, indexing="ij")
    Z = alpha_grid ** 2 + beta_grid ** 2

    summary = summarize_loss_surface(Z, lams_alpha, lams_beta, eigenvalues=[1.0, 2.0], taus=(0.5, 1.0))

    assert summary["loss_center"] == pytest.approx(0.0)
    assert summary["r0_sharpness"] == pytest.approx(2.0)
    assert summary["relative_r0_sharpness"] == pytest.approx(2.0 / 1e-12, rel=1e-12)
    assert summary["basin_fraction_tau_0.5"] == pytest.approx(1.0 / 9.0)
    assert summary["basin_fraction_tau_1.0"] == pytest.approx(5.0 / 9.0)
    assert summary["center_fd_trace_2d"] == pytest.approx(4.0)
    assert summary["lambda1"] == pytest.approx(2.0)
    assert summary["lambda2"] == pytest.approx(1.0)


def test_summarize_esd_metrics():
    eigenvalues = np.array([[-2.0, 0.0, 3.0]], dtype=np.float64)
    weights = np.array([[0.2, 0.3, 0.5]], dtype=np.float64)

    summary = summarize_esd(eigenvalues, weights, trace=5.0, right_tail_threshold=1.0)

    assert summary["lambda_max"] == pytest.approx(3.0)
    assert summary["lambda_min"] == pytest.approx(-2.0)
    assert summary["mean_eigenvalue"] == pytest.approx(1.1)
    assert summary["negative_mass"] == pytest.approx(0.2)
    assert summary["positive_mass"] == pytest.approx(0.8)
    assert summary["right_tail_mass"] == pytest.approx(0.5)
    assert summary["spectral_entropy"] == pytest.approx(-(0.2 * np.log(0.2) + 0.3 * np.log(0.3) + 0.5 * np.log(0.5)))
    assert summary["effective_rank"] == pytest.approx(np.exp(summary["spectral_entropy"]))
    assert summary["mean_trace"] == pytest.approx(5.0)


def test_unigrad_fs_records_conflict_metrics_and_plots(tmp_path):
    pytest.importorskip("matplotlib")
    from infty.optim import UniGrad_FS
    from infty.plot import visualize_conflicts

    torch.manual_seed(0)
    model = torch.nn.Linear(1, 1, bias=False)
    base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer = UniGrad_FS(
        model.parameters(),
        base_optimizer=base_optimizer,
        model=model,
        args={"task_id": 1, "utype": "model-wise", "S_T": [0.1]},
    )

    inputs = torch.ones(4, 1)

    def closure():
        pred = model(inputs)
        loss1 = pred.mean()
        loss2 = -0.5 * pred.mean()
        return pred, [loss1, loss2]

    optimizer.set_closure(closure)
    optimizer.step(delay=True)

    assert len(optimizer.conflict_records) == 1
    record = optimizer.conflict_records[0]
    assert record["task"] == 1
    assert record["iter"] == 0
    assert record["conflict"] is True
    assert record["below_threshold"] is True
    assert record["cos_after"] >= record["cos_before"]
    assert "gain_old" in record
    assert "gain_new" in record
    assert "min_gain" in record

    result = visualize_conflicts(optimizer, task=1, output_dir=tmp_path)
    assert Path(result["sim_path"]).exists()
    assert Path(result["plot_path"]).exists()
