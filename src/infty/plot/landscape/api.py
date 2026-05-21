import copy
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import tqdm

from infty.utils.hessian import hessian
from .._common.paths import DEFAULT_LANDSCAPE_DIR, ensure_parent_dir
from .._common.source import resolve_source_name


DEFAULT_OUTPUT_DIR = DEFAULT_LANDSCAPE_DIR


def summarize_loss_surface(Z, lams_alpha, lams_beta, eigenvalues=None, taus=(0.01, 0.05, 0.1)):
    Z = np.asarray(Z, dtype=np.float64)
    lams_alpha = np.asarray(lams_alpha, dtype=np.float64)
    lams_beta = np.asarray(lams_beta, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError(f"Z must be a 2D array, got shape {Z.shape}.")
    if Z.shape[0] != lams_alpha.size or Z.shape[1] != lams_beta.size:
        raise ValueError(
            "Z shape must match the lengths of lams_alpha and lams_beta, "
            f"got {Z.shape}, {lams_alpha.size}, and {lams_beta.size}."
        )

    ci, cj = Z.shape[0] // 2, Z.shape[1] // 2
    z0 = float(Z[ci, cj])
    summary = {
        "loss_center": z0,
        "loss_min": float(Z.min()),
        "loss_max": float(Z.max()),
        "r0_sharpness": float(Z.max() - z0),
        "relative_r0_sharpness": float((Z.max() - z0) / (abs(z0) + 1e-12)),
    }
    for tau in taus:
        summary[f"basin_fraction_tau_{tau}"] = float(np.mean(Z <= z0 + float(tau)))

    if Z.shape[0] >= 3 and Z.shape[1] >= 3:
        da = float(lams_alpha[1] - lams_alpha[0])
        db = float(lams_beta[1] - lams_beta[0])
        d2a = (Z[ci + 1, cj] - 2 * z0 + Z[ci - 1, cj]) / (da * da)
        d2b = (Z[ci, cj + 1] - 2 * z0 + Z[ci, cj - 1]) / (db * db)
        summary["center_fd_trace_2d"] = float(d2a + d2b)

    if eigenvalues is not None:
        eigenvalues = np.sort(np.asarray(eigenvalues, dtype=np.float64).reshape(-1))
        if eigenvalues.size >= 1:
            summary["lambda1"] = float(eigenvalues[-1])
        if eigenvalues.size >= 2:
            summary["lambda2"] = float(eigenvalues[-2])

    return summary


def get_params(model_ori, model_perb, direction, alpha, beta, device):
    for m_orig, m_perb, d0, d1 in zip(model_ori.parameters(), model_perb.parameters(), direction[0], direction[1]):
        if m_orig.data.shape == d0.shape:
            m_perb.data = m_orig.data + alpha * d0.to(device) + beta * d1.to(device)
    return model_perb


def visualize_loss_landscape(
    optimizer,
    model,
    create_loss_fn,
    loader,
    task,
    device,
    limit=0.3,
    samples=11,
    eigen_max_iter=100,
    eigen_tol=1e-3,
    output_dir=None,
    dir_path=None,
    source_name=None,
):
    output_dir = Path(output_dir) if output_dir is not None else Path(dir_path) if dir_path is not None else DEFAULT_OUTPUT_DIR
    output_dir = output_dir.expanduser().resolve()
    optimizer_name = resolve_source_name(source=optimizer, source_name=source_name, default="custom")
    save_dir = output_dir / optimizer_name

    lams_alpha = np.linspace(-limit, limit, samples).astype(np.float32)
    lams_beta = np.linspace(-limit, limit, samples).astype(np.float32)
    total_loss_list = []

    was_training = model.training
    state_backup = {k: v.detach().clone() for k, v in model.state_dict().items()}

    try:
        model.eval()

        eigen_path = save_dir / f"eigen_task{task}.pt"
        if eigen_path.exists():
            print(f"\n[Hessian] Loading existing eigenvectors for task {task} ...")
            eigen_data = torch.load(eigen_path, map_location=device)
            top_eigenvalues = eigen_data[f"top_eigenvalues_task{task}"]
            top_eigenvector = eigen_data[f"top_eigenvector_task{task}"]
            print(f"[Hessian] Loaded. Top eigenvalues: λ1 = {top_eigenvalues[-1]:.4f}, λ2 = {top_eigenvalues[-2]:.4f}")
        else:
            print(f"\n[Hessian] Computing top-2 eigenvectors for task {task} ...")
            start_time = time.time()
            hessian_comp = hessian(model, create_loss_fn, dataloader=loader, device=device)
            top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(
                maxIter=eigen_max_iter,
                tol=eigen_tol,
                top_n=2,
            )
            elapsed_time = time.time() - start_time
            print(f"[Hessian] Done. Time elapsed: {elapsed_time:.2f} seconds.")
            print(f"[Hessian] Top eigenvalues: λ1 = {top_eigenvalues[-1]:.4f}, λ2 = {top_eigenvalues[-2]:.4f}")
            torch.save(
                {
                    f"top_eigenvalues_task{task}": top_eigenvalues,
                    f"top_eigenvector_task{task}": top_eigenvector,
                },
                ensure_parent_dir(eigen_path),
            )

        loss_path = save_dir / f"loss_list_task{task}.pt"
        if loss_path.exists():
            print(f"\n[Landscape] Loading existing loss surface for task {task} ...")
            loss_data = torch.load(loss_path, map_location=device)
            total_loss_list = loss_data[f"loss_list_task{task}"]
            print(f"[Landscape] Loaded loss surface from {loss_path}")
        else:
            print(f"\n[Landscape] Sampling {samples}x{samples} grid on loss surface ...")
            model_perb = copy.deepcopy(model)
            model_perb.eval()

            for alpha in tqdm.tqdm(lams_alpha, desc="Alpha axis"):
                row_loss_list = []
                for beta in lams_beta:
                    model_perb = get_params(model, model_perb, top_eigenvector, alpha, beta, device)
                    total_loss = 0.0
                    for _, inputs, targets in loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        loss_fn = create_loss_fn(inputs, targets, model=model_perb)
                        _, loss_list = loss_fn()
                        total_loss += sum(loss_list).item()
                    row_loss_list.append(total_loss / len(loader))
                total_loss_list.append(row_loss_list)
            torch.save({f"loss_list_task{task}": total_loss_list}, ensure_parent_dir(loss_path))
            print(f"[Landscape] Loss surface saved to {loss_path}")

        plt.clf()
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["mathtext.fontset"] = "cm"

        X, Y = np.meshgrid(lams_alpha, lams_beta, indexing="ij")
        Z = np.array(total_loss_list)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color="red", edgecolor="none", alpha=0.5)

        ax.view_init(elev=30, azim=-130)
        ax.tick_params(axis="both", which="major", labelsize=28)
        ax.zaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.tick_params(axis="z", pad=13)

        metrics = summarize_loss_surface(Z, lams_alpha, lams_beta, eigenvalues=top_eigenvalues)
        metrics.update({"optimizer_name": optimizer_name, "task": int(task)})
        metrics_path = save_dir / f"landscape_metrics_task{task}.json"
        ensure_parent_dir(metrics_path)
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

        figure_path = save_dir / f"loss_surface_task{task}.pdf"
        figure_png_path = save_dir / f"loss_surface_task{task}.png"
        plt.tight_layout()
        ensure_parent_dir(figure_path)
        plt.savefig(figure_path)
        ensure_parent_dir(figure_png_path)
        plt.savefig(figure_png_path, dpi=300)
        plt.close(fig)
        return {
            "eigen_path": str(eigen_path),
            "loss_path": str(loss_path),
            "plot_path": str(figure_path),
            "plot_png_path": str(figure_png_path),
            "metrics_path": str(metrics_path),
        }
    finally:
        model.load_state_dict(state_backup, strict=True)
        model.train(was_training)


visualize_landscape = visualize_loss_landscape
