from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .._common.paths import DEFAULT_CONFLICTS_DIR, ensure_parent_dir
from .._common.source import resolve_similarity_values, resolve_source_name


DEFAULT_OUTPUT_DIR = DEFAULT_CONFLICTS_DIR


def _get_similarity_values(source=None, sim_values=None):
    sim_values = resolve_similarity_values(source=source, sim_values=sim_values)
    if sim_values is None:
        raise AttributeError("source must expose sim_list/sim_arr, or sim_values must be provided")
    return np.asarray([float(value) for value in sim_values], dtype=np.float32)


def _derive_similarity_values_from_conflict_records(conflict_records, task):
    if not conflict_records:
        return np.asarray([], dtype=np.float32)

    target_task = int(task)
    values = []
    for record in conflict_records:
        record_task = int(record.get("task", target_task))
        if record_task != target_task:
            continue
        values.append(float(record.get("cos_before", record.get("similarity", 0.0))))
    return np.asarray(values, dtype=np.float32)


def visualize_conflicts(
    optimizer=None,
    task=None,
    output_dir=None,
    task_id=None,
    dir_path=None,
    *,
    source_name=None,
    sim_values=None,
    conflict_records=None,
):
    if task is None:
        task = task_id if task_id is not None else 0
    if output_dir is None:
        output_dir = dir_path

    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    output_dir = output_dir.expanduser().resolve()

    optimizer_name = resolve_source_name(source=optimizer, source_name=source_name, default="custom")
    save_dir = output_dir / optimizer_name

    if sim_values is None:
        try:
            sim_values = _get_similarity_values(source=optimizer, sim_values=sim_values)
        except AttributeError:
            if conflict_records is not None:
                sim_values = _derive_similarity_values_from_conflict_records(conflict_records, task)
            elif optimizer is not None and getattr(optimizer, "conflict_records", None):
                sim_values = _derive_similarity_values_from_conflict_records(optimizer.conflict_records, task)
            else:
                raise
    else:
        sim_values = _get_similarity_values(source=optimizer, sim_values=sim_values)
    sim_path = save_dir / f"sim_list_task{task}.pt"
    ensure_parent_dir(sim_path)
    torch.save(sim_values.tolist(), sim_path)

    if task <= 0:
        return {"optimizer_name": optimizer_name, "task": int(task), "sim_path": str(sim_path)}

    if sim_values.size == 0:
        warning_path = save_dir / f"warning_task{task}.txt"
        ensure_parent_dir(warning_path)
        warning_path.write_text("No similarity values were recorded for conflict visualization.\n", encoding="utf-8")
        return {
            "optimizer_name": optimizer_name,
            "task": int(task),
            "sim_path": str(sim_path),
            "warning_path": str(warning_path),
        }

    figure_path = save_dir / f"similarity_{task}.pdf"
    iterations = np.arange(1, sim_values.size + 1)

    plt.figure(figsize=(12, 8))
    plt.plot(iterations, sim_values, "b-", linewidth=2, alpha=0.8, label="Cosine Similarity")
    plt.grid(True, alpha=0.3)

    fontsize = 24
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel("Iterations", fontsize=fontsize)
    plt.ylabel("Cosine Similarity", fontsize=fontsize)

    if sim_values.size > 1:
        plt.xlim(1, sim_values.size)
    y_min = float(sim_values.min())
    y_max = float(sim_values.max())
    y_pad = max(0.05, abs(y_max - y_min) * 0.05)
    plt.ylim(y_min - y_pad, y_max + y_pad)

    mean_sim = float(np.mean(sim_values))
    std_sim = float(np.std(sim_values))
    plt.text(
        0.02,
        0.98,
        f"Mean: {mean_sim:.4f}\nStd: {std_sim:.4f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=fontsize,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    ensure_parent_dir(figure_path)
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "optimizer_name": optimizer_name,
        "task": int(task),
        "sim_path": str(sim_path),
        "plot_path": str(figure_path),
    }
