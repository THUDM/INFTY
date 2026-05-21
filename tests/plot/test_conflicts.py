from pathlib import Path

import pytest


def test_visualize_conflicts_accepts_similarity_values(tmp_path):
    pytest.importorskip("matplotlib")
    from infty.plot import visualize_conflicts

    result = visualize_conflicts(
        task=1,
        output_dir=tmp_path,
        source_name="conflict_source",
        sim_values=[0.1, -0.2, 0.3],
    )
    assert Path(result["sim_path"]).exists()
    assert Path(result["plot_path"]).exists()


def test_visualize_conflicts_accepts_conflict_records(tmp_path):
    pytest.importorskip("matplotlib")
    from infty.plot import visualize_conflicts

    records = [
        {
            "task": 2,
            "iter": 0,
            "block": 0,
            "cos_before": -0.3,
            "cos_after": 0.2,
            "delta_cos": 0.5,
            "norm_old": 1.0,
            "norm_new": 2.0,
            "norm_ratio_old_new": 0.5,
            "threshold": 0.1,
            "conflict": True,
            "below_threshold": True,
            "gain_old": 0.8,
            "gain_new": 0.6,
            "min_gain": 0.6,
            "w1": 0.2,
            "w2": 0.3,
        }
    ]

    result = visualize_conflicts(
        task=2,
        output_dir=tmp_path,
        source_name="conflict_source",
        conflict_records=records,
    )
    assert Path(result["sim_path"]).exists()
    assert Path(result["plot_path"]).exists()
