import json
from pathlib import Path

import trainer as pilot_trainer


def test_conflicts_export_uses_saved_conflict_stats(tmp_path):
    args = {
        "model_name": "memo",
        "inftyopt": "pcgrad",
        "seed": 1993,
        "alpha_aux": 1.0,
        "conflict_stats_dir": str(tmp_path / "conflict_stats"),
        "plot_dir": str(tmp_path / "plots"),
    }

    stats_path = pilot_trainer._conflict_stats_file_path(args, task_id=2)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(
            {
                "records": [
                    {"task": 2, "epoch": 1, "cosine_mean": -0.2},
                    {"task": 2, "epoch": 2, "cosine_mean": 0.1},
                    {"task": 2, "epoch": 3, "cosine_mean": 0.4},
                ]
            }
        ),
        encoding="utf-8",
    )

    export_record = pilot_trainer._run_conflicts_export(None, args, task_id=2)

    assert Path(export_record["stats_path"]).is_file()
    assert Path(export_record["sim_path"]).is_file()
    assert Path(export_record["plot_path"]).is_file()
    assert export_record["num_records"] == 3
