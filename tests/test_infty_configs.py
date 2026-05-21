from pathlib import Path

import main as pilot_main


ROOT = Path(__file__).resolve().parents[1]
WORKDIR_CONFIG_ROOT = ROOT / "workdirs" / "infty_configs"


def test_workdir_infty_configs_are_complete():
    expected_files = {
        "flat_landscape/c_flat.yaml",
        "flat_landscape/c_flat_plus.yaml",
        "flat_landscape/gam.yaml",
        "flat_landscape/gsam.yaml",
        "flat_landscape/looksam.yaml",
        "flat_landscape/sam.yaml",
        "gradient_bans/zeroflow.yaml",
        "gradient_conflicts/cagrad.yaml",
        "gradient_conflicts/gradvac.yaml",
        "gradient_conflicts/ogd.yaml",
        "gradient_conflicts/pcgrad.yaml",
        "gradient_conflicts/unigrad_fs.yaml",
    }

    for rel_path in expected_files:
        assert (WORKDIR_CONFIG_ROOT / rel_path).is_file(), rel_path


def test_optimizer_config_selection_supports_workdir_layout():
    all_optimizers = (
        pilot_main.GEOMETRY_RESHAPING_OPTIMIZERS
        | pilot_main.GRADIENT_FILTERING_OPTIMIZERS
        | pilot_main.ZEROTH_ORDER_UPDATE_OPTIMIZERS
    )

    for optimizer_name in all_optimizers:
        config_path = pilot_main._select_optimizer_config(optimizer_name, WORKDIR_CONFIG_ROOT)
        assert config_path.is_file(), optimizer_name
