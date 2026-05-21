import argparse
from pathlib import Path

from trainer import train
from utils.toolkit import load_json, load_yaml


SCRIPT_DIR = Path(__file__).resolve().parent
WORKDIRS_ROOT = SCRIPT_DIR.parent
REPO_ROOT = WORKDIRS_ROOT.parent

GEOMETRY_RESHAPING_OPTIMIZERS = {
    "sam",
    "gsam",
    "looksam",
    "gam",
    "c_flat",
    "c_flat_plus",
}
ZEROTH_ORDER_UPDATE_OPTIMIZERS = {
    "zo_sgd",
    "zo_sgd_sign",
    "zo_sgd_conserve",
    "zo_adam",
    "zo_adam_sign",
    "zo_adam_conserve",
    "forward_grad",
}
GRADIENT_FILTERING_OPTIMIZERS = {
    "pcgrad",
    "gradvac",
    "cagrad",
    "unigrad_fs",
    "ogd",
}
OFFICIAL_CONFIG_SUBDIRS = {
    "geometry": "flat_landscape",
    "gradient": "gradient_conflicts",
    "zeroth_order": "gradient_bans",
}
LEGACY_CONFIG_SUBDIRS = {
    "geometry": "geometry_reshaping",
    "gradient": "gradient_filtering",
    "zeroth_order": "zeroth_order_updates",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inftyopt", type=str, default="ogd", help="select the optimizer")
    parser.add_argument("--config", type=str, default="./exps/ogd.json", help="experiment config file path")
    parser.add_argument(
        "--workdir",
        type=str,
        default=str(WORKDIRS_ROOT / "results"),
        help="root directory for runtime artifacts",
    )
    parser.add_argument("--infty_config_dir", type=str, default=None, help="directory containing INFTY yaml configs")
    parser.add_argument("--ckp_dir", type=str, default=None, help="checkpoint directory")
    parser.add_argument("--log_dir", type=str, default=None, help="log directory")
    parser.add_argument("--output_dir", type=str, default=None, help="output directory")
    parser.add_argument("--plot_dir", type=str, default=None, help="plot directory")
    parser.add_argument(
        "--postplot",
        type=str,
        default="",
        help="Comma-separated training-time diagnostics to export after selected tasks, e.g. 'landscape,esd'.",
    )
    parser.add_argument("--max_tasks", type=int, default=0, help="Maximum number of tasks to train. Use 0 for all tasks.")
    parser.add_argument(
        "--landscape_tasks",
        type=str,
        default="last",
        help="Comma-separated task ids for landscape export, or 'last'. Task ids are zero-based.",
    )
    parser.add_argument(
        "--esd_tasks",
        type=str,
        default="last",
        help="Comma-separated task ids for ESD export, or 'last'. Task ids are zero-based.",
    )
    parser.add_argument(
        "--conflicts_tasks",
        type=str,
        default="last",
        help="Comma-separated task ids for conflict export, or 'last'. Task ids are zero-based.",
    )
    parser.add_argument("--plot_batch_size", type=int, default=128, help="Probe-loader batch size used by plot diagnostics.")
    parser.add_argument(
        "--plot_max_batches",
        type=int,
        default=9999,
        help="Maximum number of train-loader batches copied for plot diagnostics.",
    )
    parser.add_argument("--landscape_samples", type=int, default=11, help="Loss-surface grid size.")
    parser.add_argument("--landscape_limit", type=float, default=0.3, help="Loss-surface axis limit.")
    parser.add_argument(
        "--landscape_eigen_max_iter",
        type=int,
        default=50,
        help="Power-iteration steps used for the Hessian eigen solve.",
    )
    parser.add_argument(
        "--landscape_eigen_tol",
        type=float,
        default=1e-3,
        help="Tolerance used for the Hessian eigen solve.",
    )
    parser.add_argument(
        "--trace_max_iter",
        type=int,
        default=100,
        help="Trace-estimation iterations used for Hessian trace computation.",
    )
    parser.add_argument(
        "--trace_tol",
        type=float,
        default=1e-3,
        help="Tolerance used for Hessian trace computation.",
    )
    parser.add_argument(
        "--density_iter",
        type=int,
        default=100,
        help="Lanczos iterations used for empirical spectral density estimation.",
    )
    parser.add_argument(
        "--density_runs",
        type=int,
        default=1,
        help="Number of random probe runs used for empirical spectral density estimation.",
    )
    parser.add_argument(
        "--conflict_probe_batches",
        type=int,
        default=5,
        help="Number of train batches probed per epoch when collecting conflict statistics.",
    )
    return parser.parse_args()


def _resolve_user_path(path_value, base_dir):
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _resolve_runtime_dir(path_value, default_path, base_dir):
    if path_value is None:
        path = default_path
    else:
        path = _resolve_user_path(path_value, base_dir)
    return path.resolve()


def _resolve_output_subdir(path_value, default_path, relative_root):
    if path_value is None:
        path = default_path
    else:
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = relative_root / path
    return path.resolve()


def _default_infty_config_dir():
    return WORKDIRS_ROOT / "infty_configs"


def _iter_optimizer_config_candidates(optimizer_name, infty_config_dir):
    if optimizer_name in GEOMETRY_RESHAPING_OPTIMIZERS:
        category = "geometry"
        filename = f"{optimizer_name}.yaml"
    elif optimizer_name in GRADIENT_FILTERING_OPTIMIZERS:
        category = "gradient"
        filename = f"{optimizer_name}.yaml"
    elif optimizer_name in ZEROTH_ORDER_UPDATE_OPTIMIZERS:
        category = "zeroth_order"
        filename = "zeroflow.yaml"
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}")

    for subdir in (OFFICIAL_CONFIG_SUBDIRS[category], LEGACY_CONFIG_SUBDIRS[category]):
        yield infty_config_dir / subdir / filename


def _select_optimizer_config(optimizer_name, infty_config_dir):
    if optimizer_name == "base":
        return None
    attempted_paths = list(_iter_optimizer_config_candidates(optimizer_name, infty_config_dir))
    for config_path in attempted_paths:
        if config_path.is_file():
            return config_path

    attempted_msg = ", ".join(str(path) for path in attempted_paths)
    raise FileNotFoundError(f"Missing optimizer config for '{optimizer_name}'. Tried: {attempted_msg}")


def main():
    cli_args = parse_args()
    optimizer_name = cli_args.inftyopt.lower()

    config_json_path = _resolve_user_path(cli_args.config, SCRIPT_DIR)
    # Anchor runtime artifacts at the repository root so relative workdirs do
    # not end up nested under PILOT/.
    workdir = _resolve_user_path(cli_args.workdir, SCRIPT_DIR)
    infty_config_dir = _resolve_runtime_dir(cli_args.infty_config_dir, _default_infty_config_dir(), SCRIPT_DIR)
    ckp_dir = _resolve_runtime_dir(cli_args.ckp_dir, workdir / "checkpoints", SCRIPT_DIR)
    log_dir = _resolve_runtime_dir(cli_args.log_dir, workdir / "logs", SCRIPT_DIR)
    output_dir = _resolve_runtime_dir(cli_args.output_dir, workdir / "outputs", SCRIPT_DIR)
    plot_dir = _resolve_runtime_dir(cli_args.plot_dir, workdir / "plots", SCRIPT_DIR)

    optimizer_config_path = _select_optimizer_config(optimizer_name, infty_config_dir)
    if optimizer_config_path is not None:
        print(f"Loading optimizer config: {optimizer_config_path}")

    args = load_json(str(config_json_path))
    if optimizer_config_path is not None:
        optimizer_args = load_yaml(str(optimizer_config_path))
        args.update(optimizer_args)

    args["inftyopt"] = optimizer_name
    args["workdir"] = str(workdir)
    args["infty_config_dir"] = str(infty_config_dir)
    args["ckp_dir"] = str(ckp_dir)
    args["log_dir"] = str(log_dir)
    args["output_dir"] = str(output_dir)
    args["plot_dir"] = str(plot_dir)
    args["postplot"] = cli_args.postplot
    args["max_tasks"] = int(cli_args.max_tasks)
    args["landscape_tasks"] = cli_args.landscape_tasks
    args["esd_tasks"] = cli_args.esd_tasks
    args["conflicts_tasks"] = cli_args.conflicts_tasks
    args["plot_batch_size"] = int(cli_args.plot_batch_size)
    args["plot_max_batches"] = int(cli_args.plot_max_batches)
    args["landscape_samples"] = int(cli_args.landscape_samples)
    args["landscape_limit"] = float(cli_args.landscape_limit)
    args["landscape_eigen_max_iter"] = int(cli_args.landscape_eigen_max_iter)
    args["landscape_eigen_tol"] = float(cli_args.landscape_eigen_tol)
    args["trace_max_iter"] = int(cli_args.trace_max_iter)
    args["trace_tol"] = float(cli_args.trace_tol)
    args["density_iter"] = int(cli_args.density_iter)
    args["density_runs"] = int(cli_args.density_runs)
    args["conflict_probe_batches"] = int(cli_args.conflict_probe_batches)
    args["metrics_json_dir"] = str(
        _resolve_output_subdir(args.get("metrics_json_dir"), output_dir / "metrics_json", output_dir)
    )
    args["conflict_stats_dir"] = str(
        _resolve_output_subdir(args.get("conflict_stats_dir"), output_dir / "conflict_stats", output_dir)
    )
    args["sharpness_json_dir"] = str(
        _resolve_output_subdir(args.get("sharpness_json_dir"), output_dir / "sharpness_json", output_dir)
    )

    train(args)


if __name__ == "__main__":
    main()
