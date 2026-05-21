#!/usr/bin/env python
"""Render trajectory plots for the methods supported by infty.plot.

Run from the repository root or any subdirectory:
    python workdirs/scripts/plot_zo_trajectories.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKDIRS_ROOT = REPO_ROOT / "workdirs"
RESULTS_ROOT = WORKDIRS_ROOT / "results"
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from infty.plot import visualize_trajectory
from infty.plot.trajectory.solvers import SOLVER_MAP


ALIASES = {
    "zo_adam_cons": "zo_adam_conserve",
}
CANONICAL_TRAJECTORY_METHODS = [
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
]
METHOD_LR_OVERRIDES = {
    "sgd": 0.01,
    "zo_sgd": 0.01,
    "zo_sgd_q4": 0.01,
    "zo_sgd_sign": 0.01,
    "zo_sgd_conserve": 0.01,
    "forward_grad": 0.01,
}


def get_supported_trajectory_methods():
    supported = [name for name in CANONICAL_TRAJECTORY_METHODS if name in SOLVER_MAP]
    supported_set = set(supported)
    alias_names = set(ALIASES)
    extras = sorted(
        name
        for name in SOLVER_MAP
        if name not in supported_set and name not in alias_names
    )
    return supported + extras


def normalize_methods(methods):
    normalized = []
    supported = set(get_supported_trajectory_methods())
    for method in methods:
        canonical = ALIASES.get(method, method)
        if canonical not in supported:
            supported_text = ", ".join(sorted(supported | set(ALIASES)))
            raise ValueError(f"Unsupported trajectory method: {method}. Supported values: {supported_text}")
        normalized.append((method, canonical))
    return normalized


def resolve_lr(method_name, base_lr):
    return METHOD_LR_OVERRIDES.get(method_name, base_lr)


def format_method_lr_pairs(method_pairs, base_lr):
    rendered = []
    for requested_name, canonical_name in method_pairs:
        effective_lr = resolve_lr(canonical_name, base_lr)
        rendered.append(f"{requested_name}:{effective_lr}")
    return ", ".join(rendered)


def parse_args():
    default_run_name = f"trajectory_methods_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    parser = argparse.ArgumentParser(description="Render trajectory plots for supported infty methods.")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Methods to render. Defaults to every canonical trajectory method supported by infty.plot.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=default_run_name,
        help="Result-directory name created under workdirs/results when --output-root is not set.",
    )
    parser.add_argument("--n-iter", type=int, default=2000, help="Number of optimizer steps per method.")
    parser.add_argument("--lr", type=float, default=0.1, help="Optimizer learning rate.")
    parser.add_argument("--grid-size", type=int, default=120, help="Contour grid size for the plot background.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory that will contain one subdirectory per method. Defaults to workdirs/results/<run-name>.",
    )
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="Print the supported trajectory method names and exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    supported = get_supported_trajectory_methods()

    if args.list_methods:
        for method in supported:
            print(method)
        if ALIASES:
            for alias, canonical in sorted(ALIASES.items()):
                print(f"{alias} -> {canonical}")
        return

    requested_methods = args.methods or supported
    method_pairs = normalize_methods(requested_methods)

    output_root = Path(args.output_root) if args.output_root is not None else RESULTS_ROOT / args.run_name
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[trajectory] output_root={output_root}")
    print(f"[trajectory] methods={', '.join(requested_methods)}")
    print(f"[trajectory] n_iter={args.n_iter} grid_size={args.grid_size}")
    print(f"[trajectory] effective_lrs={format_method_lr_pairs(method_pairs, args.lr)}")

    for requested_name, canonical_name in method_pairs:
        method_output_dir = output_root / requested_name
        method_output_dir.mkdir(parents=True, exist_ok=True)
        effective_lr = resolve_lr(canonical_name, args.lr)
        trajectory = visualize_trajectory(
            optimizer_name=canonical_name,
            n_iter=args.n_iter,
            lr=effective_lr,
            output_dir=method_output_dir,
            grid_size=args.grid_size,
        )
        print(
            f"[trajectory] saved method={requested_name} canonical={canonical_name} lr={effective_lr} "
            f"steps={trajectory.shape[0]} dir={method_output_dir}"
        )


if __name__ == "__main__":
    main()
