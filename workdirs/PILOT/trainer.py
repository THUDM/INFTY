import copy
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from infty import plot as infty_plot

from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters


SCRIPT_DIR = Path(__file__).resolve().parent
WORKDIRS_ROOT = SCRIPT_DIR.parent
REPO_ROOT = WORKDIRS_ROOT.parent


def _resolve_path(path_value, base_dir):
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _prepare_runtime_dirs(args):
    workdir = _resolve_path(args.get("workdir", str(WORKDIRS_ROOT / "results")), SCRIPT_DIR)
    args["workdir"] = str(workdir)

    primary_defaults = {
        "ckp_dir": workdir / "checkpoints",
        "log_dir": workdir / "logs",
        "output_dir": workdir / "outputs",
        "plot_dir": workdir / "plots",
    }
    for key, default_path in primary_defaults.items():
        raw_value = args.get(key)
        path = _resolve_path(raw_value, workdir) if raw_value is not None else default_path
        args[key] = str(path)

    output_dir = Path(args["output_dir"])
    derived_defaults = {
        "metrics_json_dir": output_dir / "metrics_json",
        "conflict_stats_dir": output_dir / "conflict_stats",
        "sharpness_json_dir": output_dir / "sharpness_json",
    }
    for key, default_path in derived_defaults.items():
        raw_value = args.get(key)
        path = _resolve_path(raw_value, output_dir) if raw_value is not None else default_path
        args[key] = str(path)

    return args


def train(args):
    args = _prepare_runtime_dirs(copy.deepcopy(args))
    seeds = copy.deepcopy(args["seed"])
    if not isinstance(seeds, (list, tuple)):
        seeds = [seeds]
    device = copy.deepcopy(args["device"])
    for seed in seeds:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _parse_name_list(raw_value):
    values = []
    for part in str(raw_value or "").split(","):
        token = part.strip().lower()
        if token and token != "none":
            values.append(token)
    return values


def _parse_task_selection(raw_value, last_task_index):
    raw = str(raw_value).strip().lower()
    if not raw or raw == "last":
        return {int(last_task_index)}
    task_ids = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        task_ids.add(int(part))
    return task_ids


def _make_plot_loader(train_loader, plot_batch_size, plot_max_batches):
    batch_size = max(1, int(plot_batch_size))
    max_batches = max(1, int(plot_max_batches))
    loader_batch_size = getattr(train_loader, "batch_size", None) or batch_size
    batch_size = min(loader_batch_size, batch_size)
    max_samples = batch_size * max_batches
    dataset = train_loader.dataset

    subset = torch.utils.data.Subset(dataset, range(min(len(dataset), max_samples)))
    plot_loader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=getattr(train_loader, "collate_fn", None),
    )
    return plot_loader


def _prepare_ease_targets(learner, targets):
    aux_targets = targets.clone()
    return torch.where(
        aux_targets - learner._known_classes >= 0,
        aux_targets - learner._known_classes,
        -1,
    )


def _make_diagnostic_loss_factory(learner):
    if not hasattr(learner, "create_loss_fn"):
        raise AttributeError(f"{learner.__class__.__name__} does not define create_loss_fn().")

    def create_loss_fn(inputs, targets, model=None):
        if learner.args.get("model_name") == "ease":
            diag_targets = _prepare_ease_targets(learner, targets)
            return learner.create_loss_fn(inputs, diag_targets, model=model)
        return learner.create_loss_fn(inputs, targets, model=model)

    return create_loss_fn


def _conflict_weight_metadata(args):
    if "alpha_aux" in args:
        return "alpha_aux", float(args["alpha_aux"])
    if "alpha_kd" in args:
        return "alpha_kd", float(args["alpha_kd"])
    return "alpha", 1.0


def _conflict_file_tag(args):
    weight_name, weight_value = _conflict_weight_metadata(args)
    safe_name = weight_name.replace("_", "")
    safe_value = str(weight_value).replace("-", "m").replace(".", "p")
    return f"{safe_name}{safe_value}"


def _conflict_stats_file_path(args, task_id):
    output_dir = Path(args["conflict_stats_dir"]).expanduser().resolve()
    seed = args["seed"][0] if isinstance(args["seed"], (list, tuple)) else args["seed"]
    filename = (
        f"{args['model_name']}_{args['inftyopt']}_seed{seed}_"
        f"{_conflict_file_tag(args)}_task{int(task_id)}.json"
    )
    return output_dir / filename


def _load_conflict_epoch_records(args, task_id):
    stats_path = _conflict_stats_file_path(args, task_id)
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing conflict statistics file: {stats_path}")
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    return payload.get("records", []), stats_path


def _run_landscape_export(learner, args, task_id):
    plot_loader = _make_plot_loader(
        learner.train_loader,
        plot_batch_size=args.get("plot_batch_size", 128),
        plot_max_batches=args.get("plot_max_batches", 9999),
    )
    plot_root = Path(args["plot_dir"]).expanduser().resolve()
    output_dir = plot_root / "diagnostics" / "landscape" / args["model_name"]
    logging.info("[Landscape] Using probe loader with %s samples.", len(plot_loader.dataset))
    return infty_plot.visualize_loss_landscape(
        optimizer=None,
        model=learner._network,
        create_loss_fn=_make_diagnostic_loss_factory(learner),
        loader=plot_loader,
        task=task_id,
        device=learner._device,
        limit=args.get("landscape_limit", 0.3),
        samples=args.get("landscape_samples", 11),
        eigen_max_iter=args.get("landscape_eigen_max_iter", 50),
        eigen_tol=args.get("landscape_eigen_tol", 1e-3),
        output_dir=output_dir,
        source_name=args["inftyopt"],
    )


def _run_esd_export(learner, args, task_id):
    plot_loader = _make_plot_loader(
        learner.train_loader,
        plot_batch_size=args.get("plot_batch_size", 128),
        plot_max_batches=args.get("plot_max_batches", 9999),
    )
    plot_root = Path(args["plot_dir"]).expanduser().resolve()
    output_dir = plot_root / "diagnostics" / "esd" / args["model_name"]
    logging.info("[ESD] Using probe loader with %s samples.", len(plot_loader.dataset))
    return infty_plot.visualize_esd(
        optimizer=None,
        model=learner._network,
        create_loss_fn=_make_diagnostic_loss_factory(learner),
        loader=plot_loader,
        task=task_id,
        device=learner._device,
        output_dir=output_dir,
        source_name=args["inftyopt"],
        trace_max_iter=args.get("trace_max_iter", 100),
        trace_tol=args.get("trace_tol", 1e-3),
        density_iter=args.get("density_iter", 100),
        density_runs=args.get("density_runs", 1),
    )


def _run_conflicts_export(learner, args, task_id):
    del learner
    records, stats_path = _load_conflict_epoch_records(args, task_id)
    sim_values = [float(record["cosine_mean"]) for record in records if "cosine_mean" in record]
    plot_root = Path(args["plot_dir"]).expanduser().resolve()
    output_dir = plot_root / "diagnostics" / "conflicts" / args["model_name"]
    export_record = infty_plot.visualize_conflicts(
        task=task_id,
        output_dir=output_dir,
        source_name=args["inftyopt"],
        sim_values=sim_values,
    )
    export_record["stats_path"] = str(stats_path)
    export_record["num_records"] = int(len(records))
    return export_record


def _save_diagnostics_summary(args, diagnostics_payload):
    workdir = Path(args["workdir"]).expanduser().resolve()
    summary_path = workdir / f"diagnostics_summary_seed{args['seed']}.json"
    summary_path.write_text(json.dumps(diagnostics_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("[Diagnostics] Saved summary to %s", summary_path)


def _train(args):
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    log_root = Path(args.get("log_dir", str(Path(args["workdir"]) / "logs"))).expanduser().resolve()
    logs_dir = log_root / (
        f"{args['model_name']}-{args['backbone_type']}-{args['dataset']}-"
        f"{init_cls}-{args['increment']}-{args['seed']}"
    )
    logs_dir.mkdir(parents=True, exist_ok=True)
    logfilename = logs_dir / f"{args['inftyopt']}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=str(logfilename)),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    set_random(args["seed"])
    set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"], args["shuffle"], args["seed"],
        args["init_cls"], args["increment"], args,
    )
    args["nb_classes"] = data_manager.nb_classes
    args["nb_tasks"] = data_manager.nb_tasks
    learner = factory.get_model(args["model_name"], args)
    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []
    enabled_postplots = set(_parse_name_list(args.get("postplot", "")))
    unknown_postplots = enabled_postplots - {"landscape", "esd", "conflicts"}
    if unknown_postplots:
        raise ValueError(f"Unsupported postplot diagnostics: {sorted(unknown_postplots)}")
    if "conflicts" in enabled_postplots:
        args["save_conflict_stats"] = True
        args["conflict_probe_batches"] = max(1, int(args.get("conflict_probe_batches", 5)))
    diagnostics_payload = {
        "model_name": args["model_name"],
        "backbone_type": args["backbone_type"],
        "inftyopt": args["inftyopt"],
        "seed": int(args["seed"]),
        "postplot": sorted(enabled_postplots),
        "landscape_records": [],
        "esd_records": [],
        "conflicts_records": [],
    }

    tasks_to_run = data_manager.nb_tasks
    max_tasks = int(args.get("max_tasks", 0) or 0)
    if max_tasks > 0:
        tasks_to_run = min(tasks_to_run, max_tasks)
    last_task_index = tasks_to_run - 1

    landscape_tasks = _parse_task_selection(args.get("landscape_tasks", "last"), last_task_index)
    esd_tasks = _parse_task_selection(args.get("esd_tasks", "last"), last_task_index)
    conflicts_tasks = _parse_task_selection(args.get("conflicts_tasks", "last"), last_task_index)

    for task in range(tasks_to_run):
        logging.info(f"All params: {count_parameters(learner._network)}")
        logging.info(f"Trainable params: {count_parameters(learner._network, True)}")
        learner.incremental_train(data_manager)

        if "landscape" in enabled_postplots and task in landscape_tasks:
            try:
                export_record = _run_landscape_export(learner, args, task)
                diagnostics_payload["landscape_records"].append({"task": int(task), **export_record})
                logging.info("[Landscape] Exported task %s to %s", task, export_record["plot_path"])
            except Exception as exc:
                logging.exception("[Landscape] Failed on task %s: %s", task, exc)
                raise

        if "esd" in enabled_postplots and task in esd_tasks:
            try:
                export_record = _run_esd_export(learner, args, task)
                diagnostics_payload["esd_records"].append({"task": int(task), **export_record})
                logging.info("[ESD] Exported task %s to %s", task, export_record["plot_path"])
            except Exception as exc:
                logging.exception("[ESD] Failed on task %s: %s", task, exc)
                raise

        if "conflicts" in enabled_postplots and task in conflicts_tasks:
            try:
                export_record = _run_conflicts_export(learner, args, task)
                diagnostics_payload["conflicts_records"].append({"task": int(task), **export_record})
                if "plot_path" in export_record:
                    logging.info("[Conflicts] Exported task %s to %s", task, export_record["plot_path"])
                else:
                    logging.info("[Conflicts] Exported task %s similarity data to %s", task, export_record["sim_path"])
            except Exception as exc:
                logging.exception("[Conflicts] Failed on task %s: %s", task, exc)
                raise

        cnn_accy, nme_accy = learner.eval_task()
        learner.after_task()
        update_matrix_and_curve(cnn_accy, nme_accy, cnn_matrix, nme_matrix, cnn_curve, nme_curve)
    cnn_forgetting = compute_forgetting(cnn_matrix, last_task_index)
    if args.get("print_forget", True):
        print_forgetting(cnn_matrix, nme_matrix, last_task_index)
    if args.get("save_metrics_json", False):
        save_metrics_json(args, cnn_curve, cnn_matrix, cnn_forgetting)
    if enabled_postplots:
        diagnostics_payload["cnn_curve_top1"] = [float(x) for x in cnn_curve["top1"]]
        diagnostics_payload["cnn_curve_top5"] = [float(x) for x in cnn_curve["top5"]]
        diagnostics_payload["cnn_matrix"] = [[float(v) for v in row] for row in cnn_matrix]
        diagnostics_payload["forgetting"] = float(cnn_forgetting)
        diagnostics_payload["tasks_trained"] = int(tasks_to_run)
        _save_diagnostics_summary(args, diagnostics_payload)


def update_matrix_and_curve(cnn_accy, nme_accy, cnn_matrix, nme_matrix, cnn_curve, nme_curve):
    logging.info(f"CNN: {cnn_accy['grouped']}")
    cnn_keys = [k for k in cnn_accy["grouped"] if "-" in k]
    cnn_matrix.append([cnn_accy["grouped"][k] for k in cnn_keys])
    cnn_curve["top1"].append(cnn_accy["top1"])
    cnn_curve["top5"].append(cnn_accy["top5"])
    logging.info(f"CNN top1 curve: {cnn_curve['top1']}")
    logging.info(f"CNN top5 curve: {cnn_curve['top5']}")
    print("Average Accuracy (CNN):", sum(cnn_curve["top1"]) / len(cnn_curve["top1"]))
    logging.info(f"Average Accuracy (CNN): {sum(cnn_curve['top1']) / len(cnn_curve['top1'])}")
    # if nme_accy is not None:
    #     logging.info(f"NME: {nme_accy['grouped']}")
    #     nme_keys = [k for k in nme_accy["grouped"] if '-' in k]
    #     nme_matrix.append([nme_accy["grouped"][k] for k in nme_keys])
    #     nme_curve["top1"].append(nme_accy["top1"])
    #     nme_curve["top5"].append(nme_accy["top5"])
    #     logging.info(f"NME top1 curve: {nme_curve['top1']}")
    #     logging.info(f"NME top5 curve: {nme_curve['top5']}")
    #     print('Average Accuracy (NME):', sum(nme_curve["top1"]) / len(nme_curve["top1"]))
    #     logging.info(f"Average Accuracy (NME): {sum(nme_curve['top1']) / len(nme_curve['top1'])}")


def print_forgetting(cnn_matrix, nme_matrix, task):
    if cnn_matrix:
        np_acctable = np.zeros([task + 1, task + 1])
        for idx, line in enumerate(cnn_matrix):
            np_acctable[idx, :len(line)] = np.array(line)
        np_acctable = np_acctable.T
        forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
        print("Accuracy Matrix (CNN):")
        print(np_acctable)
        logging.info(f"Forgetting (CNN): {forgetting}")
    # if nme_matrix:
    #     np_acctable = np.zeros([task + 1, task + 1])
    #     for idx, line in enumerate(nme_matrix):
    #         np_acctable[idx, :len(line)] = np.array(line)
    #     np_acctable = np_acctable.T
    #     forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
    #     print('Accuracy Matrix (NME):')
    #     print(np_acctable)
    #     logging.info(f'Forgetting (NME): {forgetting}')


def compute_forgetting(cnn_matrix, task):
    if not cnn_matrix or task <= 0:
        return 0.0
    np_acctable = np.zeros([task + 1, task + 1])
    for idx, line in enumerate(cnn_matrix):
        np_acctable[idx, :len(line)] = np.array(line)
    np_acctable = np_acctable.T
    return float(np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task]))


def save_metrics_json(args, cnn_curve, cnn_matrix, cnn_forgetting):
    output_dir = Path(args.get("metrics_json_dir", "./metrics_json")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    weight_name = "alpha_aux" if "alpha_aux" in args else "alpha_kd" if "alpha_kd" in args else "alpha"
    weight_value = float(args.get(weight_name, 1.0))
    weight_tag = f"{weight_name.replace('_', '')}{str(weight_value).replace('-', 'm').replace('.', 'p')}"

    payload = {
        "model_name": args["model_name"],
        "method": args["inftyopt"],
        "seed": args["seed"],
        "weight_name": weight_name,
        "weight_value": weight_value,
        "cnn_curve_top1": [float(x) for x in cnn_curve["top1"]],
        "cnn_curve_top5": [float(x) for x in cnn_curve["top5"]],
        "cnn_matrix": [[float(v) for v in row] for row in cnn_matrix],
        "last_accuracy": float(cnn_curve["top1"][-1]) if cnn_curve["top1"] else 0.0,
        "mean_accuracy": float(sum(cnn_curve["top1"]) / len(cnn_curve["top1"])) if cnn_curve["top1"] else 0.0,
        "forgetting": float(cnn_forgetting),
    }

    output_path = output_dir / build_metrics_filename(args, weight_tag)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    logging.info(f"[Metrics] Saved summary to {output_path}")


def build_metrics_filename(args, weight_tag):
    return f"{args['model_name']}_{args['inftyopt']}_seed{args['seed']}_{weight_tag}.json"


def set_device(args):
    gpus = [torch.device("cpu") if d == -1 else torch.device(f"cuda:{d}") for d in args["device"]]
    args["device"] = gpus


def set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
