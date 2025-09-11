import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import numpy as np


def train(args):
    seeds = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])
    for seed in seeds:
        args["seed"] = seed
        args["device"] = device
        _train(args)

def _train(args):
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    logs_dir = f"logs/{args['model_name']}-{args['backbone_type']}-{args['dataset']}-{init_cls}-{args['increment']}"
    os.makedirs(logs_dir, exist_ok=True)
    logfilename = f"{logs_dir}/{args['inftyopt']}"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
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
    model = factory.get_model(args["model_name"], args)
    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []
    for task in range(data_manager.nb_tasks):
        logging.info(f"All params: {count_parameters(model._network)}")
        logging.info(f"Trainable params: {count_parameters(model._network, True)}")
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()
        update_matrix_and_curve(cnn_accy, nme_accy, cnn_matrix, nme_matrix, cnn_curve, nme_curve)
    if args.get('print_forget', True):
        print_forgetting(cnn_matrix, nme_matrix, task)

def update_matrix_and_curve(cnn_accy, nme_accy, cnn_matrix, nme_matrix, cnn_curve, nme_curve):
    logging.info(f"CNN: {cnn_accy['grouped']}")
    cnn_keys = [k for k in cnn_accy["grouped"] if '-' in k]
    cnn_matrix.append([cnn_accy["grouped"][k] for k in cnn_keys])
    cnn_curve["top1"].append(cnn_accy["top1"])
    cnn_curve["top5"].append(cnn_accy["top5"])
    logging.info(f"CNN top1 curve: {cnn_curve['top1']}")
    logging.info(f"CNN top5 curve: {cnn_curve['top5']}")
    print('Average Accuracy (CNN):', sum(cnn_curve["top1"]) / len(cnn_curve["top1"]))
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
        print('Accuracy Matrix (CNN):')
        print(np_acctable)
        logging.info(f'Forgetting (CNN): {forgetting}')
    # if nme_matrix:
    #     np_acctable = np.zeros([task + 1, task + 1])
    #     for idx, line in enumerate(nme_matrix):
    #         np_acctable[idx, :len(line)] = np.array(line)
    #     np_acctable = np_acctable.T
    #     forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
    #     print('Accuracy Matrix (NME):')
    #     print(np_acctable)
    #     logging.info(f'Forgetting (NME): {forgetting}')

def set_device(args):
    gpus = [torch.device('cpu') if d == -1 else torch.device(f'cuda:{d}') for d in args["device"]]
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