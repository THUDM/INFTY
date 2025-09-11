import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.inc_net import CosineIncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
from utils.core import get_infty_optimizer
from infty import plot as infty_plot
import os

EPSILON = 1e-8
num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _load_checkpoint(self, train_loader, test_loader, optimizer, scheduler):
        ckp_path = os.path.join(self.args["ckp_dir"], f"{self.args['model_name']}-{self.args['backbone_type']}-init.pth")
        if os.path.exists(ckp_path):
            print(f"\033[92m[Infty Info]\033[0m Loading checkpoint from {ckp_path}")
            state_dict = torch.load(ckp_path, map_location="cpu")["model"]
            model_state = self._network.state_dict()
            new_state_dict = {}

            def strip_module_prefix(key): return key.replace("module.", "", 1)
            def add_module_prefix(key): return f"module.{key}" if not key.startswith("module.") else key

            if list(state_dict.keys())[0].startswith("module.") and not list(model_state.keys())[0].startswith("module."):
                state_dict = {strip_module_prefix(k): v for k, v in state_dict.items()}
            elif not list(state_dict.keys())[0].startswith("module.") and list(model_state.keys())[0].startswith("module."):
                state_dict = {add_module_prefix(k): v for k, v in state_dict.items()}
            self._network.load_state_dict(state_dict, strict=False)
        else:
            print(f"\033[93m[Infty Warning]\033[0m No checkpoint found at {ckp_path}, running init training.")
            self._init_train(train_loader, test_loader, optimizer, scheduler)
            torch.save({"model": self._network.state_dict()}, ckp_path)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            base_optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.args["init_lr"],
                weight_decay=self.args["init_weight_decay"],
            )
            run_args={**self.args, "task_id": self._cur_task}
            optimizer = get_infty_optimizer(params=self._network.parameters(), base_optimizer=base_optimizer, model=self._network, args=run_args)
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"]
            )
            # self._init_train(train_loader, test_loader, optimizer, scheduler)
            self._load_checkpoint(train_loader, test_loader, optimizer, scheduler)
        else:
            base_optimizer = optim.SGD(
                self._network.parameters(),
                lr=self.args["lrate"],
                momentum=0.9,
                weight_decay=self.args["weight_decay"],
            )  # 1e-5
            # TODO (step 1): wrap the base_optimizer with Infty optimizer
            run_args={**self.args, "task_id": self._cur_task}
            optimizer = get_infty_optimizer(params=self._network.parameters(), base_optimizer=base_optimizer, model=self._network, args=run_args)
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["milestones"], gamma=self.args["lrate_decay"]
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
        # for update the subspace basis
        optimizer.post_process(train_loader)

        # Optional: infty plots
        # infty_plot.visualize_loss_landscape(optimizer, self._network, self.create_loss_fn, train_loader, self._cur_task, self._device)
        # infty_plot.visualize_esd(optimizer, self._network, self.create_loss_fn, train_loader, self._cur_task, self._device)
        # infty_plot.visualize_conflicts(optimizer, self._cur_task)

    # TODO (step 2): Implement the create_loss_fn function
    def create_loss_fn(self, inputs, targets, model=None):
        """
        Create a closure to calculate the loss
        """
        if model is None:
            model = self._network
            
        def loss_fn():
            if self._cur_task == 0:
                logits = model(inputs)["logits"]
                loss_clf = F.cross_entropy(logits, targets)
                return logits, [loss_clf]
            else:
                logits = model(inputs)["logits"]
                loss_clf = F.cross_entropy(logits, targets)
                loss_kd = _KD_loss(
                    logits[:, : self._known_classes],
                    self._old_network(inputs)["logits"],
                    self.args["T"],
                )
                return logits, [loss_clf, loss_kd]
        return loss_fn
    
    # WARNING: if use forward_grad instead, 
    @staticmethod
    @torch.no_grad()
    def create_jvp_loss_fn(inputs, targets):
        """
        Create a closure to calculate the loss for jvp
        """
        from torch.func import functional_call
        batch = (inputs, targets)
        def functional_call_loss(params, names, buffers, model, batch):
            params = {k: v for k, v in zip(names, params)}
            (inputs, targets) = batch
            if self._cur_task == 0:
                logits = functional_call(model, (params, buffers), (inputs, False))["logits"]
                loss_clf = F.cross_entropy(logits, targets)
                return logits, [loss_clf]
            else:
                logits = functional_call(model, (params, buffers), (inputs, False))["logits"]
                loss_clf = F.cross_entropy(logits, targets)
                loss_kd = _KD_loss(
                    logits[:, : self._known_classes],
                    self._old_network(inputs)["logits"],
                    self.args["T"],
                )
                return logits, [loss_clf, loss_kd]
        return functional_call_loss, batch
    
    
    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                # logits = self._network(inputs)["logits"]

                # loss = F.cross_entropy(logits, targets)
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                # losses += loss.item()

                # TODO (step 3): Use the loss_fn to calculate the loss and backward
                if self.args["inftyopt"] == "forward_grad":  
                    loss_fn, batch = self.create_jvp_loss_fn(inputs, targets)
                    optimizer.set_closure(loss_fn, batch)
                else:
                    loss_fn = self.create_loss_fn(inputs, targets)
                    optimizer.set_closure(loss_fn)
                logits, loss_list = optimizer.step()
                losses += sum(loss_list)

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)
        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                # logits = self._network(inputs)["logits"]
                # loss_clf = F.cross_entropy(logits, targets)
                # loss_kd = _KD_loss(
                #     logits[:, : self._known_classes],
                #     self._old_network(inputs)["logits"],
                #     self.args["T"],
                # )

                # loss = loss_clf + loss_kd

                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                # losses += loss.item()

                # TODO (step 3): Use the loss_fn to calculate the loss and backward
                if self.args["inftyopt"] == "forward_grad":  
                    loss_fn, batch = self.create_jvp_loss_fn(inputs, targets)
                    optimizer.set_closure(loss_fn, batch)
                else:
                    loss_fn = self.create_loss_fn(inputs, targets)
                    optimizer.set_closure(loss_fn)
                logits, loss_list = optimizer.step()
                losses += sum(loss_list)

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
