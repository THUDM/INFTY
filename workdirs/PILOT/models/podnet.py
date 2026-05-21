import logging
import math

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.base import BaseLearner
from utils.core import get_infty_optimizer
from utils.inc_net import CosineIncrementalNet
from utils.toolkit import tensor2numpy


num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = CosineIncrementalNet(
            args, pretrained=False, nb_proxy=args.get("nb_proxy", 10)
        )
        self._class_means = None

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self.task_size = self._total_classes - self._known_classes
        self._network.update_fc(self._total_classes, self._cur_task)
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
            train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=num_workers,
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args["batch_size"],
            shuffle=False,
            num_workers=num_workers,
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(data_manager, self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _network_ptr(self):
        if hasattr(self._network, "module"):
            return self._network.module
        return self._network

    def _build_optimizer(self, base_optimizer, model=None):
        if model is None:
            model = self._network
        run_args = {**self.args, "task_id": self._cur_task}
        return get_infty_optimizer(
            params=model.parameters(),
            base_optimizer=base_optimizer,
            model=model,
            args=run_args,
        )

    def create_loss_fn(self, inputs, targets, model=None, old_model=None):
        if model is None:
            model = self._network
        if old_model is None:
            old_model = self._old_network

        def loss_fn():
            outputs = model(inputs)
            logits = outputs["logits"]
            features = outputs["features"]
            fmaps = outputs["fmaps"]
            lsc_loss = nca(logits, targets)

            if old_model is None:
                return logits, [lsc_loss]

            with torch.no_grad():
                old_outputs = old_model(inputs)
            old_features = old_outputs["features"]
            old_fmaps = old_outputs["fmaps"]

            flat_loss = (
                F.cosine_embedding_loss(
                    features,
                    old_features.detach(),
                    torch.ones(inputs.shape[0]).to(self._device),
                )
                * self.factor
                * self.args.get("lambda_f_base", 1.0)
            )
            spatial_loss = (
                pod_spatial_loss(fmaps, old_fmaps)
                * self.factor
                * self.args.get("lambda_c_base", 5.0)
            )
            return logits, [lsc_loss, flat_loss, spatial_loss]

        return loss_fn

    def _train(self, data_manager, train_loader, test_loader):
        if self._cur_task == 0:
            self.factor = 0.0
        else:
            self.factor = math.sqrt(
                self._total_classes / (self._total_classes - self._known_classes)
            )
        logging.info("Adaptive factor: {}".format(self.factor))

        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        network_ptr = self._network_ptr()
        if self._cur_task == 0:
            network_params = self._network.parameters()
        else:
            ignored_params = list(map(id, network_ptr.fc.fc1.parameters()))
            base_params = filter(
                lambda p: id(p) not in ignored_params, self._network.parameters()
            )
            network_params = [
                {
                    "params": base_params,
                    "lr": self.args["lrate"],
                    "weight_decay": self.args["weight_decay"],
                },
                {
                    "params": network_ptr.fc.fc1.parameters(),
                    "lr": 0,
                    "weight_decay": 0,
                },
            ]

        base_optimizer = optim.SGD(
            network_params,
            lr=self.args["lrate"],
            momentum=0.9,
            weight_decay=self.args["weight_decay"],
        )
        optimizer = self._build_optimizer(base_optimizer)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.args["epochs"]
        )
        self._run(train_loader, test_loader, optimizer, scheduler, self.args["epochs"])
        optimizer.post_process(train_loader)

        if self._cur_task == 0:
            return

        logging.info(
            "Finetune the network (classifier part) with the undersampled dataset!"
        )
        if self._fixed_memory:
            finetune_samples_per_class = self._memory_per_class
            self._construct_exemplar_unified(data_manager, finetune_samples_per_class)
        else:
            finetune_samples_per_class = self._memory_size // self._known_classes
            self._reduce_exemplar(data_manager, finetune_samples_per_class)
            self._construct_exemplar(data_manager, finetune_samples_per_class)

        finetune_train_dataset = data_manager.get_dataset(
            [], source="train", mode="train", appendent=self._get_memory()
        )
        finetune_train_loader = DataLoader(
            finetune_train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=num_workers,
        )
        logging.info(
            "The size of finetune dataset: {}".format(len(finetune_train_dataset))
        )

        network_ptr = self._network_ptr()
        ignored_params = list(map(id, network_ptr.fc.fc1.parameters()))
        base_params = filter(
            lambda p: id(p) not in ignored_params, self._network.parameters()
        )
        network_params = [
            {
                "params": base_params,
                "lr": self.args["ft_lrate"],
                "weight_decay": self.args["weight_decay"],
            },
            {
                "params": network_ptr.fc.fc1.parameters(),
                "lr": 0,
                "weight_decay": 0,
            },
        ]
        base_optimizer = optim.SGD(
            network_params,
            lr=self.args["ft_lrate"],
            momentum=0.9,
            weight_decay=self.args["weight_decay"],
        )
        optimizer = self._build_optimizer(base_optimizer)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.args["ft_epochs"]
        )
        self._run(
            finetune_train_loader,
            test_loader,
            optimizer,
            scheduler,
            self.args["ft_epochs"],
        )
        optimizer.post_process(finetune_train_loader)

        if self._fixed_memory:
            self._data_memory = self._data_memory[
                : -self._memory_per_class * self.task_size
            ]
            self._targets_memory = self._targets_memory[
                : -self._memory_per_class * self.task_size
            ]
            assert (
                len(
                    np.setdiff1d(
                        self._targets_memory, np.arange(0, self._known_classes)
                    )
                )
                == 0
            ), "Exemplar error!"

    def _run(self, train_loader, test_loader, optimizer, scheduler, epochs):
        for epoch in range(1, epochs + 1):
            self._network.train()
            lsc_losses = 0.0
            spatial_losses = 0.0
            flat_losses = 0.0
            correct, total = 0, 0
            for batch_idx, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                loss_fn = self.create_loss_fn(inputs, targets)
                optimizer.set_closure(loss_fn)
                logits, loss_list = optimizer.step()

                lsc_losses += loss_list[0]
                if len(loss_list) > 1:
                    flat_losses += loss_list[1]
                    spatial_losses += loss_list[2]

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler is not None:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = (
                "Task {}, Epoch {}/{} (LR {:.5f}) => "
                "LSC_loss {:.2f}, Spatial_loss {:.2f}, Flat_loss {:.2f}, "
                "Train_acc {:.2f}, Test_acc {:.2f}"
            ).format(
                self._cur_task,
                epoch,
                epochs,
                optimizer.param_groups[0]["lr"],
                lsc_losses / (batch_idx + 1),
                spatial_losses / (batch_idx + 1),
                flat_losses / (batch_idx + 1),
                train_acc,
                test_acc,
            )
            logging.info(info)


def pod_spatial_loss(old_fmaps, fmaps, normalize=True):
    loss = torch.tensor(0.0).to(fmaps[0].device)
    for old_map, fmap in zip(old_fmaps, fmaps):
        assert old_map.shape == fmap.shape, "Shape error"

        old_map = torch.pow(old_map, 2)
        fmap = torch.pow(fmap, 2)

        old_h = old_map.sum(dim=3).view(old_map.shape[0], -1)
        fmap_h = fmap.sum(dim=3).view(fmap.shape[0], -1)
        old_w = old_map.sum(dim=2).view(old_map.shape[0], -1)
        fmap_w = fmap.sum(dim=2).view(fmap.shape[0], -1)

        old_repr = torch.cat([old_h, old_w], dim=-1)
        fmap_repr = torch.cat([fmap_h, fmap_w], dim=-1)

        if normalize:
            old_repr = F.normalize(old_repr, dim=1, p=2)
            fmap_repr = F.normalize(fmap_repr, dim=1, p=2)

        layer_loss = torch.mean(torch.frobenius_norm(old_repr - fmap_repr, dim=-1))
        loss += layer_loss

    return loss / len(fmaps)


def nca(
    similarities,
    targets,
    class_weights=None,
    focal_gamma=None,
    scale=1.0,
    margin=0.6,
    exclude_pos_denominator=True,
    hinge_proxynca=False,
    memory_flags=None,
):
    margins = torch.zeros_like(similarities)
    margins[torch.arange(margins.shape[0]), targets] = margin
    similarities = scale * (similarities - margins)

    if exclude_pos_denominator:
        similarities = similarities - similarities.max(1)[0].view(-1, 1)

        disable_pos = torch.zeros_like(similarities)
        disable_pos[torch.arange(len(similarities)), targets] = similarities[
            torch.arange(len(similarities)), targets
        ]

        numerator = similarities[torch.arange(similarities.shape[0]), targets]
        denominator = similarities - disable_pos

        losses = numerator - torch.log(torch.exp(denominator).sum(-1))
        if class_weights is not None:
            losses = class_weights[targets] * losses

        losses = -losses
        if hinge_proxynca:
            losses = torch.clamp(losses, min=0.0)

        return torch.mean(losses)

    return F.cross_entropy(
        similarities, targets, weight=class_weights, reduction="mean"
    )
