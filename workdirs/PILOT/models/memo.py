import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import copy
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import AdaptiveNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
from utils.core import get_infty_optimizer
from infty import plot as infty_plot
import os
from infty.utils.hessian import hessian
from infty.utils.running import disable_running_stats, enable_running_stats
import json

num_workers=8
EPSILON = 1e-8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._old_base = None
        self._network = AdaptiveNet(args, args["pretrained"])
        logging.info(f'>>> train generalized blocks:{self.args["train_base"]} train_adaptive:{self.args["train_adaptive"]}')

    def after_task(self):
        self._known_classes = self._total_classes
        if self._cur_task == 0:
            if self.args['train_base']:
                logging.info("Train Generalized Blocks...")
                self._network.TaskAgnosticExtractor.train()
                for param in self._network.TaskAgnosticExtractor.parameters():
                    param.requires_grad = True
            else:
                logging.info("Fix Generalized Blocks...")
                self._network.TaskAgnosticExtractor.eval()
                for param in self._network.TaskAgnosticExtractor.parameters():
                    param.requires_grad = False
        
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        if self._cur_task>0:
            for i in range(self._cur_task):
                for p in self._network.AdaptiveExtractors[i].parameters():
                    if self.args['train_adaptive']:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False

        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source='train',
            mode='train', 
            appendent=self._get_memory()
        )
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.args["batch_size"], 
            shuffle=True, 
            num_workers=num_workers
        )
        
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), 
            source='test', 
            mode='test'
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.args["batch_size"],
            shuffle=False, 
            num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
    
    def set_network(self):
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._network.train()                   #All status from eval to train
        if self.args['train_base']:
            self._network.TaskAgnosticExtractor.train()
        else:
            self._network.TaskAgnosticExtractor.eval()
        
        # set adaptive extractor's status
        self._network.AdaptiveExtractors[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                if self.args['train_adaptive']:
                    self._network.AdaptiveExtractors[i].train()
                else:
                    self._network.AdaptiveExtractors[i].eval()
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
    
    def _sharpness_unpack_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                _, inputs, targets = batch
                return inputs, targets
            if len(batch) == 2:
                inputs, targets = batch
                return inputs, targets
        raise ValueError("Unsupported batch format.")

    def _sharpness_trainable_params(self, model):
        return [p for p in model.parameters() if p.requires_grad]

    def _sharpness_random_direction_like(self, model):
        params = self._sharpness_trainable_params(model)
        direction = [torch.randn_like(p, device=self._device) for p in params]
        sq_norm = sum(torch.sum(d * d) for d in direction)
        norm = torch.sqrt(sq_norm) + 1e-12
        direction = [d / norm for d in direction]
        return direction

    def _sharpness_set_perturbed_params(self, model_ori, model_perb, direction, alpha):
        ori_params = self._sharpness_trainable_params(model_ori)
        perb_params = self._sharpness_trainable_params(model_perb)
        for p_ori, p_perb, d in zip(ori_params, perb_params, direction):
            p_perb.data = p_ori.data + alpha * d.to(self._device)
        return model_perb

    @torch.no_grad()
    def _sharpness_evaluate_model(self, model_eval, loader):
        model_eval.eval()

        total_loss = 0.0
        total_num = 0
        total_correct = 0

        for batch in loader:
            inputs, targets = self._sharpness_unpack_batch(batch)
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            loss_fn = self.create_loss_fn(inputs, targets, model=model_eval)
            logits, loss_list = loss_fn()
            loss = sum(loss_list)

            batch_size = targets.size(0)
            total_loss += float(loss.item()) * batch_size
            total_num += batch_size

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == targets).sum().item()

        mean_loss = total_loss / max(total_num, 1)
        mean_acc = total_correct / max(total_num, 1)
        return mean_loss, mean_acc

    def _sharpness_scan_one_direction(self, model, eval_loader, direction, alpha_list):
        model_perb = copy.deepcopy(model).to(self._device)
        model_perb.eval()

        results = []
        for alpha in alpha_list:
            self._sharpness_set_perturbed_params(model, model_perb, direction, float(alpha))
            loss, acc = self._sharpness_evaluate_model(model_perb, eval_loader)
            results.append([float(alpha), float(loss), float(acc)])
        return results

    def _sharpness_build_bidirectional_alphas(self, alpha_list):
        magnitudes = sorted({abs(float(alpha)) for alpha in alpha_list})
        signed_alphas = {-alpha for alpha in magnitudes if alpha > 0.0}
        signed_alphas.update(magnitudes)
        return sorted(signed_alphas)

    def save_sharpness_scan_json(
        self,
        optimizer,
        hessian_loader,
        eval_loader,
        output_dir="./sharpness_json",
        alpha_list=None,
        maxIter=100,
        tol=1e-3,
    ):
        if alpha_list is None:
            alpha_list = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
        alpha_list = self._sharpness_build_bidirectional_alphas(alpha_list)

        os.makedirs(output_dir, exist_ok=True)

        method_name = self.args["inftyopt"]
        seed = self.args["seed"][0] if isinstance(self.args["seed"], (list, tuple)) else self.args["seed"]
        task = self._cur_task

        json_path = os.path.join(
            output_dir,
            f"{method_name}_results_seed{seed}_task{task}.json",
        )

        if os.path.exists(json_path):
            print(f"\033[92m[Sharpness]\033[0m JSON already exists, skip: {json_path}")
            return

        model_for_scan = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        model_for_scan.eval()

        print(f"\033[94m[Sharpness]\033[0m Computing top Hessian eigenpair for task {task} ...")
        hessian_comp = hessian(
            model_for_scan,
            self.create_loss_fn,
            dataloader=hessian_loader,
            device=self._device,
        )
        top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(
            maxIter=maxIter,
            tol=tol,
            top_n=1,
        )

        lambda_max = float(top_eigenvalues[0])
        vmax_direction = [d.detach().clone().to(self._device) for d in top_eigenvector[0]]

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        rand_direction = self._sharpness_random_direction_like(model_for_scan)

        print(f"\033[94m[Sharpness]\033[0m Scanning vmax direction ...")
        res_vmax = self._sharpness_scan_one_direction(
            model=model_for_scan,
            eval_loader=eval_loader,
            direction=vmax_direction,
            alpha_list=alpha_list,
        )

        print(f"\033[94m[Sharpness]\033[0m Scanning random direction ...")
        res_rand = self._sharpness_scan_one_direction(
            model=model_for_scan,
            eval_loader=eval_loader,
            direction=rand_direction,
            alpha_list=alpha_list,
        )

        result = {
            "lambda_max": lambda_max,
            "res_vmax": res_vmax,
            "res_rand": res_rand,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        print(f"\033[92m[Sharpness]\033[0m Saved to: {json_path}")
    
    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task==0:
            base_optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=self.args["init_lr"],
                weight_decay=self.args["init_weight_decay"]
            )
            # TODO (step 1): Wrap your base optimizer with an INFTY optimizer
            run_args={**self.args, "task_id": self._cur_task}
            optimizer = get_infty_optimizer(params=self._network.parameters(), base_optimizer=base_optimizer, model=self._network, args=run_args)
            if self.args['scheduler'] == 'steplr':
                scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, 
                milestones=self.args['init_milestones'], 
                gamma=self.args['init_lr_decay']
            )
            elif self.args['scheduler'] == 'cosine':
                assert self.args['t_max'] is not None
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=self.args['t_max']
                )
            else:   
                raise NotImplementedError
            if not self.args['skip']:
                ckp_path = os.path.join(self.args["ckp_dir"], f"{self.args['model_name']}-{self.args['backbone_type']}-init-seed{self.args['seed']}.pth")
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

                    # Load
                    self._network.load_state_dict(state_dict, strict=False)
                else:
                    print(f"\033[93m[Infty Warning]\033[0m No checkpoint found at {ckp_path}, running init training.")
                    self._init_train(train_loader, test_loader, optimizer, scheduler)
                    os.makedirs(os.path.dirname(ckp_path), exist_ok=True)
                    torch.save({"model": self._network.state_dict()}, ckp_path)
                #self._init_train(train_loader, test_loader, optimizer, scheduler)
            else:
                if isinstance(self._network, nn.DataParallel):
                    self._network = self._network.module
                load_acc = self._network.load_checkpoint(self.args)
                self._network.to(self._device)

                if len(self._multiple_gpus) > 1:
                    self._network = nn.DataParallel(self._network, self._multiple_gpus)
                
                cur_test_acc = self._compute_accuracy(self._network, self.test_loader)
                logging.info(f"Loaded_Test_Acc:{load_acc} Cur_Test_Acc:{cur_test_acc}")
        else:
            base_optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=self.args["lrate"],
                weight_decay=self.args["weight_decay"]
            )
            # TODO (step 1): Wrap your base optimizer with an INFTY optimizer
            run_args={**self.args, "task_id": self._cur_task}
            optimizer = get_infty_optimizer(params=self._network.parameters(), base_optimizer=base_optimizer, model=self._network, args=run_args)
            if self.args['scheduler'] == 'steplr':
                scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, 
                milestones=self.args['milestones'], 
                gamma=self.args['lrate_decay']
            )
            elif self.args['scheduler'] == 'cosine':
                assert self.args['t_max'] is not None
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=self.args['t_max']
                )
            else:   
                raise NotImplementedError
            # ckp_path = os.path.join(self.args["ckp_dir"], f"{self.args['model_name']}-{self.args['backbone_type']}-{self.args['inftyopt']}-task{self._cur_task}.pth")
            # if os.path.exists(ckp_path):
            #     print(f"\033[92m[Infty Info]\033[0m Loading checkpoint from {ckp_path}")
            #     state_dict = torch.load(ckp_path, map_location="cpu")["model"]
            #     model_state = self._network.state_dict()

            #     # Automatically handle "module." prefix matching
            #     new_state_dict = {}

            #     def strip_module_prefix(key): return key.replace("module.", "", 1)
            #     def add_module_prefix(key): return f"module.{key}" if not key.startswith("module.") else key

            #     if list(state_dict.keys())[0].startswith("module.") and not list(model_state.keys())[0].startswith("module."):
            #         # Weights were saved with multi-GPU, current is single-GPU
            #         state_dict = {strip_module_prefix(k): v for k, v in state_dict.items()}
            #     elif not list(state_dict.keys())[0].startswith("module.") and list(model_state.keys())[0].startswith("module."):
            #         # Weights were saved with single-GPU, current is multi-GPU
            #         state_dict = {add_module_prefix(k): v for k, v in state_dict.items()}

            #     # Load
            #     self._network.load_state_dict(state_dict, strict=False)
            # else:
            #     print(f"\033[93m[Infty Warning]\033[0m No checkpoint found at {ckp_path}, running init training.")
            #     self._update_representation(train_loader, test_loader, optimizer, scheduler)
            #     torch.save({"model": self._network.state_dict()}, ckp_path)
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

        # for update the subspace basis
        optimizer.post_process(train_loader)

        if self._cur_task > 0 and self.args.get("enable_weight_align", False):
            if len(self._multiple_gpus) > 1:
                self._network.module.weight_align(self._total_classes - self._known_classes)
            else:
                self._network.weight_align(self._total_classes - self._known_classes)

        # Optional: infty plots
        # infty_plot.visualize_loss_landscape(optimizer, self._network, self.create_loss_fn, train_loader, self._cur_task, self._device)
        # infty_plot.visualize_esd(optimizer, self._network, self.create_loss_fn, train_loader, self._cur_task, self._device)
        # infty_plot.visualize_conflicts(optimizer, self._cur_task)

        if self.args.get("save_sharpness_json", False) and self._cur_task != 0:
            try:
                sharpness_loader = test_loader if self.args.get("sharpness_use_test_loader", True) else train_loader
                self.save_sharpness_scan_json(
                    optimizer=optimizer,
                    hessian_loader=sharpness_loader,
                    eval_loader=sharpness_loader,
                    output_dir=self.args.get("sharpness_json_dir", "./sharpness_json"),
                    alpha_list=self.args.get(
                        "sharpness_alpha_list",
                        [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
                    ),
                    maxIter=self.args.get("sharpness_maxIter", 100),
                    tol=self.args.get("sharpness_tol", 1e-3),
                )
            except Exception as e:
                logging.exception(f"[Sharpness] Failed on task {self._cur_task}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # TODO (step 2): Implement the create_loss_fn function
    def create_loss_fn(self, inputs, targets, model=None):
        """
        Create a closure to calculate the loss
        """
        if model is None:
            model = self._network
        def loss_fn():
            outputs = model(inputs)
            if self._cur_task == 0:
                logits = outputs["logits"]
                loss_clf = F.cross_entropy(logits, targets)
                return logits, [loss_clf]
            else:
                logits, aux_logits = outputs["logits"], outputs["aux_logits"]
                loss_clf = F.cross_entropy(logits, targets)
                aux_targets = targets.clone()
                aux_targets=torch.where(aux_targets-self._known_classes+1>0,  aux_targets-self._known_classes+1,0)
                loss_aux = self.args['alpha_aux'] * F.cross_entropy(aux_logits, aux_targets)
                return logits, [loss_clf, loss_aux]
        return loss_fn

    # WARNING: if use forward_grad instead, 
    # @staticmethod
    @torch.no_grad()
    def create_jvp_loss_fn(self, inputs, targets):
        """
        Create a closure to calculate the loss for jvp
        """
        from torch.func import functional_call
        batch = (inputs, targets)
        def functional_call_loss(params, names, buffers, model, batch):
            params = {k: v for k, v in zip(names, params)}
            (inputs, targets) = batch
            outputs = functional_call(model, (params, buffers), (inputs))
            logits, aux_logits = outputs["logits"], outputs["aux_logits"]
            aux_targets = targets.clone()
            aux_targets = torch.where(aux_targets - self._known_classes + 1 > 0, aux_targets - self._known_classes + 1, 0)
            loss_clf = F.cross_entropy(logits, targets)
            loss_aux = self.args['alpha_aux'] * F.cross_entropy(aux_logits, aux_targets)
            return logits, [loss_clf, loss_aux]
        return functional_call_loss, batch


    def _init_train(self,train_loader,test_loader,optimizer,scheduler):
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                # logits = self._network(inputs)['logits']
                # loss=F.cross_entropy(logits,targets) 
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
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            if epoch%5==0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args['init_epoch'], losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args['init_epoch'], losses/len(train_loader), train_acc)
            prog_bar.set_description(info)
        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["epochs"]))
        task_conflict_records = []
        for _, epoch in enumerate(prog_bar):
            self.set_network()
            losses = 0.
            losses_clf=0.
            losses_aux=0.
            correct, total = 0, 0
            epoch_conflict_metrics = []
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                # outputs= self._network(inputs)
                # logits,aux_logits=outputs["logits"],outputs["aux_logits"]
                # loss_clf=F.cross_entropy(logits,targets)
                # aux_targets = targets.clone()
                # aux_targets=torch.where(aux_targets-self._known_classes+1>0,  aux_targets-self._known_classes+1,0)
                # loss_aux=F.cross_entropy(aux_logits,aux_targets)
                # loss=loss_clf+self.args['alpha_aux']*loss_aux

                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                # losses += loss.item()
                # losses_aux+=loss_aux.item()
                # losses_clf+=loss_clf.item()

                if self._conflict_stats_enabled() and i < self._conflict_probe_batches():
                    disable_running_stats(self._network)
                    probe_loss_fn = self.create_loss_fn(inputs, targets)
                    _, probe_loss_list = probe_loss_fn()
                    probe_metrics = self._compute_conflict_metrics(probe_loss_list, model=self._network)
                    enable_running_stats(self._network)
                    if probe_metrics is not None:
                        epoch_conflict_metrics.append(probe_metrics)

                # TODO (step 3): Use the loss_fn to calculate the loss and backward
                if self.args["inftyopt"] == "forward_grad":  
                    loss_fn, batch = self.create_jvp_loss_fn(inputs, targets)
                    optimizer.set_closure(loss_fn, batch)
                else:
                    loss_fn = self.create_loss_fn(inputs, targets)
                    optimizer.set_closure(loss_fn)
                logits, loss_list = optimizer.step()
                losses += sum(loss_list)
                losses_clf += loss_list[0]
                losses_aux += loss_list[1]


                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)

            if epoch%5==0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux  {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args["epochs"], losses/len(train_loader),losses_clf/len(train_loader),losses_aux/len(train_loader),train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self.args["epochs"], losses/len(train_loader), losses_clf/len(train_loader),losses_aux/len(train_loader),train_acc)
            prog_bar.set_description(info)

            if epoch_conflict_metrics:
                epoch_record = self._aggregate_conflict_epoch_metrics(
                    task=self._cur_task,
                    epoch=epoch + 1,
                    metrics_list=epoch_conflict_metrics,
                )
                if epoch_record is not None:
                    task_conflict_records.append(epoch_record)
        logging.info(info)
        self._save_conflict_task_records(self._cur_task, task_conflict_records)
    
    def save_checkpoint(self, test_acc):
        assert self.args['model_name'] == 'finetune'
        checkpoint_name = f"checkpoints/finetune_{self.args['csv_name']}"
        _checkpoint_cpu = copy.deepcopy(self._network)
        if isinstance(_checkpoint_cpu, nn.DataParallel):
            _checkpoint_cpu = _checkpoint_cpu.module
        _checkpoint_cpu.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "backbone": _checkpoint_cpu.backbone.state_dict(),
            "fc":_checkpoint_cpu.fc.state_dict(),
            "test_acc": test_acc
        }
        torch.save(save_dict, "{}_{}.pkl".format(checkpoint_name, self._cur_task))
    
    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                data, targets, idx_dataset = data_manager.get_dataset(
                    np.arange(class_idx, class_idx + 1),
                    source="train",
                    mode="test",
                    ret_data=True,
                )
                idx_loader = DataLoader(
                    idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4
                )
                vectors, _ = self._extract_vectors(idx_loader)
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                class_mean = np.mean(vectors, axis=0)

                # Select
                selected_exemplars = []
                exemplar_vectors = []  # [n, feature_dim]
                for k in range(1, m + 1):
                    S = np.sum(
                        exemplar_vectors, axis=0
                    )  # [feature_dim] sum of selected exemplars vectors
                    mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                    i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                    selected_exemplars.append(
                        np.array(data[i])
                    )  # New object to avoid passing by inference
                    exemplar_vectors.append(
                        np.array(vectors[i])
                    )  # New object to avoid passing by inference

                    vectors = np.delete(
                        vectors, i, axis=0
                    )  # Remove it to avoid duplicative selection
                    data = np.delete(
                        data, i, axis=0
                    )  # Remove it to avoid duplicative selection
                    
                    if len(vectors) == 0:
                        break
                # uniques = np.unique(selected_exemplars, axis=0)
                # print('Unique elements: {}'.format(len(uniques)))
                selected_exemplars = np.array(selected_exemplars)
                # exemplar_targets = np.full(m, class_idx)
                exemplar_targets = np.full(selected_exemplars.shape[0], class_idx)
                self._data_memory = (
                    np.concatenate((self._data_memory, selected_exemplars))
                    if len(self._data_memory) != 0
                    else selected_exemplars
                )
                self._targets_memory = (
                    np.concatenate((self._targets_memory, exemplar_targets))
                    if len(self._targets_memory) != 0
                    else exemplar_targets
                )

                # Exemplar mean
                idx_dataset = data_manager.get_dataset(
                    [],
                    source="train",
                    mode="test",
                    appendent=(selected_exemplars, exemplar_targets),
                )
                idx_loader = DataLoader(
                    idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4
                )
                vectors, _ = self._extract_vectors(idx_loader)
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                mean = np.mean(vectors, axis=0)
                mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean
