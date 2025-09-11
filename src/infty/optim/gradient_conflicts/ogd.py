import torch
from torch.nn.utils.convert_parameters import _check_param_device
from collections import defaultdict
from tqdm import tqdm
from infty.optim.gradient_conflicts.base import EasyCLMultiObjOptimizer
from infty.utils.memory import Memory
from types import SimpleNamespace
from torch.nn import functional as F
import os

class OGD(EasyCLMultiObjOptimizer):
    def __init__(self, params, base_optimizer, model, args, **kwargs):
        default_args = {
            "strategy": "base",
            "pca": False,
            "num_sample_per_task": 20,
        }
        merged_args = {**default_args, **args}
        args_ns = SimpleNamespace(**merged_args)
        super().__init__(params, base_optimizer, model, **kwargs)
        self.ogd_path = "./ckp/ogd_basis.pt"
        self.task_id = args_ns.task_id if hasattr(args_ns, "task_id") else 0

        self.init_ogd_basis()
        self.strategy = args_ns.strategy
        self.pca = args_ns.pca
        
        self.num_sample_per_task = args_ns.num_sample_per_task
            

    def init_ogd_basis(self):
        if self.task_id == 0:
            self.ogd_basis = torch.empty(self.get_n_trainable(), 0)
            self.ogd_basis_ids = {}
            self.task_memory = {}
            self.task_grad_memory = {}

            os.makedirs(os.path.dirname(self.ogd_path), exist_ok=True)
            torch.save({
                "ogd_basis": self.ogd_basis,
                "ogd_basis_ids": self.ogd_basis_ids,
                "task_grad_memory": self.task_grad_memory,
                "task_memory": self.task_memory,
            }, self.ogd_path)
        else:
            data = torch.load(self.ogd_path)
            self.ogd_basis = data["ogd_basis"]
            self.ogd_basis_ids = data["ogd_basis_ids"]
            self.task_grad_memory = data["task_grad_memory"]
            self.task_memory = data["task_memory"]

    def parameters_to_vector(self):
        param_device = None
        vec = []
        for name, param in self.model.named_parameters(): 
            if param.requires_grad and param.grad is not None:
                param_device = _check_param_device(param, param_device)
                vec.append(param.view(-1))
        return torch.cat(vec)

    def parameters_to_grad_vector(self):
        param_device = None
        vec = []
        for name, param in self.model.named_parameters(): 
            if param.requires_grad and param.grad is not None:
                param_device = _check_param_device(param, param_device)
                vec.append(param.grad.view(-1))
        return torch.cat(vec)

    def vector_to_parameters(self, vec):
        if not isinstance(vec, torch.Tensor):
            raise TypeError(f"expected torch.Tensor, but got: {torch.typename(vec)}")
        param_device = None
        pointer = 0

        for name, param in self.model.named_parameters(): 
            if param.requires_grad and param.grad is not None:
                param_device = _check_param_device(param, param_device)
                num_param = param.numel()
                # param.data = vec[pointer: pointer + num_param].view_as(param).detach().clone()
                param.data.copy_(vec[pointer: pointer + num_param].view_as(param))
                pointer += num_param

    def vector_to_grad(self, vec):
        if not isinstance(vec, torch.Tensor):
            raise TypeError(f"expected torch.Tensor, but got: {torch.typename(vec)}")
        param_device = None
        pointer = 0

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_device = _check_param_device(param, param_device)
                num_param = param.numel()
                # 将 vec 映射回梯度
                grad_slice = vec[pointer: pointer + num_param].view_as(param)
                if param.grad is None:
                    param.grad = grad_slice.clone().detach()
                else:
                    param.grad.copy_(grad_slice)
                pointer += num_param

    def project_vec(self, vec, proj_basis):
        if proj_basis.shape[1] == 0:
            return torch.zeros_like(vec)

        proj_basis = proj_basis.to(vec.device)
        proj = torch.mv(proj_basis, torch.mv(proj_basis.t(), vec))
        return proj

    def _get_new_ogd_basis(self, train_loader):
        self.model.eval()
        new_basis = []
        for i, (_, inputs, targets) in tqdm(enumerate(train_loader)):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.model.zero_grad()

            logits = self.model(inputs)["logits"]
            # TODO: should we involve other regularization terms?
            loss = F.cross_entropy(logits, targets)
            loss.backward()

            grad_vec = self.parameters_to_grad_vector()
            new_basis.append(grad_vec[:self.get_n_trainable()])
        new_basis_tensor = torch.stack(new_basis).T
        return new_basis_tensor

    # WARNING: shared backbone layers
    def get_n_trainable(self):
        return sum(param.numel() for name, param in self.model.named_parameters()
                if param.requires_grad and "backbone" in name)

    def _update_mem(self, train_loader, task_count):  
        self.model.eval() 
        
        # 1. Randomly select new task samples to store in memory
        self.task_memory[task_count] = Memory()
        randind = torch.randperm(len(train_loader.dataset))[:self.num_sample_per_task]
        for ind in randind:
            self.task_memory[task_count].append(train_loader.dataset[ind])

        # 2. Get non-orthogonal gradient basis for new task
        if self.strategy == "base":
            ogd_train_loader = torch.utils.data.DataLoader(
                self.task_memory[task_count], batch_size=1, shuffle=False)
        elif self.strategy == "plus":
            all_task_memory = []
            for task_id, mem in self.task_memory.items():
                all_task_memory.extend(mem)
            ogd_memory = Memory()
            for obs in all_task_memory:
                ogd_memory.append(obs)
            ogd_train_loader = torch.utils.data.DataLoader(
                ogd_memory, batch_size=1, shuffle=False)
        
        new_basis_tensor = self._get_new_ogd_basis(ogd_train_loader).cpu()
        # 3. incremental orthogonalization (only process new basis)
        new_basis_ortho = torch.zeros_like(new_basis_tensor)
        for i in range(new_basis_tensor.shape[1]):
            v = new_basis_tensor[:, i]
            # Project onto existing space
            if self.ogd_basis.shape[1] > 0:
                proj = torch.mv(self.ogd_basis, torch.mv(self.ogd_basis.t(), v))
                v = v - proj
            # Normalize and add
            norm_v = torch.norm(v)
            if norm_v < 1e-8:
                v = v / (1e-8)
            else:
                v = v / norm_v
            new_basis_ortho[:, i] = v
            self.ogd_basis = torch.cat([self.ogd_basis, v.unsqueeze(1)], dim=1)
        
        # 4. Store new basis
        self.task_grad_memory[task_count] = Memory()
        for i in range(new_basis_ortho.shape[1]):
            self.task_grad_memory[task_count].append(new_basis_ortho[:, i])
        
        # 5. Save updated basis
        os.makedirs(os.path.dirname(self.ogd_path), exist_ok=True)
        torch.save({
            "ogd_basis": self.ogd_basis,
            "ogd_basis_ids": self.ogd_basis_ids,
            "task_grad_memory": self.task_grad_memory,
            "task_memory": self.task_memory,
        }, self.ogd_path)

        try:
            cond = torch.linalg.cond(self.ogd_basis).item()
        except:
            cond = float("inf")

        print(f"[OGD] Updated basis: shape={self.ogd_basis.shape}, "
            f"norm={self.ogd_basis.norm():.4f}, cond={cond:.4f}")

    def get_backbone_indices(self):
        end_idx = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if "backbone" in name:
                    end_idx += param.numel()
        return end_idx

    def step(self, closure=None, delay=False):
        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_func
        logits, loss_list = get_grad(back=True)
        param_vec = self.parameters_to_vector()
        grad_vec = self.parameters_to_grad_vector()
        if self.strategy in ["base", "plus"] and self.task_id > 0:
            backbone_end = self.get_backbone_indices()
            grad_vec_proj = grad_vec.clone()
            backbone_grad = grad_vec[:backbone_end]
            
            ogd_basis = self.ogd_basis.to(grad_vec.device)
            
            # Project onto orthogonal complement space
            proj_backbone_grad = self.project_vec(backbone_grad, proj_basis=ogd_basis)
            grad_vec_proj[:backbone_end] = backbone_grad - proj_backbone_grad
            new_grad_vec = grad_vec_proj
        else:
            new_grad_vec = grad_vec

        # param_vec -= self.lr * new_grad_vec
        # self.vector_to_parameters(param_vec)
        self.vector_to_grad(new_grad_vec)
        self.base_optimizer.step()
        return logits, loss_list

    def post_process(self, train_loader):
        if train_loader is None:
            raise ValueError("train_loader cannot be None when updating OGD basis")
        self._update_mem(train_loader, task_count=self.task_id)
    
    def __repr__(self):
        return f'OGD({self.base_optimizer.__class__.__name__})'