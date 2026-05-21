from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm

from infty.optim.gradient_filtering.base import EasyCLMultiObjOptimizer
from infty.utils.memory import Memory

REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class OGD(EasyCLMultiObjOptimizer):
    """Paper-faithful OGD-GTL implementation from Farajtabar et al. (2020).

    The paper stores gradients of the ground-truth logit, orthogonalizes them
    into a basis after each task, and projects future task loss gradients onto
    the orthogonal complement of that basis.
    """

    def __init__(self, params, base_optimizer, model, args, **kwargs):
        default_args = {
            "strategy": "base",
            "pca": False,
            "num_sample_per_task": 20,
            "init_cls": None,
            "increment": None,
        }
        merged_args = {**default_args, **args}
        args_ns = SimpleNamespace(**merged_args)
        super().__init__(params, base_optimizer, model, **kwargs)

        if args_ns.strategy != "base":
            raise NotImplementedError(
                f"OGD original paper implementation only supports the standard setting; "
                f"got strategy='{args_ns.strategy}'."
            )
        if args_ns.pca:
            raise NotImplementedError(
                "PCA compression is not part of the original OGD paper implementation."
            )

        self.task_id = getattr(args_ns, "task_id", 0)
        self.num_sample_per_task = int(args_ns.num_sample_per_task)
        self.init_cls = args_ns.init_cls
        self.increment = args_ns.increment

        basis_path = getattr(args_ns, "basis_path", None)
        if basis_path is None:
            ckp_root = Path(
                getattr(args_ns, "ckp_dir", REPO_ROOT / "workdirs" / "checkpoints")
            ).expanduser()
            basis_path = ckp_root / "ogd_basis.pt"
        self.basis_path = Path(basis_path).expanduser().resolve()
        self.ogd_path = str(self.basis_path)

        self.param_metadata = self._current_param_metadata()
        self.init_ogd_basis()

    def _current_param_metadata(self):
        metadata = []
        offset = 0
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            numel = param.numel()
            metadata.append(
                {
                    "name": name,
                    "numel": numel,
                    "shape": tuple(param.shape),
                    "start": offset,
                    "end": offset + numel,
                }
            )
            offset += numel
        return metadata

    def _metadata_lookup(self, metadata):
        return {entry["name"]: entry for entry in metadata}

    def _num_trainable_params(self):
        if not self.param_metadata:
            return 0
        return self.param_metadata[-1]["end"]

    def _empty_basis(self):
        return torch.empty(self._num_trainable_params(), 0, dtype=torch.float32)

    def _align_basis_to_current_model(self, loaded_basis, loaded_metadata):
        if not loaded_metadata:
            return self._empty_basis()

        current_lookup = self._metadata_lookup(self.param_metadata)
        aligned = torch.zeros(
            self._num_trainable_params(),
            loaded_basis.shape[1],
            dtype=torch.float32,
        )

        for old_entry in loaded_metadata:
            current_entry = current_lookup.get(old_entry["name"])
            if current_entry is None:
                continue

            overlap = min(old_entry["numel"], current_entry["numel"])
            if overlap <= 0:
                continue

            old_start = old_entry["start"]
            current_start = current_entry["start"]
            aligned[current_start: current_start + overlap] = loaded_basis[
                old_start: old_start + overlap
            ]

        return aligned

    def init_ogd_basis(self):
        if self.task_id == 0:
            self.ogd_basis = self._empty_basis()
            self.ogd_basis_ids = {}
            self.task_memory = {}
            self.task_grad_memory = {}
            self.loaded_param_metadata = list(self.param_metadata)
            self._save_basis_state()
            return

        if not self.basis_path.is_file():
            raise FileNotFoundError(f"Missing OGD basis checkpoint: {self.basis_path}")

        data = _load_checkpoint(self.ogd_path)
        loaded_basis = data["ogd_basis"].to(torch.float32)
        self.ogd_basis_ids = data.get("ogd_basis_ids", {})
        self.task_grad_memory = data.get("task_grad_memory", {})
        self.task_memory = data.get("task_memory", {})
        self.loaded_param_metadata = data.get("param_metadata", [])
        self.ogd_basis = self._align_basis_to_current_model(
            loaded_basis, self.loaded_param_metadata
        )

    def _save_basis_state(self):
        self.basis_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "ogd_basis": self.ogd_basis.cpu(),
                "ogd_basis_ids": self.ogd_basis_ids,
                "task_grad_memory": self.task_grad_memory,
                "task_memory": self.task_memory,
                "param_metadata": self.param_metadata,
            },
            self.ogd_path,
        )

    def _trainable_named_parameters(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                yield name, param

    def _flatten_grads(self, grads_by_name):
        flat_parts = []
        for name, param in self._trainable_named_parameters():
            grad = grads_by_name.get(name)
            if grad is None:
                flat_parts.append(
                    torch.zeros_like(param, memory_format=torch.preserve_format).reshape(-1)
                )
            else:
                flat_parts.append(grad.reshape(-1))
        if not flat_parts:
            return torch.empty(0, device=self.device)
        return torch.cat(flat_parts)

    def parameters_to_grad_vector(self):
        grads_by_name = {}
        for name, param in self._trainable_named_parameters():
            grads_by_name[name] = None if param.grad is None else param.grad.detach()
        return self._flatten_grads(grads_by_name)

    def vector_to_grad(self, vec):
        if not isinstance(vec, torch.Tensor):
            raise TypeError(f"expected torch.Tensor, but got: {torch.typename(vec)}")

        pointer = 0
        for _, param in self._trainable_named_parameters():
            num_param = param.numel()
            grad_slice = vec[pointer: pointer + num_param].view_as(param)
            if param.grad is None:
                param.grad = grad_slice.clone().detach()
            else:
                param.grad.copy_(grad_slice)
            pointer += num_param

    def _project_with_basis(self, vec_cpu):
        if self.ogd_basis.shape[1] == 0:
            return torch.zeros_like(vec_cpu)
        coeffs = self.ogd_basis.t().matmul(vec_cpu)
        return self.ogd_basis.matmul(coeffs)

    def project_vec(self, vec):
        if self.ogd_basis.shape[1] == 0:
            return torch.zeros_like(vec)

        vec_cpu = vec.detach().to(device="cpu", dtype=torch.float32)
        proj_cpu = self._project_with_basis(vec_cpu)
        return proj_cpu.to(device=vec.device, dtype=vec.dtype)

    def _compute_gtl_logit_grad_vector(self, inputs, targets):
        if inputs.size(0) != 1 or targets.numel() != 1:
            raise ValueError(
                "OGD-GTL basis construction expects per-sample gradients; use batch_size=1."
            )

        params = [param for _, param in self._trainable_named_parameters()]
        logits = self.model(inputs)["logits"]
        gt_logit = logits.gather(1, targets.view(-1, 1).long()).sum()
        grads = torch.autograd.grad(gt_logit, params, allow_unused=True)
        grads_by_name = {}
        for (name, _), grad in zip(self._trainable_named_parameters(), grads):
            grads_by_name[name] = None if grad is None else grad.detach()
        return self._flatten_grads(grads_by_name).detach().to(device="cpu", dtype=torch.float32)

    def _orthogonalize_vector(self, vec_cpu):
        residual = vec_cpu
        if self.ogd_basis.shape[1] > 0:
            residual = residual - self._project_with_basis(residual)
        norm = residual.norm(p=2)
        if norm <= 1e-12:
            return None
        return residual / norm

    def _current_task_class_range(self):
        if self.init_cls is None or self.increment is None:
            return None
        if self.task_id == 0:
            return 0, int(self.init_cls)
        start = int(self.init_cls) + (int(self.task_id) - 1) * int(self.increment)
        end = start + int(self.increment)
        return start, end

    def _belongs_to_current_task(self, target):
        class_range = self._current_task_class_range()
        if class_range is None:
            return True
        start, end = class_range
        target_value = int(target)
        return start <= target_value < end

    def _sample_current_task_memory(self, train_loader, task_count):
        dataset = train_loader.dataset
        current_task_samples = []
        for index in range(len(dataset)):
            sample = dataset[index]
            target = sample[2]
            if self._belongs_to_current_task(target):
                current_task_samples.append(sample)

        if not current_task_samples:
            raise ValueError(
                f"OGD could not find any current-task samples while building the basis for task {task_count}."
            )

        selected_count = min(len(current_task_samples), self.num_sample_per_task)
        permutation = torch.randperm(len(current_task_samples))[:selected_count]
        sampled_memory = Memory()
        for idx in permutation.tolist():
            sampled_memory.append(current_task_samples[idx])
        self.task_memory[task_count] = sampled_memory
        return sampled_memory

    def _update_mem(self, train_loader, task_count):
        self.model.eval()
        sampled_memory = self._sample_current_task_memory(train_loader, task_count)
        ogd_train_loader = torch.utils.data.DataLoader(
            sampled_memory, batch_size=1, shuffle=False
        )

        task_basis_memory = Memory()
        for _, inputs, targets in tqdm(ogd_train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            basis_vec = self._compute_gtl_logit_grad_vector(inputs, targets)
            basis_vec = self._orthogonalize_vector(basis_vec)
            if basis_vec is None:
                continue
            self.ogd_basis = torch.cat([self.ogd_basis, basis_vec.unsqueeze(1)], dim=1)
            task_basis_memory.append(basis_vec.clone())

        self.task_grad_memory[task_count] = task_basis_memory
        self.param_metadata = self._current_param_metadata()
        self._save_basis_state()

    def step(self, closure=None, delay=False):
        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_func

        logits, loss_list = get_grad(back=True)
        grad_vec = self.parameters_to_grad_vector()

        if self.task_id > 0 and self.ogd_basis.shape[1] > 0:
            proj_grad = self.project_vec(grad_vec)
            new_grad_vec = grad_vec - proj_grad
        else:
            new_grad_vec = grad_vec

        self.vector_to_grad(new_grad_vec)
        self.base_optimizer.step()
        return logits, loss_list

    def post_process(self, train_loader):
        if train_loader is None:
            raise ValueError("train_loader cannot be None when updating OGD basis")
        self._update_mem(train_loader, task_count=self.task_id)

    def __repr__(self):
        return f"OGD({self.base_optimizer.__class__.__name__})"
