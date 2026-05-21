import torch
from types import SimpleNamespace

from infty.optim.gradient_filtering.base import EasyCLMultiObjOptimizer


class UniGrad_FS(EasyCLMultiObjOptimizer):
    def __init__(self, params, base_optimizer, model, args, **kwargs):
        default_args = {
            "utype": "model-wise",
            "k_idx": [-1],
            "S_T": [0.1],
            "beta": 0.9,
            "rho": 0.05,
            "perturb_eps": 1e-12,
            "adaptive": False,
        }
        merged_args = {**default_args, **args}
        args_ns = SimpleNamespace(**merged_args)
        super().__init__(params, base_optimizer, model, **kwargs)
        self.name = "unigrad_fs"
        self.task_id = getattr(args_ns, "task_id", 0)
        self.k_idx = args_ns.k_idx
        self.utype = args_ns.utype
        self._s_t_template = torch.tensor(args_ns.S_T, dtype=torch.float32, device=next(model.parameters()).device)
        self.S_T = self._s_t_template.clone()
        self.beta = args_ns.beta

        self.rho = args_ns.rho
        if isinstance(args_ns.perturb_eps, str):
            self.perturb_eps = float(args_ns.perturb_eps)
        else:
            self.perturb_eps = args_ns.perturb_eps
        self.adaptive = args_ns.adaptive

    def _ensure_similarity_thresholds(self, size):
        if self.S_T.numel() == size:
            return
        if self._s_t_template.numel() == 1:
            self.S_T = self._s_t_template.repeat(size)
            return
        if self._s_t_template.numel() == size:
            self.S_T = self._s_t_template.clone()
            return
        raise ValueError(
            f"S_T must have length 1 or match the number of gradient blocks, got {self._s_t_template.numel()} and {size}."
        )

    def set_k_idx(self):
        if self.utype == "model-wise":
            self.k_idx = [-1]
        elif self.utype == "layer-wise":
            self.k_idx = list(self.grad_index)
        else:
            raise ValueError(f'Unsupported utype: {self.utype}. Must be "model-wise" or "layer-wise"')
        self._ensure_similarity_thresholds(len(self.k_idx))

    @torch.no_grad()
    def perturb_weights(self):
        grad_norm = torch.norm(
            torch.stack([
                ((torch.abs(p.data) if self.adaptive else 1.0) * p.grad).norm(p=2)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        scale = self.rho / (grad_norm + self.perturb_eps)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                if "old_g" not in self.state[p]:
                    self.state[p]["old_g"] = torch.zeros_like(p.grad)
                self.state[p]["old_g"].copy_(p.grad.data)

                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p.data, 2)

                p.add_(e_w)

                if "e_w" not in self.state[p]:
                    self.state[p]["e_w"] = torch.zeros_like(e_w)
                self.state[p]["e_w"].copy_(e_w)

    @torch.no_grad()
    def unperturb(self, perturb_key):
        for group in self.param_groups:
            for p in group["params"]:
                if perturb_key in self.state[p]:
                    p.data.sub_(self.state[p][perturb_key])

    def step(self, closure=None, delay=False):
        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_func

        if self.task_id == 0:
            logits, loss_list = get_grad(back=True)
        else:
            get_grad(back=True)
            self.perturb_weights()
            logits, loss_list = get_grad()
            self.unperturb("e_w")

            self._compute_grad_dim()
            self.set_k_idx()
            grads = self._compute_grad(loss_list, mode="backward")
            if len(loss_list) != 2:
                raise ValueError("UniGrad_FS only supports two losses: old_loss and new_loss")
            uni_grads = grads.clone()
            iteration = self._current_conflict_iteration()
            for k in range(len(self.k_idx)):
                beg, end = sum(self.k_idx[:k]), sum(self.k_idx[:k + 1])
                if end == -1:
                    end = grads.size()[-1]
                g1 = uni_grads[0, beg:end].clone()
                g2 = uni_grads[1, beg:end].clone()
                norm_g1 = g1.norm()
                norm_g2 = g2.norm()
                s_t = torch.dot(g1, g2) / (norm_g1 * norm_g2 + 1e-8)
                self._append_similarity(s_t)
                S_T = self.S_T[k]
                g1_new = g1
                g2_new = g2
                w1 = torch.zeros((), dtype=g1.dtype, device=g1.device)
                w2 = torch.zeros((), dtype=g2.dtype, device=g2.device)
                if s_t < S_T:
                    w1 = norm_g1 * (S_T * torch.sqrt(1 - s_t ** 2) - s_t * torch.sqrt(1 - S_T ** 2)) / (
                        norm_g2 * torch.sqrt(1 - S_T ** 2) + 1e-8
                    )
                    w2 = norm_g2 * (S_T * torch.sqrt(1 - s_t ** 2) - s_t * torch.sqrt(1 - S_T ** 2)) / (
                        norm_g1 * torch.sqrt(1 - S_T ** 2) + 1e-8
                    )
                    g1_new = g1 + g2 * w1
                    g2_new = g2 + g1 * w2
                    uni_grads[0, beg:end] = g1_new
                    uni_grads[1, beg:end] = g2_new
                    self.S_T[k] = (1 - self.beta) * S_T + self.beta * s_t
                self._record_conflict_pair(
                    g1,
                    g2,
                    block=k,
                    threshold=S_T,
                    task=self.task_id,
                    iteration=iteration,
                    g_old_after=g1_new,
                    g_new_after=g2_new,
                    w1=w1,
                    w2=w2,
                )
            self._advance_conflict_iteration()
            new_grads = uni_grads.sum(0)
            self._reset_grad(new_grads)
        if not delay:
            self.base_optimizer.step()
        return logits, loss_list

    def __repr__(self):
        return f"UniGrad_FS({self.base_optimizer.__class__.__name__})"
