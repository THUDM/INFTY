import torch
from types import SimpleNamespace

from infty.optim.gradient_filtering.base import EasyCLMultiObjOptimizer


class GradVac(EasyCLMultiObjOptimizer):
    def __init__(self, params, base_optimizer, model, args, **kwargs):
        default_args = {
            "utype": "model-wise",
            "k_idx": [-1],
            "S_T": [0.1],
            "beta": 0.9,
        }
        merged_args = {**default_args, **args}
        args_ns = SimpleNamespace(**merged_args)
        super().__init__(params, base_optimizer, model, **kwargs)
        self.name = "gradvac"
        self.task_id = getattr(args_ns, "task_id", 0)
        self.k_idx = args_ns.k_idx
        self.utype = args_ns.utype
        self._s_t_template = torch.tensor(args_ns.S_T, dtype=torch.float32, device=next(model.parameters()).device)
        self.S_T = self._s_t_template.clone()
        self.beta = args_ns.beta

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
            raise ValueError("No support utype {}".format(self.utype))
        self._ensure_similarity_thresholds(len(self.k_idx))

    def step(self, closure=None, delay=False):
        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_func

        if self.task_id == 0:
            logits, loss_list = get_grad(back=True)
        else:
            logits, loss_list = get_grad()
            self._compute_grad_dim()
            self.set_k_idx()
            grads = self._compute_grad(loss_list, mode="backward")
            vac_grads = grads.clone()
            iteration = self._current_conflict_iteration()
            for k in range(len(self.k_idx)):
                beg, end = sum(self.k_idx[:k]), sum(self.k_idx[:k + 1])
                if end == -1:
                    end = grads.size()[-1]
                g1 = vac_grads[0, beg:end].clone()
                g2 = vac_grads[1, beg:end].clone()
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
                    vac_grads[0, beg:end] = g1_new
                    vac_grads[1, beg:end] = g2_new
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
            new_grads = vac_grads.sum(0)
            self._reset_grad(new_grads)
            self._advance_conflict_iteration()
        if not delay:
            self.base_optimizer.step()
        return logits, loss_list

    def __repr__(self):
        return f"GradVac({self.base_optimizer.__class__.__name__})"
