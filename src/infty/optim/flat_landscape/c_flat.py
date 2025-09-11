import torch
from types import SimpleNamespace
from .base import InftyBaseOptimizer
from infty.utils.running import enable_running_stats, disable_running_stats
import numpy as np


class C_Flat(InftyBaseOptimizer):
    def __init__(self, params, base_optimizer, model, args, adaptive=False, perturb_eps=1e-12,
                 grad_reduce='mean', **kwargs):
        default_args = {
            "strategy": "basic",
            "rho": 0.1,
            "lamb": 0.2,
            "rho_scheduler": None,
            "A": 5.0,
            "k": 0.01,
            "t0": 80,
            "cof": 1.0,
        }
        merged_args = {**default_args, **args}
        args_ns = SimpleNamespace(**merged_args)
        super().__init__(params, base_optimizer, model, adaptive=adaptive, perturb_eps=perturb_eps, grad_reduce=grad_reduce, **kwargs)
        self.name = "c_flat"
        self.strategy = args_ns.strategy
        self.rho = args_ns.rho
        self.lamb = args_ns.lamb
        self.rho_scheduler = args_ns.rho_scheduler
        self.A = args_ns.A
        self.k = args_ns.k
        self.t0 = args_ns.t0
        self.cof = args_ns.cof
        
        self.update_rho_t()
        self.get_grad_reduce(grad_reduce)
        self._step_count = 0

    @torch.no_grad()
    def cflat_perturb_weights(self, perturb_idx: int):
        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        rho = self.rho if self.rho is not None else 1.0
        scale = rho / (grad_norm + self.perturb_eps)
        if perturb_idx == 0:
            super().perturb_weights(scale, state_g_key="g_0", ew_key="e_w_0", adaptive=self.adaptive, accumulate=False)
        elif perturb_idx == 1:
            super().perturb_weights(scale, state_g_key="g_2", ew_key="e_w_1_2", adaptive=self.adaptive, accumulate=True)
        else:
            raise ValueError('"perturb_idx" should be one of [0, 1].')

    @torch.no_grad()
    def gradient_aggregation(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                p.grad.data = self.state[p]['g_1'] + self.lamb * (p.grad.data.detach().clone() - self.state[p]['g_2'])
    
    @torch.no_grad()
    def get_grad_l2_norm(self, model):
        grad_list = []
        for param in model.parameters():
            if param.grad is not None:
                grad_list.append(param.grad.view(-1))
        if len(grad_list) == 0:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        full_grad = torch.cat(grad_list)
        grad_norm = torch.norm(full_grad, p=2)
        return grad_norm

    def _cflat_step_common(self, get_grad):
        self.cflat_perturb_weights(perturb_idx=0)
        disable_running_stats(self.model)
        get_grad()
        self.unperturb("e_w_0")
        self.grad_norm_ascent(g0_key="g_0", g1_key="g_1", ew_key="e_w_1_2", adaptive=self.adaptive)
        get_grad()
        self.cflat_perturb_weights(perturb_idx=1)
        get_grad()
        self.gradient_aggregation()
        self.unperturb("e_w_1_2")

    def step(self, closure=None, delay=False):
        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        self._step_count += 1
        with self.maybe_no_sync():
            logits, loss_list = get_grad()
            if self.strategy == "plus":
                grad_norm = self.get_grad_l2_norm(self.model)
                E = self.A / (1 + np.exp(-self.k * (self._step_count - self.t0))) - grad_norm
                self.A = self.A - self.cof * E
                if E <= 0:
                    self._cflat_step_common(get_grad)
            elif self.strategy == "basic":
                self._cflat_step_common(get_grad)
            else:
                raise ValueError(f'Invalid strategy: {self.strategy}')
        self._sync_grad()
        if not delay:
            self.base_optimizer.step()
        enable_running_stats(self.model)
        return logits, loss_list
    
    def __repr__(self):
        return f'C_Flat({self.base_optimizer.__class__.__name__})'

