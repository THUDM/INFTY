import torch
from types import SimpleNamespace
from .base import InftyBaseOptimizer     
from infty.utils.running import enable_running_stats, disable_running_stats


class GAM(InftyBaseOptimizer):
    def __init__(self, params, base_optimizer, model, args, adaptive=False, perturb_eps=1e-12, grad_reduce='mean', **kwargs):
        default_args = {
            "rho": 0.1,
            "norm_rho": 0.02,
            "grad_beta_1": 1,
            "grad_beta_2": -1,
            "grad_beta_3": 1,
            "grad_gamma": 0.3,
            "rho_scheduler": None,
            "norm_rho_scheduler": None,
        }
        merged_args = {**default_args, **args}
        args_ns = SimpleNamespace(**merged_args)
        super().__init__(params, base_optimizer, model, adaptive=adaptive, perturb_eps=perturb_eps, grad_reduce=grad_reduce, **kwargs)
        self.name = "gam"
        self.rho = args_ns.rho
        self.norm_rho = args_ns.norm_rho
        self.grad_beta_1 = args_ns.grad_beta_1
        self.grad_beta_2 = args_ns.grad_beta_2
        self.grad_beta_3 = args_ns.grad_beta_3
        self.grad_gamma = args_ns.grad_gamma
        self.rho_scheduler = args_ns.rho_scheduler
        self.norm_rho_scheduler = args_ns.norm_rho_scheduler

        self.update_rho_t()
        self.get_grad_reduce(grad_reduce)

    @torch.no_grad()
    def gam_perturb_weights(self, perturb_idx: int):
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
    def gradient_decompose(self):
        inner_prod = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                self.state[p]['pro_m'] = self.state[p]['g_0'] + abs(self.grad_beta_2) * self.state[p]['g_2']
                p.grad.data = self.grad_beta_1 * self.state[p]["g_1"] + self.grad_beta_3 * p.grad.data.detach().clone()
                inner_prod += torch.sum(self.state[p]['pro_m'] * p.grad.data)

        new_grad_norm = self._grad_norm()
        old_grad_norm = self._grad_norm(by='pro_m')
        cosine = inner_prod / (new_grad_norm * old_grad_norm + self.perturb_eps)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                vertical = self.state[p]['pro_m'] - cosine * old_grad_norm * p.grad.data / (new_grad_norm + self.perturb_eps)
                p.grad.data.add_(vertical, alpha=-self.grad_gamma)

    def step(self, closure=None, delay=False):
        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        with self.maybe_no_sync():
            logits, loss_list = get_grad()
            self.gam_perturb_weights(perturb_idx=0)
            disable_running_stats(self.model)
            get_grad()
            self.unperturb("e_w_0")
            self.grad_norm_ascent(g0_key="g_0", g1_key="g_1", ew_key="e_w_1_2", adaptive=self.adaptive)
            get_grad()
            self.gam_perturb_weights(perturb_idx=1)
            get_grad()
            self.gradient_decompose()
            self.unperturb("e_w_1_2")

        self._sync_grad()
        if not delay:
            self.base_optimizer.step()
        enable_running_stats(self.model)
        return logits, loss_list

    def __repr__(self):
        return f'GAM({self.base_optimizer.__class__.__name__})'