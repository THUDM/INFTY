import torch
from types import SimpleNamespace
from .base import InftyBaseOptimizer
from infty.utils.running import enable_running_stats, disable_running_stats


class GSAM(InftyBaseOptimizer):
    def __init__(self, params, base_optimizer, model, args, adaptive=False, perturb_eps=1e-12,
                 grad_reduce='mean', **kwargs):
        default_args = {
            "rho": 0.1,
            "alpha": 0.2,
            "rho_scheduler": None,
        }
        merged_args = {**default_args, **args}
        args_ns = SimpleNamespace(**merged_args)
        super().__init__(params, base_optimizer, model, adaptive=adaptive, perturb_eps=perturb_eps, grad_reduce=grad_reduce, **kwargs)
        self.name = "gsam"
        self.rho = args_ns.rho
        self.alpha = args_ns.alpha
        self.rho_scheduler = args_ns.rho_scheduler
        
        self.update_rho_t()
        self.get_grad_reduce(grad_reduce)

    
    @torch.no_grad()
    def gsam_perturb_weights(self):
        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        rho = self.rho if self.rho is not None else 1.0
        scale = rho / (grad_norm + self.perturb_eps)
        super().perturb_weights(scale, state_g_key="old_g", ew_key="e_w", adaptive=self.adaptive, accumulate=False)

    @torch.no_grad()
    def gradient_decompose(self):
        inner_prod = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                inner_prod += torch.sum(self.state[p]['old_g'] * p.grad.data)
        new_grad_norm = self._grad_norm()
        old_grad_norm = self._grad_norm(by='old_g')
        cosine = inner_prod / (new_grad_norm * old_grad_norm + self.perturb_eps)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                vertical = self.state[p]['old_g'] - cosine * old_grad_norm * p.grad.data / (new_grad_norm + self.perturb_eps)
                p.grad.data.add_(vertical, alpha=-self.alpha)

    def step(self, closure=None, delay=False):
        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func
            
        with self.maybe_no_sync():
            logits, loss_list = get_grad()
            self.gsam_perturb_weights()
            disable_running_stats(self.model)
            get_grad()
            self.gradient_decompose()
            self.unperturb("e_w")

        self._sync_grad()
        if not delay:
            self.base_optimizer.step()
        enable_running_stats(self.model)
        return logits, loss_list
    
    def __repr__(self):
        return f'GSAM({self.base_optimizer.__class__.__name__})'