import torch
from types import SimpleNamespace
from .base import InftyBaseOptimizer      
from infty.utils.running import enable_running_stats, disable_running_stats


class LookSAM(InftyBaseOptimizer):
    def __init__(self, params, base_optimizer, model, args, adaptive=False, perturb_eps=1e-12,
                 grad_reduce='mean', **kwargs):
        default_args = {
            "rho": 0.1,
            "k": 5,
            "alpha": 0.7,
            "rho_scheduler": None,
        }
        merged_args = {**default_args, **args}
        args_ns = SimpleNamespace(**merged_args)
        super().__init__(params, base_optimizer, model, adaptive=adaptive, perturb_eps=perturb_eps, grad_reduce=grad_reduce, **kwargs)
        self.name = "looksam"
        self.rho = args_ns.rho
        self.k = args_ns.k
        self.alpha = args_ns.alpha
        self.rho_scheduler = args_ns.rho_scheduler
        
        self.update_rho_t()
        self.get_grad_reduce(grad_reduce)
        self._step_count = 0

    
    @torch.no_grad()
    def sam_perturb_weights(self):
        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        rho = self.rho if self.rho is not None else 1.0
        scale = rho / (grad_norm + self.perturb_eps)
        super().perturb_weights(scale, state_g_key="old_g", ew_key="e_w", adaptive=self.adaptive, accumulate=False)

    @torch.no_grad()
    def compute_gv(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or 'old_grad' not in self.state[p]:
                    continue
                old_grad = self.state[p]['old_grad']
                new_grad = p.grad
                g_grad_norm = old_grad / (old_grad.norm(p=2) + 1e-8)
                g_new_grad_norm = new_grad / (new_grad.norm(p=2) + 1e-8)
                cos_theta = torch.sum(g_grad_norm * g_new_grad_norm)
                gv = new_grad - new_grad.norm(p=2) * cos_theta * g_grad_norm
                self.state[p]['gv'] = gv

    @torch.no_grad()
    def apply_gv(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or 'gv' not in self.state[p]:
                    continue
                gv = self.state[p]['gv']
                p.grad.add_(self.alpha.to(p) * (p.grad.norm(p=2) / (gv.norm(p=2) + 1e-8)) * gv)

    def step(self, closure=None, delay=False):
        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        logits, loss_list = get_grad()
        self._step_count += 1
        if not self._step_count % self.k:
            self.sam_perturb_weights()
            disable_running_stats(self.model)
            get_grad()
            self.unperturb("e_w")
            self.compute_gv()
        else:
            self.apply_gv()

        self._sync_grad()
        if not delay:
            self.base_optimizer.step()
        enable_running_stats(self.model)
        return logits, loss_list

    def __repr__(self):
        return f'LookSAM({self.base_optimizer.__class__.__name__})'