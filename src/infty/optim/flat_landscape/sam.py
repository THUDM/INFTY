import torch    
from types import SimpleNamespace
from .base import InftyBaseOptimizer
from infty.utils.running import enable_running_stats, disable_running_stats


class SAM(InftyBaseOptimizer):
    def __init__(self, params, base_optimizer, model, args, adaptive=False, perturb_eps=1e-12,
                 grad_reduce='mean', **kwargs):
        default_args = {
            "rho": 0.1,
            "rho_scheduler": None,
        }
        merged_args = {**default_args, **args}
        args_ns = SimpleNamespace(**merged_args)
        super().__init__(params, base_optimizer, model, adaptive=adaptive, perturb_eps=perturb_eps, grad_reduce=grad_reduce, **kwargs)
        self.name = "sam"
        self.rho = args_ns.rho
        self.rho_scheduler = args_ns.rho_scheduler
        
        self.update_rho_t()
        self.get_grad_reduce(grad_reduce)

    
    
    @torch.no_grad()
    def sam_perturb_weights(self):
        grad_norm = self._grad_norm(weight_adaptive=self.adaptive)
        rho = self.rho if self.rho is not None else 1.0
        scale = rho / (grad_norm + self.perturb_eps)
        super().perturb_weights(scale, state_g_key="old_g", ew_key="e_w", adaptive=self.adaptive, accumulate=False)


    def step(self, closure=None, delay=False):
        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func
            
        with self.maybe_no_sync():
            logits, loss_list = get_grad()
            self.sam_perturb_weights()
            disable_running_stats(self.model)
            get_grad()
            self.unperturb("e_w")

        self._sync_grad()
        if not delay:
            self.base_optimizer.step()
        enable_running_stats(self.model)
        return logits, loss_list

    def __repr__(self):
        return f'SAM({self.base_optimizer.__class__.__name__})'