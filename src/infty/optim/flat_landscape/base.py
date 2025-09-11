import torch
import contextlib
from torch.distributed import ReduceOp
from infty.utils.running import enable_running_stats, disable_running_stats

class InftyBaseOptimizer(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, model, adaptive=False, perturb_eps=1e-12, grad_reduce='mean', **kwargs):
        defaults = dict(adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.name = "base"
        self.adaptive = adaptive
        if isinstance(perturb_eps, str):
            self.perturb_eps = float(perturb_eps)
        else:
            self.perturb_eps = perturb_eps
        self.get_grad_reduce(grad_reduce)
        self.rho_scheduler = None
        self.norm_rho_scheduler = None

    @torch.no_grad()
    def update_rho_t(self):
        if getattr(self, 'rho_scheduler', None) is not None:
            self.rho = self.rho_scheduler.step()
        if getattr(self, 'norm_rho_scheduler', None) is not None:
            self.norm_rho = self.norm_rho_scheduler.step()

    def get_grad_reduce(self, grad_reduce: str):
        if grad_reduce.lower() == 'mean':
            if hasattr(ReduceOp, 'AVG'):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else:
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == 'sum':
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce" should be one of ["mean", "sum"].')

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)

    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    def normalized(self, g):
        return g / (g.norm(p=2) + 1e-8)

    @torch.no_grad()
    def _grad_norm(self, by=None, weight_adaptive=False):
        # by: None, 'grad', or other state keys
        if not by:
            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * p.grad).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        else:
            norm = torch.norm(
                torch.stack([
                    ((torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        return norm

    def set_closure(self, loss_fn):
        def get_grad():
            self.zero_grad()
            with torch.enable_grad():
                logits, loss_list = loss_fn()
                total_loss = torch.sum(torch.stack(loss_list))
            total_loss.backward()
            return logits, loss_list
        self.forward_backward_func = get_grad

    @torch.no_grad()
    def perturb_weights(self, scale, state_g_key, ew_key, adaptive=None, accumulate=False):
        if adaptive is None:
            adaptive = self.adaptive
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p][state_g_key] = p.grad.data.clone()
                e_w = p.grad * scale.to(p)
                if adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)
                if accumulate and ew_key in self.state[p]:
                    self.state[p][ew_key] += e_w
                else:
                    self.state[p][ew_key] = e_w

    @torch.no_grad()
    def grad_norm_ascent(self, g0_key="g_0", g1_key="g_1", ew_key="e_w_1_2", adaptive=None):
        if adaptive is None:
            adaptive = self.adaptive
        scale = self.grad_norm_rho if hasattr(self, 'grad_norm_rho') and self.grad_norm_rho is not None else 1.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p][g1_key] = p.grad.data.clone()
                p.grad.data -= self.state[p][g0_key]
        grad_norm = self._grad_norm(weight_adaptive=adaptive)
        scale_val = scale / (grad_norm + self.perturb_eps)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale_val.to(p)
                if adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)
                self.state[p][ew_key] = e_w

    @torch.no_grad()
    def unperturb(self, perturb_key):
        for group in self.param_groups:
            for p in group['params']:
                if perturb_key in self.state[p].keys():
                    p.data.sub_(self.state[p][perturb_key])

    def step(self, closure=None, delay=False):
        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func
            
        with self.maybe_no_sync():
            logits, loss_list = get_grad()

        self._sync_grad()
        if not delay:
            self.base_optimizer.step()
        enable_running_stats(self.model)
        return logits, loss_list

    def delay_step(self):
        self.base_optimizer.step() 

    def post_process(self, train_loader=None):
        pass
