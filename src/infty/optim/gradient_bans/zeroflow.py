import torch
import numpy as np
import torch.nn.functional as F
from functools import partial
from torch._functorch.functional_call import functional_call
from torch._functorch.eager_transforms import jvp
from infty.utils.running import fast_random_mask_like
from types import SimpleNamespace

from torch.optim import Optimizer
from types import SimpleNamespace

class ZeroFlow(Optimizer):
    def __init__(self, params, base_optimizer, model, args, **kwargs):
        defaults = {'lr': 1e-3, **kwargs}
        super().__init__(params, defaults)

        default_args = {
            "q": 1,
            "inftyopt": "zo_sgd",
            "perturbation_mode": "two_side",
            "zo_eps": 0.001,
            "use_history_grad": False,
            "alpha": 0.9,
            "gradient_sparsity": None,
            "memory_efficient": False,
        }
        merged_args = {**default_args, **args}
        args_ns = SimpleNamespace(**merged_args)

        self.q = args_ns.q
        self.zo_eps = args_ns.zo_eps
        self.perturbation_mode = args_ns.perturbation_mode
        self.use_history_grad = args_ns.use_history_grad
        self.alpha = args_ns.alpha
        self.gradient_sparsity = args_ns.gradient_sparsity
        self.memory_efficient = args_ns.memory_efficient

        # generator
        self.sparse_grad_rng = torch.Generator(device='cuda' if torch.cuda.is_available() else 'cpu')
        self.sparse_grad_random_seed = np.random.randint(int(1e9))

        # inject model and base_optimizer
        self.inftyopt = args_ns.inftyopt
        self.name = "zeroflow"
        self.sign = False
        self.model = model
        self.base_optimizer = base_optimizer


    @torch.no_grad()
    def zo_step(self, memory_efficient=False):
        """
        impleneted by MeZO: https://arxiv.org/abs/2305.17333
        only support q=1 for the memory efficiency.
        if you want to implement q>1, need to store random seeds to save memory.
        in addition, we need to set different random seed for different z in the q-loop.
        """
        assert self.q == 1
        self.named_parameters_to_optim = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                param.grad = None

        self.zo_random_seed = np.random.randint(int(1e9))
        self.zo_perturb_parameters(scaling_factor=1, memory_efficient=memory_efficient, initial_perturbation=True)
        logits, loss1 = self.zo_forward()
        if self.perturbation_mode == "one_side":
            self.zo_perturb_parameters(scaling_factor=-1, memory_efficient=memory_efficient)
            _, loss2 = self.zo_forward()
            projected_grad = ((sum(loss1) - sum(loss2)) / self.zo_eps).item()
        elif self.perturbation_mode == "two_side":
            self.zo_perturb_parameters(scaling_factor=-2, memory_efficient=memory_efficient)
            _, loss2 = self.zo_forward()
            projected_grad = ((sum(loss1) - sum(loss2)) / (2 * self.zo_eps)).item()
            self.zo_perturb_parameters(scaling_factor=1, memory_efficient=memory_efficient)
        else:
            raise ValueError(f"Invalid perturbation mode: {self.perturbation_mode}")
        self.set_estimated_grad(projected_grad, memory_efficient=memory_efficient)
        self.base_optimizer.step()
        self.base_optimizer.zero_grad(set_to_none=True)
        return logits, loss1

    @torch.no_grad()
    def zo_step_v1(self, memory_efficient=False):
        """
        support ZO gradient estimation with query > 1
        """
        assert self.q > 1
        self.named_parameters_to_optim = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                param.grad = None
        for i_q in range(self.q):
            self.zo_random_seed = np.random.randint(int(1e9))
            self.zo_perturb_parameters(scaling_factor=1, memory_efficient=memory_efficient, initial_perturbation=True)
            logits, loss1 = self.zo_forward()
            if self.perturbation_mode == "one_side":
                self.zo_perturb_parameters(scaling_factor=-1, memory_efficient=memory_efficient)
                _, loss2 = self.zo_forward()
                projected_grad = ((sum(loss1) - sum(loss2)) / self.zo_eps).item()
            elif self.perturbation_mode == "two_side":
                self.zo_perturb_parameters(scaling_factor=-2, memory_efficient=memory_efficient)
                _, loss2 = self.zo_forward()
                projected_grad = ((sum(loss1) - sum(loss2)) / (2 * self.zo_eps)).item()
                self.zo_perturb_parameters(scaling_factor=1, memory_efficient=memory_efficient)
            else:
                raise ValueError(f"Invalid perturbation mode: {self.perturbation_mode}")
            self.set_estimated_grad(projected_grad, memory_efficient=memory_efficient)
        self.base_optimizer.step()
        self.base_optimizer.zero_grad(set_to_none=True)
        return logits, loss1

    @torch.no_grad()
    def zo_conserv_step(self, memory_efficient=False):
        """
        Conservative update: reject update if loss does not decrease.
        """
        self.named_parameters_to_optim = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                param.grad = None
        logits, loss0 = self.zo_forward()
        self.zo_random_seed = np.random.randint(int(1e9))
        self.zo_perturb_parameters(scaling_factor=1, memory_efficient=memory_efficient, initial_perturbation=True)
        _, loss1 = self.zo_forward()
        if self.perturbation_mode == "one_side":
            self.zo_perturb_parameters(scaling_factor=-1, memory_efficient=memory_efficient)
            _, loss2 = self.zo_forward()
            projected_grad = ((sum(loss1) - sum(loss2)) / self.zo_eps).item()
        elif self.perturbation_mode == "two_side":
            self.zo_perturb_parameters(scaling_factor=-2, memory_efficient=memory_efficient)
            _, loss2 = self.zo_forward()
            projected_grad = ((sum(loss1) - sum(loss2)) / (2 * self.zo_eps)).item()
            self.zo_perturb_parameters(scaling_factor=1, memory_efficient=memory_efficient)
        else:
            raise ValueError(f"Invalid perturbation mode: {self.perturbation_mode}")

        def update_params(projected_grad, cons_scale=1.0):
            self.set_estimated_grad(projected_grad, memory_efficient=memory_efficient, cons_scale=cons_scale)
            self.base_optimizer.step()
            self.base_optimizer.zero_grad(set_to_none=True)
            
        update_params(projected_grad)
        _, loss1 = self.zo_forward()
        update_params(projected_grad, cons_scale=-2.0)
        _, loss2 = self.zo_forward()
        if sum(loss1) > sum(loss0):
            if sum(loss0) < sum(loss2):
                update_params(projected_grad)
        else:
            if sum(loss1) < sum(loss2):
                update_params(projected_grad, cons_scale=2.0)
        return logits, loss1

    def forward_grad_step(self):
        """
        Forward gradient method (Forward Gradient Method), no need for backpropagation.
        Paper: https://arxiv.org/pdf/2202.08587.pdf
        """
        self.named_parameters_to_optim = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                param.grad = None
        self.zo_random_seed = np.random.randint(int(1e9))
        torch.manual_seed(self.zo_random_seed)
        vs = [torch.randn_like(p) for _, p in self.named_parameters_to_optim]
        f = partial(
            self.functional_call_loss,
            names=[n for n, _ in self.named_parameters_to_optim],
            buffers=dict(self.model.named_buffers()),
            model=self.model, batch=self.batch
        )
        with torch.no_grad():
            outputs, jvp_ = jvp(f, (list([p for _, p in self.named_parameters_to_optim]),), (vs,))
        jvp_ = sum(jvp_[1]).detach()
        with torch.no_grad():
            for v, (n, p) in zip(vs, [(n, p) for n, p in self.named_parameters_to_optim]):
                if "bias" not in n and "layer_norm" not in n and "layernorm" not in n:
                    p.data.sub_(self.defaults.get("lr", 1e-3) * (v * jvp_.to(p.device) + self.defaults.get("weight_decay", 0) * p.data))
                else:
                    p.data.sub_(self.defaults.get("lr", 1e-3) * (v * jvp_.to(p.device)))
        return outputs[0], outputs[1]

    def zo_perturb_parameters(self, scaling_factor=1, initial_perturbation=False, memory_efficient=False, random_seed=None):
        # if initial_perturbation:
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            grad_sparsity = self.get_grad_sparsity_by_name(name)
            if grad_sparsity is not None:
                z[fast_random_mask_like(z, grad_sparsity, generator=self.sparse_grad_rng)] = 0
            param.data += z * scaling_factor * self.zo_eps
            if not memory_efficient and initial_perturbation:
                self.state[param]['z'] = z
            
    def set_estimated_grad(self, projected_grad, memory_efficient=False, random_seed=None, cons_scale=1.0):
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        self.sparse_grad_rng.manual_seed(self.sparse_grad_random_seed)
        for name, param in self.named_parameters_to_optim:
            if not memory_efficient:
                z = self.state[param]['z']
            else:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                grad_sparsity = self.get_grad_sparsity_by_name(name)
                if grad_sparsity is not None:
                    z[fast_random_mask_like(z, grad_sparsity, generator=self.sparse_grad_rng)] = 0

            if self.use_history_grad:
                state = self.state[param]
                if "history_grad" not in state:
                    state["history_grad"] = z.clone()
                else:
                    state["history_grad"].mul_(self.alpha).add_(z, alpha=1 - self.alpha)
                smoothed_z = state["history_grad"]
            else:
                smoothed_z = z
            graddiff_times_z = (np.sign(projected_grad) if self.sign else projected_grad) * smoothed_z * cons_scale

            if param.grad is None:
                param.grad = graddiff_times_z / self.q
            else:
                param.grad += graddiff_times_z / self.q

    def zo_forward(self):
        self.model.eval()
        with torch.inference_mode():
            logits, loss_list = self.forward_func()
        return logits, loss_list

    def get_grad_sparsity_by_name(self, name):
        if self.gradient_sparsity is None:
            return None
        elif isinstance(self.gradient_sparsity, float):
            return self.gradient_sparsity
        elif isinstance(self.gradient_sparsity, dict):
            return self.gradient_sparsity.get(name, None)

    @torch.no_grad()
    def set_closure(self, loss_fn, batch=None):
        if batch is None:
            def get_grad():
                self.model.eval()
                with torch.inference_mode():
                    logits, loss_list = loss_fn()
                return logits, loss_list
            self.forward_func = get_grad
        else:
            self.functional_call_loss = loss_fn
            self.batch = batch


    def step(self, closure=None, delay=False):
        assert self.q > 0
        if self.inftyopt in ["zo_sgd", "zo_adam"]:
            if self.q == 1:
                logits, loss_list = self.zo_step(memory_efficient=self.memory_efficient)
            else:
                logits, loss_list = self.zo_step_v1(memory_efficient=self.memory_efficient)
        elif self.inftyopt in ["zo_sgd_sign", "zo_adam_sign"]:
            self.sign = True
            if self.q == 1:
                logits, loss_list = self.zo_step(memory_efficient=self.memory_efficient)
            else:
                logits, loss_list = self.zo_step_v1(memory_efficient=self.memory_efficient)
        elif self.inftyopt in ["zo_sgd_conserve", "zo_adam_conserve"]:
            assert self.q == 1
            logits, loss_list = self.zo_conserv_step(memory_efficient=self.memory_efficient)
        elif self.inftyopt == "forward_grad":
            logits, loss_list = self.forward_grad_step()
        else:
            raise ValueError(f"Invalid inftyopt: {self.inftyopt}")

        self.zo_update()

        return logits, loss_list

    def zo_update(self):
        pass

    def delay_step(self):
        self.base_optimizer.step() 

    def post_process(self, train_loader):
        pass    

    def __repr__(self):
        return f'ZeroFlow({self.inftyopt})'
    
    