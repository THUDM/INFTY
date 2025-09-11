import torch
from infty.optim.gradient_conflicts.base import EasyCLMultiObjOptimizer
from types import SimpleNamespace

class UniGrad_FS(EasyCLMultiObjOptimizer):
    def __init__(self, params, base_optimizer, model, args, **kwargs):
        default_args = {
            "utype": "model-wise",  # model-wise or layer-wise
            "k_idx": [-1],          # the index of the parameters 
            "S_T": [0.1],            # similarity threshold
            "beta": 0.9,            # beta for updating S_T
            "rho": 0.05,             # perturbation scale
            "perturb_eps": 1e-12,   # epsilon for numerical stability
            "adaptive": False,      # whether to use adaptive perturbation
        }
        merged_args = {**default_args, **args}
        args_ns = SimpleNamespace(**merged_args)
        super().__init__(params, base_optimizer, model, **kwargs)
        self.name = "unigrad_fs"
        self.task_id = args_ns.task_id
        self.k_idx = args_ns.k_idx
        self.utype = args_ns.utype
        self.S_T = torch.tensor(args_ns.S_T, dtype=torch.float32, device=next(model.parameters()).device)
        self.beta = args_ns.beta

        self.rho = args_ns.rho
        if isinstance(args_ns.perturb_eps, str):
            self.perturb_eps = float(args_ns.perturb_eps)
        else:
            self.perturb_eps = args_ns.perturb_eps
        self.adaptive = args_ns.adaptive
        self.sim_arr = []


    def set_k_idx(self):
        """Set parameter indices based on the update type."""
        if self.utype == "model-wise":
            self.k_idx = [-1]
        elif self.utype == "layer-wise":
            self.k_idx = []
            for param in self.model.parameters():
                if param.requires_grad:
                    self.k_idx.append(param.data.numel()) 
            self.S_T = self.S_T * len(self.k_idx)
        else:
            raise ValueError(f'Unsupported utype: {self.utype}. Must be "model-wise" or "layer-wise"')

    @torch.no_grad()
    def perturb_weights(self):
        """Perturb model weights for gradient estimation."""
        grad_norm = torch.norm(
            torch.stack([
                ((torch.abs(p.data) if self.adaptive else 1.0) * p.grad).norm(p=2)
                for group in self.param_groups 
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2
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
        """Remove perturbation from model weights."""
        for group in self.param_groups:
            for p in group['params']:
                if perturb_key in self.state[p]:
                    p.data.sub_(self.state[p][perturb_key])
    
    def step(self, closure=None, delay=False):
        """Perform one optimization step."""
        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_func

        if self.task_id == 0:
            logits, loss_list = get_grad(back=True)
        else:
            get_grad(back=True)
            # flat sharpness
            self.perturb_weights()
            logits, loss_list = get_grad()
            self.unperturb("e_w")
            
            self._compute_grad_dim()
            self.set_k_idx()
            grads = self._compute_grad(loss_list, mode='backward') 
            if len(loss_list) != 2:
                raise ValueError("UniGrad_FS only supports two losses: old_loss and new_loss")
            uni_grads = grads.clone()
            for k in range(len(self.k_idx)):
                beg, end = sum(self.k_idx[:k]), sum(self.k_idx[:k + 1])
                if end == -1:
                    end = grads.size()[-1]
                g1 = uni_grads[0, beg:end]
                g2 = uni_grads[1, beg:end]
                norm_g1 = g1.norm()
                norm_g2 = g2.norm()
                s_t = torch.dot(g1, g2) / (norm_g1 * norm_g2 + 1e-8)
                self.sim_arr.append(s_t.cpu().numpy())
                S_T = self.S_T[k]
                if s_t < S_T:
                    w1 = norm_g1 * (S_T * torch.sqrt(1 - s_t ** 2) - s_t * torch.sqrt(1 - S_T ** 2)) / (norm_g2 * torch.sqrt(1 - S_T ** 2) + 1e-8)
                    w2 = norm_g2 * (S_T * torch.sqrt(1 - s_t ** 2) - s_t * torch.sqrt(1 - S_T ** 2)) / (norm_g1 * torch.sqrt(1 - S_T ** 2) + 1e-8)
                    uni_grads[0, beg:end] = g1 + g2 * w1
                    uni_grads[1, beg:end] = g2 + g1 * w2
                    self.S_T[k] = (1 - self.beta) * S_T + self.beta * s_t
            new_grads = uni_grads.sum(0)
            self._reset_grad(new_grads)
        if not delay:
            self.base_optimizer.step()
        return logits, loss_list


    def __repr__(self):
        return f'UniGrad_FS({self.base_optimizer.__class__.__name__})'