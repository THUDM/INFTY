import torch
from infty.optim.gradient_conflicts.base import EasyCLMultiObjOptimizer
from types import SimpleNamespace

class GradVac(EasyCLMultiObjOptimizer):
    def __init__(self, params, base_optimizer, model, args, **kwargs):
        default_args = {
            "utype": "model-wise",  # model-wise or layer-wise
            "k_idx": [-1],          # the index of the parameters 
            "S_T": [0.1],            # similarity threshold
            "beta": 0.9,            # beta for updating S_T
        }
        merged_args = {**default_args, **args}
        args_ns = SimpleNamespace(**merged_args)
        super().__init__(params, base_optimizer, model, **kwargs)
        self.name = "gradvac"
        self.task_id = args_ns.task_id
        self.k_idx = args_ns.k_idx
        self.utype = args_ns.utype
        self.S_T = torch.tensor(args_ns.S_T, dtype=torch.float32, device=next(model.parameters()).device)
        self.beta = args_ns.beta

        self.sim_arr = []


    def set_k_idx(self):
        """Set parameter indices based on the update type."""
        if self.utype == "model-wise":
            self.k_idx = [-1]
        elif self.utype == "layer-wise":
            self.k_idx = []
            for param in self.model.parameters():
                if param.grad is not None:
                    self.k_idx.append(param.data.numel()) 
            self.S_T = self.S_T * len(self.k_idx)
        else:
            raise ValueError('No support utype {}'.format(self.utype))


    def step(self, closure=None, delay=False):
        """Perform one optimization step."""
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
            grads = self._compute_grad(loss_list, mode='backward') 
            vac_grads = grads.clone()
            for k in range(len(self.k_idx)):
                beg, end = sum(self.k_idx[:k]), sum(self.k_idx[:k + 1])
                if end == -1:
                    end = grads.size()[-1]
                g1 = vac_grads[0, beg:end]
                g2 = vac_grads[1, beg:end]
                norm_g1 = g1.norm()
                norm_g2 = g2.norm()
                s_t = torch.dot(g1, g2) / (norm_g1 * norm_g2 + 1e-8)
                self.sim_arr.append(s_t.cpu().numpy())
                S_T = self.S_T[k]
                if s_t < S_T:
                    w1 = norm_g1 * (S_T * torch.sqrt(1 - s_t ** 2) - s_t * torch.sqrt(1 - S_T ** 2)) / (norm_g2 * torch.sqrt(1 - S_T ** 2) + 1e-8)
                    w2 = norm_g2 * (S_T * torch.sqrt(1 - s_t ** 2) - s_t * torch.sqrt(1 - S_T ** 2)) / (norm_g1 * torch.sqrt(1 - S_T ** 2) + 1e-8)
                    vac_grads[0, beg:end] = g1 + g2 * w1
                    vac_grads[1, beg:end] = g2 + g1 * w2
                    self.S_T[k] = (1 - self.beta) * S_T + self.beta * s_t
            new_grads = vac_grads.sum(0)
            self._reset_grad(new_grads)
        if not delay:
            self.base_optimizer.step()
        return logits, loss_list


    def __repr__(self):
        return f'GradVac({self.base_optimizer.__class__.__name__})'