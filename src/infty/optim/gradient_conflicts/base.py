import torch
import torch.nn as nn
import torch.nn.functional as F

class EasyCLMultiObjOptimizer(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, model, **kwargs):
        super().__init__(params, defaults=kwargs)
        self.params = params
        self.model = model
        self.base_optimizer = base_optimizer
        if not self.base_optimizer.param_groups:
             raise ValueError("base_optimizer has no param_groups. It might not be initialized with parameters.")
        self.param_groups = self.base_optimizer.param_groups
        self.lr = self.base_optimizer.param_groups[0]["lr"]
        if not self.base_optimizer.param_groups[0]['params']:
            raise ValueError("base_optimizer's first param_group has no parameters.")
        self.device = self.base_optimizer.param_groups[0]['params'][0].device
        self.conflict_num = 0
        self.total_num = 0


    def _compute_grad_dim(self):
        if hasattr(self, "grad_index") and self.grad_index:
            return
        self.grad_index = []
        for group in self.param_groups:
            for param in group['params']:
                self.grad_index.append(param.data.numel())
        self.grad_dim = sum(self.grad_index)

    def _grad2vec(self):
        grad = torch.zeros(self.grad_dim, device=self.device)
        count = 0
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    beg = 0 if count == 0 else sum(self.grad_index[:count])
                    end = sum(self.grad_index[:(count+1)])
                    grad[beg:end] = param.grad.data.view(-1)
                count += 1
        return grad
    
    def get_share_params(self):
        params = []
        for group in self.param_groups:
            for param in group['params']:
                if param.requires_grad:
                    params.append(param)
        return params

    def _compute_grad(self, loss_list, mode='backward'):
        task_num = len(loss_list)
        grads = torch.zeros(task_num, self.grad_dim).to(self.device)
        for tn in range(task_num):
            self.zero_grad()
            if mode == 'backward':
                loss_list[tn].backward(retain_graph=True) if (tn+1)!=task_num else loss_list[tn].backward()
                grads[tn] = self._grad2vec()
            elif mode == 'autograd':
                grad = list(torch.autograd.grad(loss_list[tn], self.get_share_params(), retain_graph=True))
                grads[tn] = torch.cat([g.view(-1) for g in grad])
            else:
                raise ValueError(f'No support {mode} mode for gradient computation')
        return grads

    def _reset_grad(self, new_grads):
        count = 0
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    beg = 0 if count == 0 else sum(self.grad_index[:count])
                    end = sum(self.grad_index[:(count+1)])
                    param.grad.data = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
                count += 1

    def get_similarity(self, grads):
        grads_norm = F.normalize(grads, dim=1)  # shape (m, n)

        sim_matrix = grads_norm @ grads_norm.T
        m = grads.size(0)
        mask = torch.eye(m, device=grads.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, 0.0)
        avg_similarity = sim_matrix.sum() / (m * (m - 1))
        return avg_similarity

    def set_closure(self, loss_fn):
        def get_grad(back=False):
            self.zero_grad()
            with torch.enable_grad():
                logits, loss_list = loss_fn()
                if back:
                    sum(loss_list).backward()
            return logits, loss_list
        self.forward_func = get_grad
    
    def step(self, closure=None, delay=False):
        # need to be implemented by subclass
        raise NotImplementedError

    def delay_step(self):
        self.base_optimizer.step() 
            
    def post_process(self, train_loader=None):
        pass

    def zero_grad(self, set_to_none: bool = False):
        self.base_optimizer.zero_grad(set_to_none)
