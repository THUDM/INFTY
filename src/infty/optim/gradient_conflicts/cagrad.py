import torch
import numpy as np
from scipy.optimize import minimize
from infty.optim.gradient_conflicts.base import EasyCLMultiObjOptimizer
from types import SimpleNamespace

class CAGrad(EasyCLMultiObjOptimizer):
    def __init__(self, params, base_optimizer, model, args, **kwargs):
        default_args = {
            "calpha": 0.1,
            "rescale": 0,
        }
        merged_args = {**default_args, **args}
        args_ns = SimpleNamespace(**merged_args)
        super().__init__(params, base_optimizer, model, **kwargs)
        self.name = "cagrad"
        self.calpha = args_ns.calpha
        self.rescale = args_ns.rescale

    def step(self, closure=None, delay=False):
        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_func

        logits, loss_list = get_grad()
        task_num = len(loss_list)
        self._compute_grad_dim()
        grads = self._compute_grad(loss_list, mode='backward')
        
        GG = torch.matmul(grads, grads.t()).cpu()
        g0_norm = (GG.mean()+1e-8).sqrt() 

        x_start = np.ones(task_num) / task_num
        bnds = tuple((0,1) for x in x_start)
        cons=({'type':'eq','fun':lambda x:1-sum(x)})
        A = GG.numpy()
        b = x_start.copy()
        c = (self.calpha*g0_norm+1e-8).item()
        def objfn(x):
            return (x.reshape(1,-1).dot(A).dot(b.reshape(-1,1))+c*np.sqrt(x.reshape(1,-1).dot(A).dot(x.reshape(-1,1))+1e-8)).sum()
        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(self.device)
        gw = (grads * ww.view(-1, 1)).sum(0)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm+1e-8)
        g = grads.mean(0) + lmbda * gw  
        if self.rescale == 0:
            new_grads = g
        elif self.rescale == 1:
            new_grads = g / (1+self.calpha**2)
        elif self.rescale == 2:
            new_grads = g / (1 + self.calpha)
        else:
            raise ValueError('No support rescale type {}'.format(self.rescale))
        self._reset_grad(new_grads)
        if not delay:
            self.base_optimizer.step()
        return logits, loss_list
    
    def __repr__(self):
        return f'CAGrad({self.base_optimizer.__class__.__name__})'