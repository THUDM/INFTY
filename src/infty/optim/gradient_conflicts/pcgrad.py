import torch
import random
from infty.optim.gradient_conflicts.base import EasyCLMultiObjOptimizer
from types import SimpleNamespace

class PCGrad(EasyCLMultiObjOptimizer):
    def __init__(self, params, base_optimizer, model, args, **kwargs):
        default_args = {
        }
        merged_args = {**default_args, **args}
        args_ns = SimpleNamespace(**merged_args)
        super().__init__(params, base_optimizer, model, **kwargs)
        self.name = "pcgrad"
        self.sim_list = []


    def step(self, closure=None, delay=False):
        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_func

        logits, loss_list = get_grad()
        obj_num = len(loss_list)
        self._compute_grad_dim()
        grads = self._compute_grad(loss_list, mode='backward')
        similarity = self.get_similarity(grads)
        self.sim_list.append(similarity.cpu().numpy())

        pc_grads = grads.clone()
        for obj_i in range(obj_num):
            obj_index = list(range(obj_num))
            obj_index.remove(obj_i)
            random.shuffle(obj_index)
            for obj_j in obj_index:
                g_ij = torch.dot(pc_grads[obj_i], grads[obj_j])
                if g_ij < 0:
                    pc_grads[obj_i] -= g_ij * grads[obj_j] / (grads[obj_j].norm().pow(2)+1e-8)
        new_grads = pc_grads.sum(0)
        self._reset_grad(new_grads)
        if not delay:
            self.base_optimizer.step()
        return logits, loss_list

    def __repr__(self):
        return f'PCGrad({self.base_optimizer.__class__.__name__})'

