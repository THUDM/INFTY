from itertools import combinations

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
        self.sim_arr = []
        self.conflict_records = []
        self._conflict_iteration = 0


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

    @property
    def sim_list(self):
        return self.sim_arr

    def _to_python_scalar(self, value):
        if torch.is_tensor(value):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return value.detach().cpu().tolist()
        return value

    def _safe_float(self, value):
        return float(self._to_python_scalar(value))

    def _append_similarity(self, value):
        self.sim_arr.append(self._safe_float(value))

    def _current_conflict_iteration(self):
        return int(self._conflict_iteration)

    def _advance_conflict_iteration(self):
        self._conflict_iteration += 1

    def _build_conflict_record(
        self,
        g_old,
        g_new,
        *,
        block=0,
        threshold=None,
        task=None,
        iteration=None,
        g_old_after=None,
        g_new_after=None,
        update_vector=None,
        w1=0.0,
        w2=0.0,
        metadata=None,
    ):
        eps = 1e-8
        if task is None:
            task = getattr(self, "task_id", 0)
        if iteration is None:
            iteration = self._current_conflict_iteration()

        g_old = g_old.detach()
        g_new = g_new.detach()
        norm_old = g_old.norm()
        norm_new = g_new.norm()
        cos_before = torch.dot(g_old, g_new) / (norm_old * norm_new + eps)

        if threshold is None:
            threshold_value = 0.0
            below_threshold = bool((cos_before < 0).detach().cpu().item())
        else:
            threshold_value = self._safe_float(threshold)
            below_threshold = bool((cos_before < threshold).detach().cpu().item())

        if g_old_after is None:
            g_old_after = g_old
        else:
            g_old_after = g_old_after.detach()
        if g_new_after is None:
            g_new_after = g_new
        else:
            g_new_after = g_new_after.detach()

        cos_after = torch.dot(g_old_after, g_new_after) / (g_old_after.norm() * g_new_after.norm() + eps)
        if update_vector is None:
            update_vector = g_old_after + g_new_after
        update_vector = update_vector.detach()
        gain_old = torch.dot(g_old, update_vector) / (norm_old + eps)
        gain_new = torch.dot(g_new, update_vector) / (norm_new + eps)

        record = {
            "task": int(task),
            "iter": int(iteration),
            "block": int(block),
            "cos_before": self._safe_float(cos_before),
            "cos_after": self._safe_float(cos_after),
            "delta_cos": self._safe_float(cos_after - cos_before),
            "norm_old": self._safe_float(norm_old),
            "norm_new": self._safe_float(norm_new),
            "norm_ratio_old_new": self._safe_float(norm_old / (norm_new + eps)),
            "threshold": float(threshold_value),
            "conflict": bool((cos_before < 0).detach().cpu().item()),
            "below_threshold": below_threshold,
            "gain_old": self._safe_float(gain_old),
            "gain_new": self._safe_float(gain_new),
            "min_gain": self._safe_float(torch.minimum(gain_old, gain_new)),
            "w1": self._safe_float(w1),
            "w2": self._safe_float(w2),
        }
        if metadata:
            for key, value in metadata.items():
                record[key] = self._to_python_scalar(value)
        return record

    def _record_conflict_pair(self, g_old, g_new, **kwargs):
        record = self._build_conflict_record(g_old, g_new, **kwargs)
        self.conflict_records.append(record)
        return record

    def _record_pairwise_conflicts(
        self,
        grads_before,
        *,
        grads_after=None,
        threshold=None,
        task=None,
        iteration=None,
        block=0,
        update_vector=None,
        metadata=None,
    ):
        if grads_before.ndim != 2:
            raise ValueError(f"grads_before must be a 2D tensor, got shape {tuple(grads_before.shape)}.")

        if grads_after is not None and grads_after.shape != grads_before.shape:
            raise ValueError(
                "grads_after must match grads_before shape, "
                f"got {tuple(grads_after.shape)} and {tuple(grads_before.shape)}."
            )

        pair_records = []
        for source_old, source_new in combinations(range(grads_before.size(0)), 2):
            pair_metadata = {"source_old": int(source_old), "source_new": int(source_new), "pair": f"{source_old}-{source_new}"}
            if metadata:
                pair_metadata.update(metadata)
            pair_records.append(
                self._record_conflict_pair(
                    grads_before[source_old],
                    grads_before[source_new],
                    block=block,
                    threshold=threshold,
                    task=task,
                    iteration=iteration,
                    g_old_after=None if grads_after is None else grads_after[source_old],
                    g_new_after=None if grads_after is None else grads_after[source_new],
                    update_vector=update_vector,
                    metadata=pair_metadata,
                )
            )
        return pair_records

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
