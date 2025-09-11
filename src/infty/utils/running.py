import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch import optim



def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            setattr(module, "backup_momentum", module.momentum)
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = getattr(module, "backup_momentum")

    model.apply(_enable)


@torch.no_grad()
def random_mask_like(tensor, nonzero_ratio, generator=None):
    """
    Return a random mask with the same shape as the input tensor, where the fraction of True is equal to the sparsity.

    Examples
    --------
    >>> random_mask_like(torch.randn(10, 10), 0.1).count_nonzero()
    tensor(10)
    """
    mask = torch.zeros_like(tensor)
    mask.view(-1)[torch.randperm(mask.numel(), generator=generator)[:int(nonzero_ratio * mask.numel())]] = 1
    return mask.bool()


@torch.no_grad()
def fast_random_mask_like(tensor, nonzero_ratio, generator=None):
    """
    A much faster version of random_zero_mask_like, but the sparsity is not guaranteed.

    Examples
    --------
    >>> fast_random_mask_like(torch.randn(10, 10), 0.1).count_nonzero() < 20
    tensor(True)
    """
    mask = torch.empty_like(tensor).normal_(generator=generator) < nonzero_ratio
    return mask.bool()


@torch.no_grad()
def estimate_pretrained_model_magnitude_pruning_threshold(model, global_sparsity):
    """
    Compute the magnitude threshold for pruning based on the global sparsity requirement.
    """
    all_weights = []
    for param in model.parameters():
        all_weights.append(
            param.view(-1).abs().clone().detach().cpu()
        )
    all_weights = torch.cat(all_weights)
    # subsample 102400 elements to estimate the threshold
    sample_size = int(min(1e7, all_weights.numel()))
    print(f"[Sparse gradient] Subsampling {sample_size} elements to estimate the threshold.")
    sub_weights = all_weights[torch.randperm(all_weights.numel())[:sample_size]]
    return torch.quantile(sub_weights.float(), global_sparsity).item()


@torch.no_grad()
def compute_named_parameters_to_sparsity(model, threshold):
    """
    Compute the sparsity of each named parameter in the model.
    """
    named_parameters_to_sparsity = {}
    for name, param in model.named_parameters():
        named_parameters_to_sparsity[name] = param.abs().le(threshold).float().mean().item()
    return named_parameters_to_sparsity


def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def group_add(params, update, alpha=1):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i, p in enumerate(params):
        params[i].data.add_(update[i] * alpha)
    return params


def normalization(v):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    s = group_product(v, v)
    s = s**0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v


def get_params_grad(model):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
    return params, grads


def hessian_vector_product(gradsH, params, v):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """
    hv = torch.autograd.grad(gradsH,
                             params,
                             grad_outputs=v,
                             only_inputs=True,
                             retain_graph=True)
    return hv


def orthnormal(w, v_list):
    """
    make vector w orthogonal to each vector in v_list.
    afterwards, normalize the output w
    """
    for v in v_list:
        w = group_add(w, v, alpha=-group_product(w, v))
    return normalization(w)

