import numpy as np
import torch
from tqdm import tqdm

from .toy_problem import ZO_EPS


SGD_TRAJECTORY_METHODS = {
    "sgd",
    "zo_sgd",
    "zo_sgd_q4",
    "zo_sgd_sign",
    "zo_sgd_conserve",
    "forward_grad",
}
ADAM_TRAJECTORY_METHODS = {
    "adam",
    "zo_adam",
    "zo_adam_q4",
    "zo_adam_sign",
    "zo_adam_cons",
    "zo_adam_conserve",
}
ADAMW_TRAJECTORY_METHODS = {"adamw"}
ZO_EPS_TRAJECTORY_METHODS = {
    "zo_sgd",
    "zo_sgd_q4",
    "zo_sgd_sign",
    "zo_sgd_conserve",
    "zo_adam",
    "zo_adam_q4",
    "zo_adam_sign",
    "zo_adam_cons",
    "zo_adam_conserve",
}


def mean_grad(problem, x, optimizer):
    _, grads = problem(x, True)
    x.grad = grads.mean(1)
    optimizer.step()
    x.grad = None


def zo_perturb_parameters(x, zo_random_seed, zo_eps, random_seed=None, scaling_factor=1):
    torch.manual_seed(random_seed if random_seed is not None else zo_random_seed)
    z = torch.normal(mean=0, std=1, size=x.data.size(), device=x.data.device, dtype=x.data.dtype)
    x.data = x.data + scaling_factor * z * zo_eps


def zo_forward(problem, x):
    with torch.inference_mode():
        values = problem(x)
    return values[0].detach(), values[1].detach()


def forward_grad(problem, x, optimizer):
    # The toy problem exposes exact gradients, so we can compute the directional
    # derivative of the summed objective without building a functional-call path.
    with torch.enable_grad():
        _, grads = problem(x, True)
    with torch.no_grad():
        z = torch.normal(mean=0, std=1, size=x.data.size(), device=x.data.device, dtype=x.data.dtype)
        total_grad = grads.sum(1).to(x.dtype)
        projected_grad = torch.dot(total_grad, z)
        x.grad = projected_grad * z
        optimizer.step()
        x.grad = None


@torch.no_grad()
def zo_step(problem, x, optimizer, zo_eps=ZO_EPS):
    zo_random_seed = np.random.randint(10000000)
    zo_perturb_parameters(x, zo_random_seed, zo_eps, scaling_factor=1)
    loss1a, loss1b = zo_forward(problem, x)
    zo_perturb_parameters(x, zo_random_seed, zo_eps, scaling_factor=-2)
    loss2a, loss2b = zo_forward(problem, x)
    projected_grada = ((loss1a - loss2a) / (2 * zo_eps)).item()
    projected_gradb = ((loss1b - loss2b) / (2 * zo_eps)).item()
    zo_perturb_parameters(x, zo_random_seed, zo_eps, scaling_factor=1)
    torch.manual_seed(zo_random_seed)
    z = torch.normal(mean=0, std=1, size=x.data.size(), device=x.data.device, dtype=x.data.dtype)
    x.grad = projected_grada * z + projected_gradb * z
    optimizer.step()
    x.grad = None


@torch.no_grad()
def zo_step_sign(problem, x, optimizer, zo_eps=ZO_EPS):
    zo_random_seed = np.random.randint(10000000)
    zo_perturb_parameters(x, zo_random_seed, zo_eps, scaling_factor=1)
    loss1a, loss1b = zo_forward(problem, x)
    zo_perturb_parameters(x, zo_random_seed, zo_eps, scaling_factor=-2)
    loss2a, loss2b = zo_forward(problem, x)
    projected_grada = ((loss1a - loss2a) / (2 * zo_eps)).item()
    projected_gradb = ((loss1b - loss2b) / (2 * zo_eps)).item()
    zo_perturb_parameters(x, zo_random_seed, zo_eps, scaling_factor=1)
    torch.manual_seed(zo_random_seed)
    z = torch.normal(mean=0, std=1, size=x.data.size(), device=x.data.device, dtype=x.data.dtype)
    x.grad = np.sign(projected_grada) * z + np.sign(projected_gradb) * z
    optimizer.step()
    x.grad = None


@torch.no_grad()
def zo_step_q4(problem, x, optimizer, zo_eps=ZO_EPS):
    for query_index in range(4):
        zo_random_seed = np.random.randint(10000000)
        zo_perturb_parameters(x, zo_random_seed, zo_eps, scaling_factor=1)
        loss1a, loss1b = zo_forward(problem, x)
        zo_perturb_parameters(x, zo_random_seed, zo_eps, scaling_factor=-2)
        loss2a, loss2b = zo_forward(problem, x)
        projected_grada = ((loss1a - loss2a) / (2 * zo_eps)).item()
        projected_gradb = ((loss1b - loss2b) / (2 * zo_eps)).item()
        zo_perturb_parameters(x, zo_random_seed, zo_eps, scaling_factor=1)
        torch.manual_seed(zo_random_seed)
        z = torch.normal(mean=0, std=1, size=x.data.size(), device=x.data.device, dtype=x.data.dtype)
        g = projected_grada * z + projected_gradb * z
        if query_index == 0:
            x.grad = g / 4
        else:
            x.grad += g / 4
    optimizer.step()
    x.grad = None
    optimizer.zero_grad()


@torch.no_grad()
def zo_conserv_step(problem, x, optimizer, zo_eps=ZO_EPS):
    loss0a, loss0b = zo_forward(problem, x)
    zo_random_seed = np.random.randint(10000000)
    zo_perturb_parameters(x, zo_random_seed, zo_eps, scaling_factor=1)
    loss1a, loss1b = zo_forward(problem, x)
    zo_perturb_parameters(x, zo_random_seed, zo_eps, scaling_factor=-2)
    loss2a, loss2b = zo_forward(problem, x)
    projected_grada = ((loss1a - loss2a) / (2 * zo_eps)).item()
    projected_gradb = ((loss1b - loss2b) / (2 * zo_eps)).item()
    zo_perturb_parameters(x, zo_random_seed, zo_eps, scaling_factor=1)

    def update_params(sign):
        torch.manual_seed(zo_random_seed)
        z = torch.normal(mean=0, std=1, size=x.data.size(), device=x.data.device, dtype=x.data.dtype)
        x.grad = sign * (projected_grada * z + projected_gradb * z)
        optimizer.step()
        x.grad = None

    update_params(sign=1.0)
    loss1a, loss1b = zo_forward(problem, x)
    update_params(sign=-2.0)
    loss2a, loss2b = zo_forward(problem, x)
    if loss1a + loss1b > loss0a + loss0b:
        if loss0a + loss0b < loss2a + loss2b:
            update_params(sign=1.0)
    else:
        if loss1a + loss1b < loss2a + loss2b:
            update_params(sign=2.0)


SOLVER_MAP = {
    "sgd": mean_grad,
    "adam": mean_grad,
    "adamw": mean_grad,
    "forward_grad": forward_grad,
    "zo_sgd": zo_step,
    "zo_sgd_q4": zo_step_q4,
    "zo_sgd_sign": zo_step_sign,
    "zo_sgd_conserve": zo_conserv_step,
    "zo_adam": zo_step,
    "zo_adam_q4": zo_step_q4,
    "zo_adam_sign": zo_step_sign,
    "zo_adam_cons": zo_conserv_step,
    "zo_adam_conserve": zo_conserv_step,
}


def run_trajectory(problem, optimizer_name, lr, init, n_iter, zo_eps=ZO_EPS):
    trajectory = []
    x = torch.tensor(init)
    x.requires_grad = True
    if optimizer_name in SGD_TRAJECTORY_METHODS:
        optimizer = torch.optim.SGD([x], lr=lr)
    elif optimizer_name in ADAM_TRAJECTORY_METHODS:
        optimizer = torch.optim.Adam([x], lr=lr)
    elif optimizer_name in ADAMW_TRAJECTORY_METHODS:
        optimizer = torch.optim.AdamW([x], lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")

    solver = SOLVER_MAP[optimizer_name]
    for _ in tqdm(range(n_iter)):
        trajectory.append(x.detach().numpy().copy())
        if optimizer_name in ZO_EPS_TRAJECTORY_METHODS:
            solver(problem, x, optimizer, zo_eps=zo_eps)
        else:
            solver(problem, x, optimizer)
    return torch.tensor(np.array(trajectory))
