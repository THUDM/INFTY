# User Guide

This guide explains how to use INFTY in continual-learning experiments and custom PyTorch training pipelines.

## What INFTY provides

INFTY exposes two main types of functionality:

1. **Optimizers** under `infty.optim`;
2. **Visualization utilities** under `infty.plot`.

The optimizer module contains three algorithm families:

| Family | Main purpose | Public classes |
| --- | --- | --- |
| Geometry reshaping | Improve flatness/generalization and reshape local loss geometry. | `C_Flat`, `SAM`, `GSAM`, `GAM`, `LookSAM` |
| Zeroth-order updates | Estimate update directions without standard backward gradients. | `ZeroFlow` |
| Gradient filtering | Mitigate interference among objectives or tasks. | `UniGrad_FS`, `GradVac`, `OGD`, `PCGrad`, `CAGrad` |

## Optimizer wrapper model

INFTY optimizers usually wrap a base PyTorch optimizer.

```python
base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = infty_optim.C_Flat(
    params=model.parameters(),
    base_optimizer=base_optimizer,
    model=model,
    args={"rho": 0.05, "lamb": 0.2},
)
```

The base optimizer controls learning rate, parameter groups, weight decay, momentum, and other ordinary PyTorch optimizer settings. The INFTY wrapper controls the continual-learning or geometry-aware update logic.

## Closure contract

A valid closure returns:

```python
return logits, loss_list
```

where:

- `logits` is the model output or a task-specific output object;
- `loss_list` is a list of scalar PyTorch tensors;
- `sum(loss_list)` should be the total training objective unless the optimizer explicitly consumes separate losses.

### Single-objective closure

```python
def make_loss_fn(inputs, targets):
    def loss_fn():
        logits = model(inputs)
        loss = criterion(logits, targets)
        return logits, [loss]
    return loss_fn
```

### Multi-objective closure

```python
def make_loss_fn(inputs, new_targets, replay_targets):
    def loss_fn():
        logits = model(inputs)
        old_task_loss = criterion(logits, replay_targets)
        new_task_loss = criterion(logits, new_targets)
        return logits, [old_task_loss, new_task_loss]
    return loss_fn
```

For `UniGrad_FS`, non-first tasks require exactly two losses: one old-task objective and one new-task objective.

## Training-loop pattern

```python
for task_id, task_loader in enumerate(task_loaders):
    model.train()
    for inputs, targets in task_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        loss_fn = make_loss_fn(inputs, targets)
        optimizer.set_closure(loss_fn)
        logits, loss_list = optimizer.step()
```

Do not call `loss.backward()` outside the closure unless the optimizer documentation explicitly instructs otherwise.

## Geometry reshaping optimizers

Geometry reshaping methods are useful when the goal is to encourage flatter or more stable local loss geometry across continual-learning tasks.

### C-Flat

`C_Flat` is the main flatness-aware optimizer exposed by INFTY.

```python
optimizer = infty_optim.C_Flat(
    params=model.parameters(),
    base_optimizer=base_optimizer,
    model=model,
    args={
        "strategy": "basic",
        "rho": 0.05,
        "lamb": 0.2,
    },
)
```

Important arguments:

| Argument | Meaning |
| --- | --- |
| `strategy` | `"basic"` or `"plus"`. |
| `rho` | Perturbation radius. |
| `lamb` | Aggregation coefficient. |
| `rho_scheduler` | Optional scheduler for perturbation radius. |
| `adaptive` | Whether to use adaptive perturbations scaled by parameter magnitude. |

### SAM, GSAM, GAM, LookSAM

These classes expose related geometry-aware optimization strategies. They follow the same wrapping logic: create a base optimizer, pass model parameters and the model, set a closure, then call `step()`.

## Zeroth-order updates

`ZeroFlow` supports zeroth-order and forward-gradient update variants. It is intended for settings where ordinary backpropagation is expensive, unavailable, or undesirable.

```python
optimizer = infty_optim.ZeroFlow(
    params=model.parameters(),
    base_optimizer=base_optimizer,
    model=model,
    args={
        "inftyopt": "zo_sgd",
        "q": 1,
        "zo_eps": 1e-3,
        "perturbation_mode": "two_side",
        "memory_efficient": False,
    },
)
```

Supported `inftyopt` values include:

| Value | Description |
| --- | --- |
| `zo_sgd` | Zeroth-order update with SGD-style base optimizer. |
| `zo_adam` | Zeroth-order update with Adam-style base optimizer. |
| `zo_sgd_sign` | Sign version of the zeroth-order update. |
| `zo_adam_sign` | Sign version with Adam-style base optimizer. |
| `zo_sgd_conserve` | Conservative update that rejects harmful moves. |
| `zo_adam_conserve` | Conservative update with Adam-style base optimizer. |
| `forward_grad` | Forward-gradient update path. |

Important arguments:

| Argument | Meaning |
| --- | --- |
| `q` | Number of zeroth-order query directions. |
| `zo_eps` | Finite-difference perturbation size. |
| `perturbation_mode` | `"one_side"` or `"two_side"`. |
| `use_history_grad` | Whether to smooth estimated directions over time. |
| `alpha` | Smoothing coefficient for historical directions. |
| `gradient_sparsity` | Optional float or dictionary controlling random sparsity masks. |
| `memory_efficient` | Recompute noise instead of storing it. |

## Gradient filtering optimizers

Gradient filtering methods are useful when multiple objectives interfere with each other, which is common in replay-based or multi-loss continual learning.

### UniGrad-FS

`UniGrad_FS` handles gradient projection with a flatter-sharpness mechanism. For tasks after the first task, it expects exactly two objective terms.

```python
optimizer = infty_optim.UniGrad_FS(
    params=model.parameters(),
    base_optimizer=base_optimizer,
    model=model,
    args={
        "task_id": task_id,
        "utype": "model-wise",
        "S_T": [0.1],
        "beta": 0.9,
        "rho": 0.05,
    },
)
```

Important arguments:

| Argument | Meaning |
| --- | --- |
| `task_id` | Current task index. Task `0` uses ordinary summed-loss behavior. |
| `utype` | `"model-wise"` or `"layer-wise"` gradient handling. |
| `S_T` | Similarity threshold. A scalar list such as `[0.1]` is accepted for model-wise mode. |
| `beta` | Threshold update coefficient. |
| `rho` | Perturbation radius used by the flatter-sharpness component. |

### PCGrad, CAGrad, GradVac, OGD

These optimizers follow the same multi-objective wrapper style. They are useful baselines or alternatives when analyzing task-gradient interference.

## Visualization utilities

### Loss landscape

```python
from infty import plot as infty_plot

infty_plot.visualize_landscape(
    optimizer=optimizer,
    model=model,
    create_loss_fn=create_loss_fn,
    loader=train_loader,
    task=task_id,
    device=device,
    output_dir="workdirs/plots/diagnostics/landscape/demo",
)
```

### Hessian ESD

```python
infty_plot.visualize_esd(
    optimizer=optimizer,
    model=model,
    create_loss_fn=create_loss_fn,
    loader=train_loader,
    task=task_id,
    device=device,
    output_dir="workdirs/plots/diagnostics/esd/demo",
)
```

### Conflict curves

```python
infty_plot.visualize_conflicts(
    optimizer=optimizer,
    task=task_id,
    output_dir="workdirs/plots/diagnostics/conflicts/demo",
)
```

This API only renders the top-level similarity curve.
The optimizer must expose `sim_list` or `sim_arr`. `UniGrad_FS` records similarity values through `sim_list`.

### Optimization trajectory

```python
infty_plot.visualize_trajectory(
    optimizer_name="adam",
    n_iter=2000,
    output_dir="workdirs/plots/diagnostics/trajectory/demo",
)
```

## Choosing an optimizer

| Situation | Suggested starting point |
| --- | --- |
| Need a flatness-aware baseline | `C_Flat` or `SAM` |
| Need continual-learning flatness behavior | `C_Flat` |
| Need backprop-free or gradient-free updates | `ZeroFlow` |
| Need old/new task objective balancing | `UniGrad_FS` |
| Need multi-objective baselines | `PCGrad`, `CAGrad`, `GradVac`, `OGD` |
| Need diagnostics only | `infty.plot` |

## Reproducibility recommendations

For every experiment, record:

- INFTY version or commit hash;
- PyTorch version;
- optimizer class and `args` dictionary;
- base optimizer and its hyperparameters;
- random seed;
- task order and dataset split;
- output directory for plot artifacts.
