# Troubleshooting

This page collects common issues when installing INFTY, writing closures, running optimizers, and generating plots.

## `ModuleNotFoundError: No module named 'infty'`

The package is not installed in the active Python environment.

Check the active Python executable:

```bash
which python
python -m pip show infty
```

Install or reinstall:

```bash
python -m pip install -e .
```

## `ModuleNotFoundError` for PILOT/demo dependencies

The PILOT demo and some optional workflows require the `examples` extra. The minimal scripts under `examples/infty_minimal/` do not. From a source checkout, run this from the repository root:

```bash
python -m pip install ".[examples]"
```

## Optimizer does not update parameters

Check that:

- model parameters have `requires_grad=True`;
- the base optimizer was initialized with the same parameters;
- the closure returns scalar losses;
- `optimizer.set_closure(loss_fn)` is called before `optimizer.step()`;
- you are not accidentally calling `base_optimizer.zero_grad()` after `optimizer.step()` in a way that hides the update.

## Closure returns the wrong format

INFTY expects:

```python
return logits, [loss]
```

not:

```python
return loss
```

and not:

```python
return logits, loss
```

For multi-objective optimizers, use:

```python
return logits, [loss_1, loss_2]
```

## `UniGrad_FS only supports two losses`

For `task_id > 0`, `UniGrad_FS` requires exactly two losses:

```python
return logits, [old_loss, new_loss]
```

If you have more than two loss terms, combine them before returning:

```python
old_loss = replay_loss + regularization_loss
new_loss = supervised_loss
return logits, [old_loss, new_loss]
```

## C-Flat or SAM-style optimizer is slow

Geometry-aware optimizers may call the closure multiple times per step. This is expected because they evaluate perturbed model states.

Mitigations:

- reduce batch size;
- use mixed precision where safe;
- use smaller models for debugging;
- start with fewer epochs;
- profile closure runtime.

## CUDA out of memory

Common causes:

- large model and large batch;
- optimizer calls closure multiple times;
- visualization computes Hessian-related quantities;
- graph retained by multi-objective gradient computation.

Mitigations:

```python
base_optimizer.zero_grad(set_to_none=True)
```

and reduce:

- batch size;
- visualization `samples`;
- trajectory `grid_size`;
- number of zeroth-order queries `q`.

## BatchNorm behaves unexpectedly

Some geometry-aware optimizers temporarily disable or enable running statistics during perturbation passes. If results look unstable:

- verify model train/eval mode;
- use sufficiently large batches for BatchNorm;
- consider freezing BatchNorm statistics in continual-learning stages;
- compare with a base optimizer baseline.

## Distributed training issue with gradient reduction

`InftyBaseOptimizer` supports distributed gradient synchronization through `grad_reduce`.

Valid values:

```python
"mean"
"sum"
```

If using DistributedDataParallel, ensure the model object passed into the optimizer is the DDP-wrapped model when `no_sync()` behavior is required.

## ZeroFlow produces unstable loss

Try:

- smaller learning rate;
- smaller or larger `zo_eps` depending on scale;
- `perturbation_mode="two_side"`;
- `q > 1` for lower-variance estimates;
- `use_history_grad=True`;
- `zo_sgd_conserve` or `zo_adam_conserve`.

## `forward_grad` path fails

The `forward_grad` variant uses a functional-call and JVP path. It is more sensitive to the exact objective signature. Start with the standard `zo_sgd` mode, then migrate to `forward_grad` after confirming the closure and batch structure.

## Loss landscape or ESD visualization is slow

These utilities compute Hessian-related quantities and can be expensive.

For debugging:

```python
visualize_landscape(..., samples=11, limit=0.3)
visualize_trajectory(..., n_iter=2000, grid_size=100)
```

## Plot output is not found

All plotting functions accept `output_dir`. Pass an explicit directory:

```python
output_dir="workdirs/plots/custom/my_experiment"
```

Most plotting functions return a dictionary or path containing the saved artifact location. Print the return value.

## Matplotlib fails on a headless server

Use a non-interactive backend before importing plotting code:

```python
import matplotlib
matplotlib.use("Agg")
```
