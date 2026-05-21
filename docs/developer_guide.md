# Developer Guide

This guide describes how to extend INFTY with new optimizers, plotting utilities, examples, and documentation.

## Repository layout

A typical INFTY repository layout is:

```text
INFTY_demo/
├── README.md
├── pyproject.toml
├── setup.py
├── src/
│   └── infty/
│       ├── optim/
│       │   ├── geometry_reshaping/
│       │   ├── gradient_filtering/
│       │   └── zeroth_order_updates/
│       ├── plot/
│       └── utils/
├── examples/
│   └── infty_minimal/
├── workdirs/
│   └── PILOT/
├── tests/
│   ├── optim/
│   └── plot/
└── docs/
```

## Development installation

```bash
git clone https://github.com/WanNaa/INFTY_demo.git
cd INFTY_demo
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[examples]"
```

Run tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests
```

Build documentation:

```bash
python -m pip install -r requirements-docs.txt
mkdocs build --strict
```

## Core design principles

### 1. Keep the PyTorch optimizer mental model

INFTY should feel like an extension of PyTorch optimizers. Users should be able to keep their existing parameter groups and base optimizer settings.

### 2. Use closures for optimizer-specific gradient logic

The outer training loop should not need to know how many backward passes or perturbation passes an optimizer uses. That logic belongs inside the optimizer.

### 3. Keep research algorithms modular

Do not place benchmark-specific assumptions inside library code. Benchmark logic belongs in `examples/` or experiment scripts.

### 4. Make diagnostics reproducible

Plotting functions should return output paths and save artifacts under deterministic directories.

## Adding a geometry reshaping optimizer

Use this family when the optimizer modifies the local geometry, perturbation direction, or flatness-aware gradient.

1. Add a new file under:

```text
src/infty/optim/geometry_reshaping/
```

2. Inherit from `InftyBaseOptimizer` when possible.

```python
from .base import InftyBaseOptimizer

class MyOptimizer(InftyBaseOptimizer):
    def __init__(self, params, base_optimizer, model, args, **kwargs):
        super().__init__(params, base_optimizer, model, **kwargs)
        self.name = "my_optimizer"
        self.args = args

    def step(self, closure=None, delay=False):
        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func
        logits, loss_list = get_grad()
        self.base_optimizer.step()
        return logits, loss_list
```

3. Export it from `src/infty/optim/__init__.py`:

```python
from .geometry_reshaping.my_optimizer import MyOptimizer

__all__ = [
    ...,
    "MyOptimizer",
]
```

4. Add tests and documentation.

## Adding a gradient filtering optimizer

Use this family when the optimizer consumes multiple losses and modifies their gradients.

1. Add a file under:

```text
src/infty/optim/gradient_filtering/
```

2. Inherit from `EasyCLMultiObjOptimizer` when the optimizer needs flattened gradient utilities.

```python
from infty.optim.gradient_filtering.base import EasyCLMultiObjOptimizer

class MyMultiObjectiveOptimizer(EasyCLMultiObjOptimizer):
    def __init__(self, params, base_optimizer, model, args, **kwargs):
        super().__init__(params, base_optimizer, model, **kwargs)
        self.name = "my_multi_obj"
        self.args = args

    def step(self, closure=None, delay=False):
        get_grad = closure if closure else self.forward_func
        logits, loss_list = get_grad()
        self._compute_grad_dim()
        grads = self._compute_grad(loss_list, mode="backward")
        new_grads = grads.sum(0)
        self._reset_grad(new_grads)
        if not delay:
            self.base_optimizer.step()
        return logits, loss_list
```

3. Export it from `src/infty/optim/__init__.py`.
4. Add a minimal two-loss test.

## Adding a zeroth-order optimizer

Use this family when the optimizer estimates update directions without ordinary `loss.backward()`.

1. Add a file under:

```text
src/infty/optim/zeroth_order_updates/
```

2. Ensure that parameter perturbations are restored correctly after finite-difference evaluation.
3. Avoid accumulating stale gradients.
4. Document whether the method uses `torch.no_grad()`, `torch.inference_mode()`, or forward-mode AD.

## Adding a plotting utility

1. Add the function under `src/infty/plot/`.
2. Make `output_dir` configurable.
3. Return a dictionary or string containing saved artifact paths.
4. Restore model state if the function perturbs model weights.
5. Export the function in `src/infty/plot/__init__.py`.
6. Add a small test that verifies the function saves an artifact on a tiny model or toy problem.

## Testing guidelines

Recommended minimal tests:

- import tests for public APIs;
- one-step optimizer tests on a tiny neural network;
- closure-shape tests;
- serialization tests for wrapped base optimizers;
- plot regression tests using small grid sizes.

Example optimizer regression test:

```python
def test_cflat_one_step():
    model = torch.nn.Linear(4, 2)
    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 2, (8,))
    base = torch.optim.SGD(model.parameters(), lr=0.01)
    opt = C_Flat(model.parameters(), base, model, args={"rho": 0.01})

    def loss_fn():
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        return logits, [loss]

    opt.set_closure(loss_fn)
    logits, losses = opt.step()
    assert len(losses) == 1
```

## Documentation rules

Every public optimizer should have:

- a short conceptual description in `docs/user_guide.md`;
- constructor arguments in `docs/api_reference.md`;
- at least one runnable example or test;
- notes about closure requirements;
- notes about task-specific assumptions.

## Release checklist

Before releasing a new version:

- [ ] run tests;
- [ ] build MkDocs documentation with `mkdocs build --strict`;
- [ ] build optional Sphinx docs if RST API pages are used;
- [ ] update `docs/changelog.md`;
- [ ] update version metadata;
- [ ] verify PyPI package metadata and homepage URL;
- [ ] verify README documentation links;
- [ ] tag the release;
- [ ] archive experiment-relevant commit hashes for paper reproducibility.

## API stability

The project is currently in public beta. If a documented public constructor, plotting contract, or closure convention changes, document it in the changelog, provide a migration note, and follow the repository's [version policy](stability_policy.md).
