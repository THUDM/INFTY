# Quick Start

This page shows the minimum patterns needed to use INFTY optimizers in a PyTorch training loop.

## Core pattern

Most INFTY optimizers use the same pattern:

1. wrap a normal PyTorch optimizer with an INFTY optimizer;
2. write a closure that returns `(logits, loss_list)`;
3. call `optimizer.set_closure(loss_fn)`;
4. call `optimizer.step()`.

The closure contract is important:

```python
logits, loss_list = loss_fn()
```

where `loss_list` must be a list of scalar tensors.

## Minimal C-Flat example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from infty.optim import C_Flat

x = torch.randn(64, 16)
y = torch.randint(0, 4, (64,))
loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)

model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

optimizer = C_Flat(
    params=model.parameters(),
    base_optimizer=base_optimizer,
    model=model,
    args={"rho": 0.05, "lamb": 0.2, "strategy": "basic"},
)

def make_loss_fn(inputs, targets):
    def loss_fn():
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets)
        return logits, [loss]
    return loss_fn

model.train()
for inputs, targets in loader:
    optimizer.set_closure(make_loss_fn(inputs, targets))
    logits, loss_list = optimizer.step()
    print(float(sum(loss_list)))
```

## More runnable examples

For runnable scripts rather than inline snippets, see [Examples](examples.md).

That page covers:

- minimal standalone scripts under `examples/infty_minimal/` for C-Flat, ZeroFlow, UniGrad-FS, and trajectory visualization;
- the retained formal PILOT launcher scripts under `workdirs/scripts/`.

## Quick visualization example

```python
from infty.plot import MINIMAL_TRAJECTORY_DIR, visualize_trajectory

visualize_trajectory(
    optimizer_name="adam",
    n_iter=2000,
    lr=0.1,
    output_dir=MINIMAL_TRAJECTORY_DIR,
    grid_size=120,
)
```
