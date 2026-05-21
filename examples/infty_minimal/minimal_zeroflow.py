"""Minimal ZeroFlow example for INFTY.

Run from repository root:
    python examples/infty_minimal/minimal_zeroflow.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from infty.optim import ZeroFlow


def main():
    torch.manual_seed(0)
    x = torch.randn(64, 8)
    y = torch.randint(0, 3, (64,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)

    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
    base_optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer = ZeroFlow(
        params=model.parameters(),
        base_optimizer=base_optimizer,
        model=model,
        args={
            "inftyopt": "zo_sgd",
            "q": 1,
            "zo_eps": 1e-3,
            "perturbation_mode": "two_side",
        },
    )

    for epoch in range(2):
        total = 0.0
        for inputs, targets in loader:
            def loss_fn(inputs=inputs, targets=targets):
                logits = model(inputs)
                loss = F.cross_entropy(logits, targets)
                return logits, [loss]

            optimizer.set_closure(loss_fn)
            _, loss_list = optimizer.step()
            total += float(sum(loss_list).detach())
        print(f"epoch={epoch} loss={total / len(loader):.4f}")


if __name__ == "__main__":
    main()
