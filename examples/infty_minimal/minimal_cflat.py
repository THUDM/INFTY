"""Minimal C-Flat example for INFTY.

Run from repository root:
    python examples/infty_minimal/minimal_cflat.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from infty.optim import C_Flat


def main():
    torch.manual_seed(0)
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
    for epoch in range(2):
        total = 0.0
        for inputs, targets in loader:
            optimizer.set_closure(make_loss_fn(inputs, targets))
            _, loss_list = optimizer.step()
            total += float(sum(loss_list).detach())
        print(f"epoch={epoch} loss={total / len(loader):.4f}")


if __name__ == "__main__":
    main()
