"""Minimal UniGrad-FS example for INFTY.

Run from repository root:
    python examples/infty_minimal/minimal_unigrad_fs.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from infty.optim import UniGrad_FS


def main():
    torch.manual_seed(0)
    x = torch.randn(64, 10)
    new_y = torch.randint(0, 5, (64,))
    old_y = torch.randint(0, 5, (64,))
    loader = DataLoader(TensorDataset(x, old_y, new_y), batch_size=16, shuffle=True)

    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer = UniGrad_FS(
        params=model.parameters(),
        base_optimizer=base_optimizer,
        model=model,
        args={
            "task_id": 1,
            "utype": "model-wise",
            "S_T": [0.1],
            "beta": 0.9,
            "rho": 0.05,
        },
    )

    model.train()
    for epoch in range(2):
        total = 0.0
        for inputs, old_targets, new_targets in loader:
            def loss_fn(inputs=inputs, old_targets=old_targets, new_targets=new_targets):
                logits = model(inputs)
                old_loss = F.cross_entropy(logits, old_targets)
                new_loss = F.cross_entropy(logits, new_targets)
                return logits, [old_loss, new_loss]

            optimizer.set_closure(loss_fn)
            _, loss_list = optimizer.step()
            total += float(sum(loss_list).detach())
        print(f"epoch={epoch} combined_loss={total / len(loader):.4f}")


if __name__ == "__main__":
    main()
