import torch


def test_sam_step_updates_parameters():
    from infty.optim import SAM

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 2),
    )
    base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer = SAM(model.parameters(), base_optimizer=base_optimizer, model=model, args={"rho": 0.05})

    x = torch.randn(6, 4)
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    criterion = torch.nn.CrossEntropyLoss()

    def closure():
        logits = model(x)
        loss = criterion(logits, y)
        return logits, [loss]

    optimizer.set_closure(closure)
    before = [p.detach().clone() for p in model.parameters()]
    optimizer.step()
    after = [p.detach().clone() for p in model.parameters()]

    assert any(not torch.equal(a, b) for a, b in zip(before, after))
