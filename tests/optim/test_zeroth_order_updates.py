import torch


def test_zeroflow_step_updates_parameters_without_backward():
    from infty.optim import ZeroFlow

    torch.manual_seed(0)
    model = torch.nn.Linear(3, 1)
    base_optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer = ZeroFlow(
        model.parameters(),
        base_optimizer=base_optimizer,
        model=model,
        args={
            "q": 1,
            "zo_eps": 1e-3,
            "perturbation_mode": "two_side",
            "inftyopt": "zo_sgd",
        },
    )

    x = torch.randn(5, 3)
    y = torch.randn(5, 1)

    def closure():
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        return pred, [loss]

    optimizer.set_closure(closure)
    before = [p.detach().clone() for p in model.parameters()]
    optimizer.step()
    after = [p.detach().clone() for p in model.parameters()]

    assert any(not torch.equal(a, b) for a, b in zip(before, after))
