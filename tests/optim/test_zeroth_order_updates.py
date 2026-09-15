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


def test_zeroflow_sparse_grad_rng_matches_parameter_device():
    from infty.optim import ZeroFlow

    torch.manual_seed(0)
    model = torch.nn.Linear(2, 1)
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
            "gradient_sparsity": 0.5,
        },
    )

    param_device = next(model.parameters()).device
    expected = "cuda" if param_device.type == "cuda" else "cpu"
    assert str(optimizer.sparse_grad_rng.device) == expected

    # Sparse masking must be usable with the constructed generator device.
    x = torch.randn(4, 2)
    y = torch.randn(4, 1)

    def closure():
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        return pred, [loss]

    optimizer.set_closure(closure)
    optimizer.step()


def test_zeroflow_does_not_hardcode_cuda_when_unavailable(monkeypatch):
    from infty.optim import ZeroFlow

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    model = torch.nn.Linear(2, 1)  # stays on CPU
    base_optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer = ZeroFlow(
        model.parameters(),
        base_optimizer=base_optimizer,
        model=model,
        args={"q": 1, "inftyopt": "zo_sgd"},
    )
    assert str(optimizer.sparse_grad_rng.device) == "cpu"
