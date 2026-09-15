import torch


def test_looksam_stores_gv_after_periodic_sam_step():
    """LookSAM must persist a vertical component (`gv`) after each k-step SAM refresh.

    Regression: compute_gv looked up state['old_grad'] while sam_perturb_weights
    stores state['old_g'], so gv was never written and non-k steps silently
    skipped apply_gv (LookSAM degraded to plain SGD between refreshes).
    """
    from infty.optim.geometry_reshaping.looksam import LookSAM

    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer = LookSAM(
        model.parameters(),
        base_optimizer=base_optimizer,
        model=model,
        args={"rho": 0.05, "k": 2, "alpha": 0.7},
    )

    x = torch.randn(6, 4)
    y = torch.tensor([0, 1, 0, 1, 0, 1])
    criterion = torch.nn.CrossEntropyLoss()

    def closure():
        logits = model(x)
        loss = criterion(logits, y)
        return logits, [loss]

    optimizer.set_closure(closure)

    optimizer.step()  # step 1: non-k, no gv yet
    optimizer.step()  # step 2: k-multiple -> SAM + compute_gv

    assert any(
        "gv" in optimizer.state.get(p, {}) for p in model.parameters()
    ), "LookSAM.compute_gv did not store 'gv' after a periodic SAM step"


def test_looksam_apply_gv_accepts_float_alpha():
    """apply_gv must accept Python-float alpha (default 0.7), not call alpha.to(p)."""
    from infty.optim.geometry_reshaping.looksam import LookSAM

    torch.manual_seed(1)
    model = torch.nn.Linear(4, 2)
    base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer = LookSAM(
        model.parameters(),
        base_optimizer=base_optimizer,
        model=model,
        args={"rho": 0.05, "k": 2, "alpha": 0.7},
    )
    assert isinstance(optimizer.alpha, float)

    for p in model.parameters():
        p.grad = torch.ones_like(p)
        optimizer.state[p]["gv"] = torch.randn_like(p)

    # Must not raise AttributeError: 'float' object has no attribute 'to'
    optimizer.apply_gv()

    for p in model.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
