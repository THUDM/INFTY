import copy

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


def test_looksam_state_dict_restores_wrapper_state_and_next_step():
    from infty.optim import LookSAM
    from infty.utils.schedulers import LinearScheduler

    def build(model_state=None):
        model = torch.nn.Linear(2, 1, bias=False)
        if model_state is not None:
            model.load_state_dict(model_state)
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        rho_scheduler = LinearScheduler(
            T_max=8,
            max_value=0.05,
            min_value=0.01,
            init_value=0.02,
            warmup_steps=1,
        )
        optimizer = LookSAM(
            model.parameters(),
            base_optimizer=base_optimizer,
            model=model,
            args={
                "rho": 0.05,
                "k": 2,
                "alpha": torch.tensor(0.7),
                "rho_scheduler": rho_scheduler,
            },
        )
        x = torch.tensor([[1.0, -1.0], [0.5, 2.0]])
        y = torch.tensor([[0.25], [-0.5]])

        def closure():
            prediction = model(x)
            loss = torch.nn.functional.mse_loss(prediction, y)
            return prediction, [loss]

        optimizer.set_closure(closure)
        return model, optimizer

    torch.manual_seed(0)
    model, optimizer = build()
    optimizer.step()
    checkpoint_model = copy.deepcopy(model.state_dict())
    checkpoint_optimizer = copy.deepcopy(optimizer.state_dict())

    optimizer.update_rho_t()
    expected_rho = optimizer.rho
    optimizer.step()
    expected_parameters = [parameter.detach().clone() for parameter in model.parameters()]

    restored_model, restored_optimizer = build(checkpoint_model)
    restored_optimizer.load_state_dict(checkpoint_optimizer)
    restored_optimizer.update_rho_t()
    restored_optimizer.step()

    assert restored_optimizer._step_count == optimizer._step_count
    assert restored_optimizer.rho == expected_rho
    assert restored_optimizer.rho_scheduler.t == optimizer.rho_scheduler.t
    assert restored_optimizer.rho_scheduler.lr() == optimizer.rho_scheduler.lr()
    assert any("old_g" in state for state in restored_optimizer.state.values())
    for expected, actual in zip(expected_parameters, restored_model.parameters()):
        torch.testing.assert_close(actual, expected)


def test_load_state_dict_accepts_legacy_base_optimizer_state():
    from infty.optim import SAM

    model = torch.nn.Linear(2, 1)
    base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = SAM(
        model.parameters(),
        base_optimizer=base_optimizer,
        model=model,
        args={"rho": 0.05},
    )

    optimizer.load_state_dict(base_optimizer.state_dict())

    assert optimizer.param_groups is optimizer.base_optimizer.param_groups


def test_proportion_scheduler_restores_wrapped_scheduler_phase():
    from infty.utils.schedulers import ProportionScheduler

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    scheduler = ProportionScheduler(
        learning_rate_scheduler,
        max_lr=0.1,
        min_lr=0.0,
        max_value=1.0,
        min_value=0.0,
    )
    optimizer.step()
    learning_rate_scheduler.step()
    expected_optimizer_state = copy.deepcopy(optimizer.state_dict())
    expected_state = copy.deepcopy(scheduler.state_dict())
    optimizer.step()
    learning_rate_scheduler.step()
    expected_value = scheduler.step()

    restored_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    restored_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        restored_optimizer, step_size=1, gamma=0.5
    )
    restored = ProportionScheduler(
        restored_lr_scheduler,
        max_lr=0.1,
        min_lr=0.0,
        max_value=1.0,
        min_value=0.0,
    )
    restored_optimizer.load_state_dict(expected_optimizer_state)
    restored.load_state_dict(expected_state)
    restored_optimizer.step()
    restored_lr_scheduler.step()

    assert restored.step() == expected_value
    assert restored.t == scheduler.t
    assert restored.pytorch_lr_scheduler.last_epoch == learning_rate_scheduler.last_epoch
