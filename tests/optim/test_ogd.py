from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class ToyLogitModel(torch.nn.Module):
    def __init__(self, in_dim=2, out_dim=2):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return {"logits": self.fc(x)}


class ToyDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return index, self.inputs[index], self.targets[index]


def test_ogd_uses_checkpoint_dir_for_basis(tmp_path):
    from infty.optim import OGD

    model = ToyLogitModel(in_dim=4, out_dim=2)
    base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    opt0 = OGD(
        model.parameters(),
        base_optimizer=base_optimizer,
        model=model,
        args={"task_id": 0, "ckp_dir": str(tmp_path)},
    )

    assert Path(opt0.basis_path).exists()
    assert Path(opt0.basis_path).parent == tmp_path.resolve()

    opt1 = OGD(
        model.parameters(),
        base_optimizer=base_optimizer,
        model=model,
        args={"task_id": 1, "ckp_dir": str(tmp_path)},
    )

    assert Path(opt1.basis_path) == Path(opt0.basis_path)


def test_ogd_gtl_basis_uses_ground_truth_logit_gradient(tmp_path):
    from infty.optim import OGD

    torch.manual_seed(0)
    model = ToyLogitModel(in_dim=3, out_dim=2)
    base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer = OGD(
        model.parameters(),
        base_optimizer=base_optimizer,
        model=model,
        args={"task_id": 0, "ckp_dir": str(tmp_path)},
    )

    inputs = torch.tensor([[0.5, -1.0, 2.0]], dtype=torch.float32)
    targets = torch.tensor([1], dtype=torch.long)

    params = [param for param in model.parameters() if param.requires_grad]
    logits = model(inputs)["logits"]
    expected_grads = torch.autograd.grad(logits[0, 1], params, allow_unused=True)
    expected = torch.cat(
        [
            grad.reshape(-1) if grad is not None else torch.zeros_like(param).reshape(-1)
            for param, grad in zip(params, expected_grads)
        ]
    ).to(torch.float32)

    actual = optimizer._compute_gtl_logit_grad_vector(inputs, targets)

    assert torch.allclose(actual, expected)


def test_ogd_filters_exemplars_when_collecting_current_task_memory(tmp_path):
    from infty.optim import OGD

    torch.manual_seed(0)
    model = ToyLogitModel(in_dim=2, out_dim=4)
    base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    OGD(
        model.parameters(),
        base_optimizer=base_optimizer,
        model=model,
        args={"task_id": 0, "ckp_dir": str(tmp_path), "init_cls": 2, "increment": 2},
    )

    optimizer = OGD(
        model.parameters(),
        base_optimizer=base_optimizer,
        model=model,
        args={
            "task_id": 1,
            "ckp_dir": str(tmp_path),
            "init_cls": 2,
            "increment": 2,
            "num_sample_per_task": 10,
        },
    )

    inputs = torch.randn(4, 2)
    targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    loader = DataLoader(ToyDataset(inputs, targets), batch_size=2, shuffle=False)

    sampled_memory = optimizer._sample_current_task_memory(loader, task_count=1)
    sampled_targets = [int(sample[2]) for sample in sampled_memory]

    assert len(sampled_targets) == 2
    assert set(sampled_targets) == {2, 3}


def test_ogd_aligns_saved_basis_when_classifier_expands(tmp_path):
    from infty.optim import OGD

    model_task0 = ToyLogitModel(in_dim=2, out_dim=2)
    base_optimizer0 = torch.optim.SGD(model_task0.parameters(), lr=0.01)
    opt0 = OGD(
        model_task0.parameters(),
        base_optimizer=base_optimizer0,
        model=model_task0,
        args={"task_id": 0, "ckp_dir": str(tmp_path), "init_cls": 2, "increment": 2},
    )

    old_dim = opt0._num_trainable_params()
    opt0.ogd_basis = torch.arange(float(old_dim), dtype=torch.float32).unsqueeze(1)
    opt0._save_basis_state()

    model_task1 = ToyLogitModel(in_dim=2, out_dim=4)
    base_optimizer1 = torch.optim.SGD(model_task1.parameters(), lr=0.01)
    opt1 = OGD(
        model_task1.parameters(),
        base_optimizer=base_optimizer1,
        model=model_task1,
        args={"task_id": 1, "ckp_dir": str(tmp_path), "init_cls": 2, "increment": 2},
    )

    assert opt1.ogd_basis.shape == (opt1._num_trainable_params(), 1)
    assert torch.allclose(opt1.ogd_basis[:4, 0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.allclose(opt1.ogd_basis[8:10, 0], torch.tensor([4.0, 5.0]))
    assert torch.count_nonzero(opt1.ogd_basis[4:8, 0]) == 0
    assert torch.count_nonzero(opt1.ogd_basis[10:, 0]) == 0
