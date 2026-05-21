import pytest
import torch


@pytest.mark.parametrize("optimizer_name", ["gradvac", "unigrad_fs"])
def test_conflict_optimizers_expose_sim_list_expand_thresholds_and_record_diagnostics(optimizer_name):
    from infty.optim import GradVac, UniGrad_FS

    optimizer_map = {
        "gradvac": GradVac,
        "unigrad_fs": UniGrad_FS,
    }
    optimizer_cls = optimizer_map[optimizer_name]
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)
    base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer = optimizer_cls(
        model.parameters(),
        base_optimizer=base_optimizer,
        model=model,
        args={"task_id": 1, "utype": "layer-wise", "S_T": [0.1]},
    )

    x = torch.randn(6, 4)
    y1 = torch.randn(6, 1)
    y2 = torch.randn(6, 1)

    def closure():
        pred = model(x)
        loss1 = torch.nn.functional.mse_loss(pred, y1)
        loss2 = torch.nn.functional.mse_loss(pred, y2)
        return pred, [loss1, loss2]

    optimizer.set_closure(closure)
    optimizer.step()

    assert len(optimizer.sim_list) == len(optimizer.k_idx)
    assert optimizer.S_T.numel() == len(optimizer.k_idx)
    assert len(optimizer.conflict_records) == len(optimizer.k_idx)
    assert optimizer.conflict_records[0]["task"] == 1
    assert "cos_before" in optimizer.conflict_records[0]
    assert "cos_after" in optimizer.conflict_records[0]
