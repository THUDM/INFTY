import torch
from torch.utils.data import DataLoader, TensorDataset

from infty.utils.hessian import unpack_batch, hessian


def test_unpack_batch_accepts_pair_and_triplet():
    inputs = torch.randn(2, 3)
    targets = torch.randn(2, 1)
    assert unpack_batch((inputs, targets))[0] is inputs
    assert unpack_batch((0, inputs, targets))[1] is targets


def test_unpack_batch_accepts_dict():
    inputs = torch.randn(2, 3)
    targets = torch.randn(2, 1)
    got_i, got_t = unpack_batch({"inputs": inputs, "targets": targets})
    assert got_i is inputs and got_t is targets


def test_dataloader_hv_product_accepts_pair_batches():
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 1)
    x = torch.randn(8, 3)
    y = torch.randn(8, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    def create_loss_fn(inputs, targets):
        def loss_fn():
            pred = model(inputs)
            loss = torch.nn.functional.mse_loss(pred, targets)
            return pred, [loss]

        return loss_fn

    h = hessian(model, create_loss_fn, dataloader=loader, device="cpu")
    v = [torch.randn_like(p) for p in h.params]
    eigenvalue, THv = h.dataloader_hv_product(v)
    assert isinstance(eigenvalue, float)
    assert len(THv) == len(h.params)


def test_dataloader_hv_product_accepts_triplet_batches():
    torch.manual_seed(1)
    model = torch.nn.Linear(3, 1)
    x = torch.randn(6, 3)
    y = torch.randn(6, 1)

    class TripletLoader:
        def __iter__(self):
            yield 0, x[:3], y[:3]
            yield 1, x[3:], y[3:]

    def create_loss_fn(inputs, targets):
        def loss_fn():
            pred = model(inputs)
            loss = torch.nn.functional.mse_loss(pred, targets)
            return pred, [loss]

        return loss_fn

    h = hessian(model, create_loss_fn, dataloader=TripletLoader(), device="cpu")
    v = [torch.randn_like(p) for p in h.params]
    eigenvalue, THv = h.dataloader_hv_product(v)
    assert isinstance(eigenvalue, float)
    assert len(THv) == len(h.params)
