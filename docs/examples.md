# Examples

This page lists runnable examples and recommended experiment entry points.

For the minimum usage pattern and the smallest inline optimizer example, start with [Quick Start](quickstart.md).

## PILOT formal runs

The repository retains formal [PILOT](https://github.com/WanNaa/INFTY_demo/tree/main/workdirs/PILOT) launcher scripts under `workdirs/scripts/`. These correspond to the preserved formal artifacts under `workdirs/results/`.

Install the PILOT/demo dependencies from the repository root:

```bash
cd INFTY_demo
python -m pip install ".[examples]"
```

The minimal scripts under `examples/infty_minimal/` do not require this extra.

Run the retained formal C-Flat + ESD job:

```bash
bash workdirs/scripts/run_memo_cflat_esd_full.sh
```

Run the retained formal ZeroFlow comparison suite:

```bash
bash workdirs/scripts/run_ease_zo_all_parallel.sh
```

Run the retained formal conflict-method suite, including UniGrad-FS:

```bash
bash workdirs/scripts/run_wa_conflicts_all_parallel.sh
```

## Minimal examples included in this documentation package

This documentation package also contains small standalone scripts under:

```text
examples/infty_minimal/
```

| File | Purpose |
| --- | --- |
| `minimal_cflat.py` | One-step/small-loop C-Flat example on toy data. |
| `minimal_zeroflow.py` | ZeroFlow example using a finite-difference update. |
| `minimal_unigrad_fs.py` | Two-loss UniGrad-FS example. |
| `minimal_visualization.py` | Toy trajectory visualization example. |

Run them from the repository root after installing INFTY:

```bash
python examples/infty_minimal/minimal_cflat.py
python examples/infty_minimal/minimal_zeroflow.py
python examples/infty_minimal/minimal_unigrad_fs.py
python examples/infty_minimal/minimal_visualization.py
```

## Writing your own example

A good example should include:

1. environment assumptions;
2. dataset or toy-data creation;
3. base optimizer construction;
4. INFTY optimizer construction;
5. closure definition;
6. `set_closure` and `step` calls;
7. expected output or artifact path.

## Example: loss-landscape visualization pattern

```python
from infty import plot as infty_plot

result = infty_plot.visualize_landscape(
    optimizer=optimizer,
    model=model,
    create_loss_fn=create_loss_fn,
    loader=train_loader,
    task=task_id,
    device=device,
    output_dir="workdirs/plots/diagnostics/landscape/demo",
)
print(result["plot_path"])
```

## Example: conflict visualization pattern

```python
from infty import plot as infty_plot

result = infty_plot.visualize_conflicts(
    optimizer=optimizer,
    task=task_id,
    output_dir="workdirs/plots/diagnostics/conflicts/demo",
)
print(result)
```
