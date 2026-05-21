# Minimal INFTY Examples

These scripts are small CPU-friendly examples intended to demonstrate the INFTY API contract. They are not benchmark reproductions.

Run from the repository root after installing INFTY:

```bash
python examples/infty_minimal/minimal_cflat.py
python examples/infty_minimal/minimal_zeroflow.py
python examples/infty_minimal/minimal_unigrad_fs.py
python examples/infty_minimal/minimal_visualization.py
```

The visualization example writes its artifact under:

```text
workdirs/plots/examples/infty_minimal/trajectory/
```

Each optimizer example uses a toy dataset and a tiny neural network. The key point is the closure format:

```python
return logits, [loss]
```

or, for two-objective continual-learning updates:

```python
return logits, [old_loss, new_loss]
```
