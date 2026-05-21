# INFTY Documentation

INFTY is an optimization-centric toolkit for Continual AI. It provides plug-and-play optimizers and visualization tools for diagnosing optimization behavior in continual-learning systems.

The library is designed for research workflows where users need to replace or augment an existing PyTorch optimizer without rewriting the whole training pipeline.

## Documentation map

### Get started

| Page | Purpose |
| --- | --- |
| [Installation](installation.md) | Install INFTY from PyPI or from source. |
| [Quick Start](quickstart.md) | Learn the minimum optimizer-wrapping pattern and a small inline example. |
| [Examples](examples.md) | Find runnable minimal scripts, visualization examples, and formal PILOT launcher scripts. |

### Use INFTY

| Page | Purpose |
| --- | --- |
| [User Guide](user_guide.md) | Understand optimizer families, closures, continual-learning workflows, and visualization utilities. |
| [API Reference](api_reference.md) | Look up public classes, functions, arguments, return values, and expected closure contracts. |
| [Troubleshooting](troubleshooting.md) | Resolve common installation, closure, training, and plotting issues. |

### Project & Development

| Page | Purpose |
| --- | --- |
| [Version Policy](stability_policy.md) | Understand INFTY's public beta status, compatibility promises, and supported Python versions. |
| [Developer Guide](developer_guide.md) | Extend the library with new optimizers, plots, examples, and documentation. |
| [Changelog](changelog.md) | Track user-facing documentation, API, and packaging changes across releases. |
| [Citation](citation.md) | Cite INFTY and related algorithm papers. |

## Main concepts

INFTY follows four design ideas:

1. **Optimizer wrapping**: most INFTY optimizers wrap a standard PyTorch optimizer such as `torch.optim.SGD` or `torch.optim.Adam`.
2. **Closure-based training**: the training loss is supplied through a closure that returns `(logits, loss_list)`.
3. **Continual-learning awareness**: several optimizers explicitly handle flatness, gradient conflicts, or zeroth-order updates, which are common concerns in continual learning.
4. **Diagnostics as first-class utilities**: the package includes tools for loss landscapes, Hessian ESD, gradient-conflict curves, and trajectory visualization.

## Recommended reading order

New users should read:

1. [Installation](installation.md)
2. [Quick Start](quickstart.md)
3. [Examples](examples.md)
4. [User Guide](user_guide.md)

Users integrating INFTY into an existing training framework should additionally read:

1. [API Reference](api_reference.md)
2. [Troubleshooting](troubleshooting.md)
3. [Version Policy](stability_policy.md)

Contributors should read:

1. [Developer Guide](developer_guide.md)
2. [Changelog](changelog.md)
3. [Citation](citation.md)

## Current status

INFTY is in public beta. The documented `infty.optim` and `infty.plot` APIs are intended for stable incremental updates, while experiment integrations and benchmark scripts may evolve faster. For long-lived use, pin the package version or commit hash and review the [version policy](stability_policy.md) before upgrading.
