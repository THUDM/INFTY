# Installation

This page explains how to install INFTY for normal use, example execution, and local development.

## Requirements

INFTY is a Python package built around PyTorch. The package metadata declares the following core dependencies:

- `torch`
- `torchvision`
- `numpy`
- `matplotlib`
- `scipy`
- `seaborn`
- `tqdm`

INFTY currently requires Python `>=3.8`. The tested CI support matrix is Python 3.8, 3.9, 3.10, and 3.11.

## Install from PyPI

```bash
python -m pip install --upgrade pip
python -m pip install infty
```

Verify the installation:

```bash
python -c "from infty.optim import C_Flat, ZeroFlow, UniGrad_FS; print('INFTY import succeeded')"
```

## Install from source

```bash
git clone https://github.com/WanNaa/INFTY_demo.git
cd INFTY_demo
python -m pip install --upgrade pip
python -m pip install .
```

For editable development:

```bash
python -m pip install -e .
```

## Install PILOT/demo dependencies

The `examples` extra primarily supports the PILOT demo and related optional workflows. It includes packages such as `easydict`, `Pillow`, `PyYAML`, `scikit-learn`, `timm`, and `umap-learn`.

The minimal scripts under `examples/infty_minimal/` use only the core INFTY installation.

From a source checkout, run this from the repository root:

```bash
python -m pip install ".[examples]"
```

If you installed INFTY from PyPI instead of a source checkout:

```bash
python -m pip install "infty[examples]"
```

For editable development with examples from a source checkout:

```bash
python -m pip install -e ".[examples]"
```

## Recommended conda environment

```bash
conda create -n infty python=3.8
conda activate infty
python -m pip install --upgrade pip
python -m pip install infty
```

If you need a specific CUDA build of PyTorch, install PyTorch first using the command matching your CUDA/runtime setup, then install INFTY.

## Common checks

Check that `infty` is imported from the expected location:

```bash
python -c "import infty; print(infty.__file__)"
```

Check optimizer exports:

```bash
python -c "from infty.optim import C_Flat, ZeroFlow, UniGrad_FS; print(C_Flat, ZeroFlow, UniGrad_FS)"
```

Check plotting exports:

```bash
python -c "from infty.plot import visualize_landscape, visualize_esd, visualize_conflicts, visualize_trajectory; print('plot utilities imported')"
```

## Troubleshooting

If installation fails, see [Troubleshooting](troubleshooting.md). The most common causes are incompatible PyTorch/CUDA versions, installing into the wrong Python environment, or missing PILOT/demo dependencies from the `examples` extra.
