# Contributing to INFTY

Thanks for contributing to `THUDM/INFTY`.

This document describes the expected workflow for bug reports, feature proposals, code changes, documentation updates, and benchmark-related contributions.

## Before you start

- Read the [README](README.md) for installation, project scope, and examples.
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md) in all project spaces.
- If you need to report a security issue, do not open a public issue. Follow [SECURITY.md](SECURITY.md) instead.

## What to contribute

Contributions are welcome in the following areas:

- bug fixes in the `infty` package
- documentation improvements
- tests and CI reliability improvements
- public API enhancements under `src/infty/`
- benchmark integration and reproducibility fixes under `workdirs/`

If you plan a large API change, workflow change, or benchmark restructuring, open an issue first so the direction can be discussed before implementation.

## Development setup

Create and activate the recommended environment:

```bash
conda create -n infty python=3.8
conda activate infty
```

Clone and install the package in editable mode:

```bash
git clone https://github.com/THUDM/INFTY.git
cd INFTY
python -m pip install -e .
```

Install optional example and benchmark dependencies when needed:

```bash
python -m pip install -e ".[examples]"
```

## Making changes

- Keep changes scoped to a clear purpose.
- Update tests when behavior changes.
- Update docs when user-facing behavior, setup, or APIs change.
- Prefer focused pull requests over large mixed changes.

For source changes, the main package lives under:

- `src/infty/`

Tests live under:

- `tests/`

Documentation sources live under:

- `docs/`
- `sphinx_docs/`
- `mkdocs.yml`
- `theme_overrides/`

Benchmark and launcher assets live under:

- `workdirs/`
- `examples/infty_minimal/`

## Validation

Run the relevant checks before opening a pull request.

Core test suite:

```bash
python -m pytest -q
```

Documentation build:

```bash
python -m mkdocs build --strict
python -m sphinx -b html sphinx_docs build/sphinx/html
```

If your change touches packaging, examples, CI, or docs publishing, validate those areas locally when possible.

## Pull requests

Open pull requests against `main`.

Each PR should include:

- a short explanation of what changed
- why the change is needed
- what validation was run
- any follow-up work or known limitations

Small, reviewable PRs are preferred over broad batches of unrelated changes.

## Documentation publishing

`THUDM/INFTY` is the source repository for docs.
The published site is served from:

- `https://infty-ai.github.io/doc/`

Documentation updates in `THUDM/INFTY` are published by workflow to the external docs repository. Do not commit generated site output for normal documentation changes.

## What not to commit

Do not commit generated or local-only artifacts such as:

- `site/`
- `build/`
- `dist/`
- `__pycache__/`
- local logs, temporary notebooks, or one-off experiment outputs

If you contribute benchmark or experiment-related updates, prefer reproducible configs and scripts over bulky generated outputs.

## Issues and feature requests

Use GitHub issues for:

- reproducible bugs
- feature requests
- documentation problems
- CI regressions

When reporting a bug, include:

- the exact commands or code used
- environment details
- traceback or logs
- expected versus actual behavior

## Questions

If you are unsure whether something belongs in the public package API, the docs site, or benchmark tooling, open an issue first and describe the intended use case.
