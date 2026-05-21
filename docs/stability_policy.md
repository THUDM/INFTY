# Version Policy

This page defines which parts of INFTY are intended to remain stable across routine updates and which parts may evolve faster.

## Release status

INFTY is currently in **public beta**.

That means:

- the package is intended for real-world experimentation and production-adjacent use;
- public APIs are expected to evolve carefully rather than arbitrarily;
- compatibility promises apply to the documented package surface, not every internal script in the repository.

## Stable public surface

The following interfaces are treated as the public, versioned API surface:

- documented imports from `infty.optim`
- documented imports from `infty.plot`
- documented closure conventions and return-value contracts in the API reference
- documented output-directory and artifact behaviors for plotting utilities

These interfaces should not change incompatibly in a routine patch release.

## Faster-moving repository areas

The following areas are useful and maintained, but are **not** held to the same compatibility bar as the public package API:

- experiment launchers under `workdirs/`
- benchmark-specific integration code under `workdirs/PILOT/`
- local scripts used for reproduction, plotting, or artifact management
- internal module layout that is not part of the documented import surface

These may change when research workflows or benchmark requirements change.

## Version expectations

INFTY follows a practical semantic version policy:

- patch releases are for bug fixes, packaging fixes, documentation fixes, and other changes that should not break documented public APIs;
- minor releases may add new optimizers, new plotting utilities, new arguments, or deprecations, but should preserve existing documented behavior unless explicitly noted;
- major releases are the place for intentional breaking changes to the documented public API.

## Deprecation policy

When a documented public API needs to change:

- add a migration note in the changelog;
- document the replacement API or new usage pattern;
- when feasible, keep a compatibility path for at least one minor release before removal.

If immediate removal is unavoidable, the breaking change must be called out clearly in the release notes and the affected documentation pages.

## Supported Python versions

INFTY currently requires **Python 3.8+**.

The CI-supported version matrix is currently:

- Python 3.8
- Python 3.9
- Python 3.10
- Python 3.11

Later Python versions may work, but they are not treated as part of the tested support matrix until CI coverage is added.

## Recommended usage for deployment

If you plan to use INFTY in a long-lived project:

- pin the package version or commit hash;
- prefer documented imports over internal module paths;
- read the changelog before upgrading;
- treat `workdirs/` utilities as repository tooling, not as stable library API.
