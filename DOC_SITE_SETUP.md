# Separate Documentation Site Setup

This repository is configured to build documentation locally and publish the generated static site into a separate GitHub Pages repository: `INFTY-AI/doc`.

## What is already configured here

- `mkdocs.yml` uses `https://INFTY-AI.github.io/doc/` as the public site URL.
- `.github/workflows/docs.yml` builds MkDocs and Sphinx in this source repository.
- The same workflow can push the built `site/` output into `INFTY-AI/doc` when the `DOC_REPO_TOKEN` secret is present.

## One-time GitHub setup

1. Create a new repository named `doc` under the `INFTY-AI` owner.
2. Make one initial commit in that repository.
3. In the `INFTY-AI/doc` repository settings, enable GitHub Pages from:
   - Branch: `main`
   - Folder: `/ (root)`
4. Create a token that has write access to `INFTY-AI/doc`.
5. Add that token as the `DOC_REPO_TOKEN` secret in the source repository `WanNaa/INFTY_demo`.

## Bootstrap the target repository locally

If you have a local clone or empty directory for the target repository, you can initialize it with:

```bash
./bootstrap_doc_site_repo.sh --copy-site /path/to/doc
```

That command:

- creates a `.nojekyll` marker file;
- writes a placeholder `index.html`;
- copies the current contents of `./site/` when `--copy-site` is used.

If you have not built the site yet, run:

```bash
conda run -n infty python -m mkdocs build --strict
```

## First publish flow

1. Push the initialized contents to `INFTY-AI/doc`.
2. Push this source repository changes.
3. Trigger the `Documentation` workflow manually, or let the next docs-related push trigger it.
4. Verify that `https://INFTY-AI.github.io/doc/` loads the generated site.

## Why the extra token is needed

GitHub's default `GITHUB_TOKEN` is scoped to the repository where the workflow runs, so cross-repository publishing generally needs separate credentials with access to the target repository.
