#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./bootstrap_doc_site_repo.sh [options] TARGET_DIR

Options:
  --copy-site    Copy the current MkDocs build output from ./site into TARGET_DIR
  -h, --help     Show this help message

Examples:
  ./bootstrap_doc_site_repo.sh ~/work/doc
  ./bootstrap_doc_site_repo.sh --copy-site ~/work/doc
EOF
}

COPY_SITE=false
TARGET_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --copy-site)
      COPY_SITE=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -n "$TARGET_DIR" ]]; then
        echo "Only one TARGET_DIR may be provided." >&2
        usage >&2
        exit 1
      fi
      TARGET_DIR="$1"
      shift
      ;;
  esac
done

if [[ -z "$TARGET_DIR" ]]; then
  usage >&2
  exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

mkdir -p "$TARGET_DIR"
TARGET_DIR="$(cd "$TARGET_DIR" && pwd)"

if [[ "$TARGET_DIR" == "$REPO_ROOT" ]]; then
  echo "TARGET_DIR must not be the current source repository." >&2
  exit 1
fi

cat > "$TARGET_DIR/.nojekyll" <<'EOF'
# GitHub Pages marker file for static site publishing.
EOF

cat > "$TARGET_DIR/index.html" <<'EOF'
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>INFTY Docs</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      :root {
        color-scheme: light;
        --bg: #f4f1e8;
        --panel: #fffdf8;
        --ink: #1f2933;
        --accent: #b45309;
        --line: #e9dcc3;
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        min-height: 100vh;
        display: grid;
        place-items: center;
        padding: 24px;
        font-family: "Georgia", "Times New Roman", serif;
        background:
          radial-gradient(circle at top left, rgba(180, 83, 9, 0.12), transparent 30%),
          linear-gradient(135deg, var(--bg), #f7f7f2);
        color: var(--ink);
      }

      main {
        max-width: 720px;
        padding: 32px;
        border: 1px solid var(--line);
        border-radius: 20px;
        background: var(--panel);
        box-shadow: 0 24px 60px rgba(31, 41, 51, 0.08);
      }

      h1 {
        margin-top: 0;
        font-size: clamp(2rem, 5vw, 3rem);
      }

      p {
        line-height: 1.65;
        font-size: 1.05rem;
      }

      a {
        color: var(--accent);
      }

      code {
        padding: 0.1rem 0.35rem;
        border-radius: 6px;
        background: rgba(31, 41, 51, 0.06);
      }
    </style>
  </head>
  <body>
    <main>
      <h1>INFTY documentation is being prepared.</h1>
      <p>
        This repository is intended to host the published static site for
        <a href="https://github.com/WanNaa/INFTY_demo">WanNaa/INFTY_demo</a>.
      </p>
      <p>
        Once the source repository finishes its documentation workflow, this
        placeholder page will be replaced automatically.
      </p>
      <p>
        If you are setting up this repository for the first time, make sure GitHub
        Pages is enabled for the <code>main</code> branch root and that the source
        repository has a <code>DOC_REPO_TOKEN</code> secret with write access here.
      </p>
    </main>
  </body>
</html>
EOF

if [[ "$COPY_SITE" == true ]]; then
  if [[ ! -d "$REPO_ROOT/site" ]]; then
    echo "No ./site directory found. Run 'mkdocs build --strict' first or omit --copy-site." >&2
    exit 1
  fi

  rsync -av --exclude '.git/' "$REPO_ROOT/site/" "$TARGET_DIR/"
fi

cat <<EOF
Bootstrapped documentation site repository contents in:
  $TARGET_DIR

Next steps:
  1. Create the GitHub repository INFTY-AI/doc if it does not exist yet.
  2. Commit and push the files in $TARGET_DIR to the main branch of that repository.
  3. Enable GitHub Pages for the main branch root in INFTY-AI/doc.
  4. Add a DOC_REPO_TOKEN secret to WanNaa/INFTY_demo with write access to INFTY-AI/doc.
  5. Trigger the Documentation workflow in this source repository.
EOF
