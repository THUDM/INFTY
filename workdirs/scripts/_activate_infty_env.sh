#!/usr/bin/env bash

if [[ "${INFTY_SKIP_ENV_ACTIVATE:-0}" == "1" ]]; then
  return 0
fi

INFTY_ENV_NAME="${INFTY_ENV_NAME:-infty}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "${INFTY_ENV_NAME}" ]]; then
  return 0
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[launcher] conda was not found in PATH." >&2
  echo "[launcher] Activate '${INFTY_ENV_NAME}' manually or set INFTY_SKIP_ENV_ACTIVATE=1." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "${INFTY_ENV_NAME}"
