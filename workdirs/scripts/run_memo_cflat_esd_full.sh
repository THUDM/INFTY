#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIRS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${WORKDIRS_ROOT}/.." && pwd)"
PILOT_DIR="${WORKDIRS_ROOT}/PILOT"

GPU_ID="${1:-1}"
ESD_TASKS="${2:-last}"
RUN_NAME="${3:-memo_cflat_esd_full_$(date -u +%Y%m%d_%H%M%S)}"
RESULTS_DIR="${WORKDIRS_ROOT}/results/${RUN_NAME}"

mkdir -p "${RESULTS_DIR}"

source "${SCRIPT_DIR}/_activate_infty_env.sh"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PILOT_DIR}"

exec > >(tee -a "${RESULTS_DIR}/launcher.log") 2>&1

echo "[launcher] start_time_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[launcher] gpu=${GPU_ID}"
echo "[launcher] esd_tasks=${ESD_TASKS}"
echo "[launcher] results_dir=${RESULTS_DIR}"

python main.py \
  --inftyopt c_flat \
  --config ./exps/memo_10split.json \
  --workdir "${RESULTS_DIR}" \
  --postplot esd \
  --esd_tasks "${ESD_TASKS}"
