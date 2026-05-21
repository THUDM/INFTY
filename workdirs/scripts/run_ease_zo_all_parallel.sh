#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIRS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${WORKDIRS_ROOT}/.." && pwd)"
PILOT_DIR="${WORKDIRS_ROOT}/PILOT"

RUN_NAME="${1:-ease_zo_all_parallel_$(date -u +%Y%m%d_%H%M%S)}"
MAX_TASKS="${2:-0}"
RESULTS_DIR="${WORKDIRS_ROOT}/results/${RUN_NAME}"

METHODS=(
  zo_sgd
  zo_sgd_sign
  zo_sgd_conserve
  zo_adam
  zo_adam_sign
  zo_adam_conserve
  forward_grad
)
GPU_IDS=(1 2 3 4 5 6 7)

if [[ "${#METHODS[@]}" -ne "${#GPU_IDS[@]}" ]]; then
  echo "Method/GPU list length mismatch" >&2
  exit 1
fi

mkdir -p "${RESULTS_DIR}"

source /data/fengtao/miniconda3/bin/activate infty
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec > >(tee -a "${RESULTS_DIR}/launcher.log") 2>&1

echo "[launcher] start_time_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[launcher] max_tasks=${MAX_TASKS}"
echo "[launcher] results_dir=${RESULTS_DIR}"

printf "method\tgpu\tpid\tstdout_log\n" > "${RESULTS_DIR}/pids.tsv"

pids=()
for idx in "${!METHODS[@]}"; do
  method="${METHODS[$idx]}"
  gpu="${GPU_IDS[$idx]}"
  job_dir="${RESULTS_DIR}/${method}"
  stdout_log="${job_dir}/stdout.log"

  mkdir -p "${job_dir}"

  (
    cd "${PILOT_DIR}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    cmd=(
      python main.py
      --inftyopt "${method}"
      --config ./exps/ease.json
      --workdir "${job_dir}"
    )
    if [[ "${MAX_TASKS}" != "0" ]]; then
      cmd+=(--max_tasks "${MAX_TASKS}")
    fi
    "${cmd[@]}"
  ) > "${stdout_log}" 2>&1 &

  pid=$!
  pids+=("${pid}")
  printf "%s\t%s\t%s\t%s\n" "${method}" "${gpu}" "${pid}" "${stdout_log}" >> "${RESULTS_DIR}/pids.tsv"
  echo "[launcher] launched method=${method} gpu=${gpu} pid=${pid}"
done

status=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  method="${METHODS[$idx]}"
  if wait "${pid}"; then
    echo "[launcher] completed method=${method} pid=${pid}"
  else
    echo "[launcher] failed method=${method} pid=${pid}" >&2
    status=1
  fi
done

echo "[launcher] end_time_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "${status}"
