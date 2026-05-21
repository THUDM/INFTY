#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIRS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${WORKDIRS_ROOT}/.." && pwd)"
PILOT_DIR="${WORKDIRS_ROOT}/PILOT"

CONFLICTS_TASKS="${1:-last}"
RUN_NAME="${2:-wa_conflicts_all_parallel_$(date -u +%Y%m%d_%H%M%S)}"
PROBE_BATCHES="${3:-5}"
RESULTS_DIR="${WORKDIRS_ROOT}/results/${RUN_NAME}"

METHODS=(pcgrad gradvac cagrad unigrad_fs ogd)
GPU_IDS=(2 3 4 5 6)

if [[ "${#METHODS[@]}" -ne "${#GPU_IDS[@]}" ]]; then
  echo "Method/GPU list length mismatch" >&2
  exit 1
fi

mkdir -p "${RESULTS_DIR}"

source "${SCRIPT_DIR}/_activate_infty_env.sh"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec > >(tee -a "${RESULTS_DIR}/launcher.log") 2>&1

echo "[launcher] start_time_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[launcher] conflicts_tasks=${CONFLICTS_TASKS}"
echo "[launcher] conflict_probe_batches=${PROBE_BATCHES}"
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
    python main.py \
      --inftyopt "${method}" \
      --config ./exps/wa_10split.json \
      --workdir "${job_dir}" \
      --postplot conflicts \
      --conflicts_tasks "${CONFLICTS_TASKS}" \
      --conflict_probe_batches "${PROBE_BATCHES}"
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
