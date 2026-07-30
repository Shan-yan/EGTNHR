#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG_SH="${DYPHRAG_CONFIG_SH:-${PROJECT_ROOT}/scripts/dyphrag_env.sh}"
if [[ ! -f "${CONFIG_SH}" ]]; then
  echo "DyPH-RAG config file not found: ${CONFIG_SH}" >&2
  exit 2
fi
# shellcheck source=dyphrag_env.sh
source "${CONFIG_SH}"

export MAX_PARALLEL SEEDS EXPERIMENT_GROUPS RETRY_COUNT FAIL_FAST
if [[ "${DYPHRAG_SKIP_PREFLIGHT:-0}" != "1" ]]; then
  python scripts/preflight_dyphrag.py
fi
echo "DyPH-RAG matrix: dataset=${DYPHRAG_DATASET} groups=${EXPERIMENT_GROUPS} seeds=${SEEDS} parallel=${MAX_PARALLEL}"
echo "Live state: ${DYPHRAG_MATRIX_DIR}/state.json; per-run logs: ${DYPHRAG_MATRIX_DIR}/runs/*/{stdout,stderr}.log"
python -m src.orchestrate
