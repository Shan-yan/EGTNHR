#!/usr/bin/env bash
# Single editable configuration file for all DyPH-RAG runs.
# It is safe to source this file; secrets are read from the environment and are
# never printed or copied into run artifacts.
set -euo pipefail

export DYPHRAG_DATASET="${DYPHRAG_DATASET:-synthetic}"   # synthetic|mimiciii|mimiciv
export DYPHRAG_MIMIC3_JSONL="${DYPHRAG_MIMIC3_JSONL:-}"
export DYPHRAG_MIMIC4_JSONL="${DYPHRAG_MIMIC4_JSONL:-}"
export DYPHRAG_MEDICAL_CORPUS="${DYPHRAG_MEDICAL_CORPUS:-}"

export EXPERIMENT_GROUPS="${EXPERIMENT_GROUPS:-all}"     # all|principal|smoke|config_name
export SEEDS="${SEEDS:-42,43,44}"
export MAX_PARALLEL="${MAX_PARALLEL:-1}"
export RETRY_COUNT="${RETRY_COUNT:-1}"
export FAIL_FAST="${FAIL_FAST:-0}"
export DYPHRAG_SKIP_PREFLIGHT="${DYPHRAG_SKIP_PREFLIGHT:-0}"
export DYPHRAG_MATRIX_DIR="${DYPHRAG_MATRIX_DIR:-results/matrix}"
export DYPHRAG_MATRIX_OVERRIDES="${DYPHRAG_MATRIX_OVERRIDES:-task=disease_pair_binary}"
export DYPHRAG_WANDB="${DYPHRAG_WANDB:-0}"
export DYPHRAG_WANDB_MODE="${DYPHRAG_WANDB_MODE:-offline}"
if [[ "${DYPHRAG_WANDB}" == "1" ]]; then
  export DYPHRAG_MATRIX_OVERRIDES="${DYPHRAG_MATRIX_OVERRIDES} tracking.wandb=true tracking.wandb_mode=${DYPHRAG_WANDB_MODE}"
fi

# Leave unset to auto-discover all GPUs, or set e.g. CUDA_VISIBLE_DEVICES=0,1.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

# Optional providers. The core trainer is offline; these are only consumed by
# external corpus-building adapters you choose to add/use.
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  exec bash "${SCRIPT_DIR}/launch_all.sh"
fi
