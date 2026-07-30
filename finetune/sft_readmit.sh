#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${KARE_TRAIN_CONFIG:-${PROJECT_ROOT}/finetune/recipes/config_local_smoke.yaml}"
NUM_PROCESSES="${KARE_NUM_PROCESSES:-1}"

cd "${PROJECT_ROOT}"
accelerate launch --num_processes="${NUM_PROCESSES}" \
  finetune/run_sft_readmission.py "${CONFIG}"
