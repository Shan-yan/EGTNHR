#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${KARE_ENV_NAME:-kare}"
TORCH_BACKEND="${KARE_TORCH_BACKEND:-cpu}"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  :
else
  conda env create -n "${ENV_NAME}" -f "${PROJECT_ROOT}/environment.yml"
fi
conda run -n "${ENV_NAME}" python -m pip install \
  torch==2.10.0 \
  --index-url "https://download.pytorch.org/whl/${TORCH_BACKEND}"
conda run -n "${ENV_NAME}" python -m pip install -r "${PROJECT_ROOT}/requirements.txt"

if [[ "${KARE_INSTALL_FINETUNE:-0}" == "1" ]]; then
  conda run -n "${ENV_NAME}" python -m pip install -r "${PROJECT_ROOT}/requirements-finetune.txt"
fi

if [[ "${KARE_INSTALL_DYPHRAG:-0}" == "1" ]]; then
  conda run -n "${ENV_NAME}" python -m pip install -r "${PROJECT_ROOT}/requirements-dyphrag.txt"
fi

if [[ "${KARE_INSTALL_WANDB:-0}" == "1" ]]; then
  conda run -n "${ENV_NAME}" python -m pip install -r "${PROJECT_ROOT}/requirements-wandb.txt"
fi

conda run -n "${ENV_NAME}" python "${PROJECT_ROOT}/scripts/preflight.py"
