#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

python -m unittest discover -s tests -v
python -m src.train \
  experiment=full_dyphrag \
  dataset=synthetic \
  seed=42 \
  data.train_size=8 \
  data.validation_size=4 \
  data.test_size=4 \
  training.epochs=1 \
  evaluation.bootstrap_samples=4 \
  runtime.device=cpu \
  runtime.progress=false \
  tracking.enabled="${DYPHRAG_TRACKING_ENABLED:-true}" \
  output_dir="${DYPHRAG_SMOKE_OUTPUT:-results/smoke/full_final_seed42}"
