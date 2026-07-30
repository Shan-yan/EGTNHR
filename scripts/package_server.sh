#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ "${DYPHRAG_PACKAGE_SKIP_TESTS:-0}" != "1" ]]; then
  python -m pytest -q
fi

mkdir -p dist
ARCHIVE="dist/KARE-DyPH-RAG-server.tar.gz"
PACKAGE_ROOT="KARE-DyPH-RAG-server"

INCLUDE=(
  .gitignore
  DYPHRAG.md
  DYPHRAG_CHANGELOG.md
  DYPHRAG_INNOVATIONS.md
  LOCAL_REPRODUCTION.md
  SERVER_DEPLOYMENT.md
  environment.yml
  pyproject.toml
  readme.md
  requirements.txt
  requirements-lock.txt
  requirements-dyphrag.txt
  requirements-finetune.txt
  requirements-wandb.txt
  apis
  apis_example
  baselines
  configs
  data_examples
  ehr_prepare
  finetune
  graph
  kare
  kg_construct
  kg_index
  patient_context
  prediction
  scripts
  src
  tests
)

tar \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache' \
  --exclude='.ruff_cache' \
  --exclude='*.key' \
  --exclude='.env' \
  --exclude='wandb' \
  --transform="s,^,${PACKAGE_ROOT}/," \
  -czf "${ARCHIVE}" \
  "${INCLUDE[@]}"

sha256sum "${ARCHIVE}" > "${ARCHIVE}.sha256"
echo "Created ${ARCHIVE}"
echo "Checksum: $(cut -d ' ' -f 1 "${ARCHIVE}.sha256")"
