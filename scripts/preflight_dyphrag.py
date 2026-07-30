#!/usr/bin/env python3
"""Fail-fast server preflight for the complete DyPH-RAG matrix."""

from __future__ import annotations

import importlib
import json
import os
import shlex
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dyphrag.config import CONFIG_ROOT, load_config
from src.dyphrag.retrieval import load_medical_corpus
from src.orchestrate import _experiments


def fail(message: str) -> None:
    raise RuntimeError(message)


def require_import(module: str) -> None:
    try:
        importlib.import_module(module)
    except ImportError as error:
        fail(f"missing Python dependency {module!r}: {error}")


def require_jsonl(path: Path, description: str) -> None:
    if not path.is_file():
        fail(f"{description} does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        first = next((line for line in handle if line.strip()), "")
    if not first:
        fail(f"{description} is empty: {path}")
    try:
        value = json.loads(first)
    except json.JSONDecodeError as error:
        fail(f"{description} is not valid JSONL: {error}")
    if not isinstance(value, dict):
        fail(f"{description} JSONL records must be objects")


def main() -> int:
    for module in ("torch", "numpy", "sklearn", "scipy", "yaml", "tqdm", "rich", "psutil"):
        require_import(module)
    groups = os.environ.get("EXPERIMENT_GROUPS", "all")
    experiments = _experiments(groups)
    if not experiments:
        fail("the selected experiment matrix is empty")
    existing = {path.stem for path in (CONFIG_ROOT / "experiment").glob("*.yaml")}
    missing = sorted(set(experiments) - existing)
    if missing:
        fail(f"experiment configs are missing: {missing}")
    dataset = os.environ.get("DYPHRAG_DATASET", "synthetic")
    if dataset not in {"synthetic", "mimiciii", "mimiciv"}:
        fail(f"unsupported DYPHRAG_DATASET={dataset!r}")
    overrides = shlex.split(os.environ.get("DYPHRAG_MATRIX_OVERRIDES", ""))
    configs = [
        load_config([f"experiment={name}", f"dataset={dataset}", *overrides])
        for name in experiments
    ]
    if any(bool(config["tracking"]["enabled"]) for config in configs):
        require_import("torch.utils.tensorboard")
        require_import("mlflow")
    if any(bool(config["tracking"]["wandb"]) for config in configs):
        require_import("wandb")
        if not os.environ.get("WANDB_API_KEY") and os.environ.get("DYPHRAG_WANDB_MODE") != "offline":
            fail("WANDB_API_KEY is required when online W&B tracking is enabled")
    if dataset != "synthetic":
        data_variable = (
            "DYPHRAG_MIMIC3_JSONL"
            if dataset == "mimiciii"
            else "DYPHRAG_MIMIC4_JSONL"
        )
        data_path = os.environ.get(data_variable, "")
        if not data_path:
            fail(f"{data_variable} must point to an authorized prepared dataset")
        require_jsonl(Path(data_path).expanduser(), data_variable)
        needs_medical = any(bool(config["retrieval"]["medical_enabled"]) for config in configs)
        if needs_medical:
            corpus = os.environ.get("DYPHRAG_MEDICAL_CORPUS", "")
            if not corpus:
                fail("DYPHRAG_MEDICAL_CORPUS is required by the selected real-data experiments")
            corpus_path = Path(corpus).expanduser()
            require_jsonl(corpus_path, "DYPHRAG_MEDICAL_CORPUS")
            load_medical_corpus(corpus_path)
    matrix_dir = Path(os.environ.get("DYPHRAG_MATRIX_DIR", "results/matrix")).expanduser()
    matrix_dir.mkdir(parents=True, exist_ok=True)
    probe = matrix_dir / ".dyphrag_write_probe"
    probe.touch()
    probe.unlink()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    torch = importlib.import_module("torch")
    if visible not in {"", "-1"} and not torch.cuda.is_available():
        fail("CUDA_VISIBLE_DEVICES requests GPUs, but this PyTorch build cannot access CUDA")
    print(
        f"DyPH-RAG preflight passed: dataset={dataset}, "
        f"experiments={len(experiments)}, matrix_dir={matrix_dir}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"DyPH-RAG preflight failed: {error}", file=sys.stderr)
        raise SystemExit(2)
