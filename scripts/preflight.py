#!/usr/bin/env python3
"""Report which KARE reproduction stages are ready on this machine."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kare.config import get_paths


CORE_MODULES = ("numpy", "pandas", "sklearn", "scipy", "networkx", "tqdm")
PIPELINE_MODULES = ("torch", "pyhealth", "faiss", "graspologic", "openai", "boto3")
FINETUNE_MODULES = ("transformers", "datasets", "trl", "accelerate", "peft", "bitsandbytes")
DYPHRAG_MODULES = ("tensorboard", "mlflow", "pytest")


def available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def gpu_info() -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.SubprocessError):
        return []


def build_report() -> dict:
    paths = get_paths()
    examples = {
        "kg_raw": PROJECT_ROOT / "graph/kg_raw.txt",
        "community_summaries": PROJECT_ROOT / "data_examples/community_summary_examples/example_community_summary.json",
        "mortality_retrieval": PROJECT_ROOT / "data_examples/retrieved_knowledge_examples/retrieved_knowledge_example_mortality.json",
        "readmission_retrieval": PROJECT_ROOT / "data_examples/retrieved_knowledge_examples/retrieved_knowledge_example_readmission.json",
    }
    modules = {
        name: available(name)
        for name in (*CORE_MODULES, *PIPELINE_MODULES, *FINETUNE_MODULES, *DYPHRAG_MODULES)
    }
    mimic = {
        "mimic3": bool(paths.mimic3_root and paths.mimic3_root.is_dir()),
        "mimic4": bool(paths.mimic4_root and paths.mimic4_root.is_dir()),
    }
    provider = {
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "bedrock": all(os.environ.get(name) for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION")),
    }
    gpus = gpu_info()
    torch_cuda = False
    if modules["torch"]:
        try:
            import torch

            torch_cuda = bool(torch.version.cuda and torch.cuda.is_available())
        except (ImportError, RuntimeError):
            pass
    return {
        "python": platform.python_version(),
        "recommended_python": "3.11",
        "paths": {
            "project": str(paths.project_root),
            "data": str(paths.data_root),
            "mimic3": str(paths.mimic3_root) if paths.mimic3_root else None,
            "mimic4": str(paths.mimic4_root) if paths.mimic4_root else None,
        },
        "examples_ready": all(path.is_file() and path.stat().st_size > 0 for path in examples.values()),
        "mimic": mimic,
        "api": provider,
        "gpu": gpus,
        "torch_cuda": torch_cuda,
        "modules": modules,
        "stages": {
            "included_examples": all(path.is_file() for path in examples.values()) and all(modules[name] for name in CORE_MODULES),
            "ehr_preprocessing": any(mimic.values()) and modules["pyhealth"] and modules["torch"],
            "kg_and_retrieval": (provider["openai"] or provider["bedrock"]) and modules["graspologic"],
            "local_finetuning": bool(gpus) and torch_cuda and all(modules[name] for name in FINETUNE_MODULES),
            "dyphrag_cpu_smoke": modules["torch"] and all(modules[name] for name in DYPHRAG_MODULES),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Fail unless EHR preprocessing is ready")
    args = parser.parse_args(argv)
    report = build_report()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Python {report['python']} (recommended: {report['recommended_python']})")
        print("GPU: " + ("; ".join(report["gpu"]) if report["gpu"] else "not available"))
        print(f"PyTorch CUDA: {'available' if report['torch_cuda'] else 'not available'}")
        for stage, ready in report["stages"].items():
            print(f"[{'READY' if ready else 'BLOCKED'}] {stage}")
        if not any(report["mimic"].values()):
            print("Set MIMIC3_ROOT or MIMIC4_ROOT to enable EHR preprocessing.")
        if not (report["api"]["openai"] or report["api"]["bedrock"]):
            print("Set OPENAI_API_KEY or AWS credentials only for LLM-dependent stages.")
    if args.strict and not report["stages"]["ehr_preprocessing"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
