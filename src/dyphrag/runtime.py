"""Reproducibility metadata and exact random-state capture."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import random
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _git_info(project_root: Path) -> dict[str, Any]:
    try:
        top = Path(
            subprocess.check_output(["git", "-C", str(project_root), "rev-parse", "--show-toplevel"], text=True, stderr=subprocess.DEVNULL).strip()
        ).resolve()
        if top != project_root.resolve():
            return {"commit": "unavailable", "dirty": None, "reason": "project is not an independent git worktree"}
        commit = subprocess.check_output(["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(project_root), "status", "--porcelain"], text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": "unavailable", "dirty": None}


def _gpu_info() -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        return []
    return [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            "cuda_runtime": torch.version.cuda,
        }
        for index in range(torch.cuda.device_count())
    ]


def package_versions() -> dict[str, str]:
    names = ("torch", "numpy", "scikit-learn", "PyYAML", "tensorboard", "mlflow", "psutil")
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def run_metadata(project_root: Path, config: dict[str, Any], split_hash: str, corpus_hash: str) -> dict[str, Any]:
    return {
        "start_timestamp": utc_now(),
        "command_line": shlex.join(sys.argv),
        "seed": int(config["seed"]),
        "dataset": config["dataset"],
        "dataset_version": config["data"]["version"],
        "split_hash": split_hash,
        "retrieval_corpus_hash": corpus_hash,
        "git": _git_info(project_root),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "cpu": platform.processor() or platform.machine(),
            "cpu_count": psutil.cpu_count(logical=True),
            "ram_bytes": psutil.virtual_memory().total,
            "gpu": _gpu_info(),
        },
        "packages": package_versions(),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
