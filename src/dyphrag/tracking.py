"""Metrics fan-out to JSONL, summary, TensorBoard and local MLflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class TrackingDependencyError(RuntimeError):
    pass


def _flatten(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if key in {"prepared_jsonl"}:
            continue
        if isinstance(item, dict):
            result.update(_flatten(item, name))
        elif isinstance(item, (str, int, float, bool)) or item is None:
            result[name] = item
    return result


class RunTracker:
    def __init__(self, run_dir: Path, config: dict[str, Any]) -> None:
        self.run_dir = run_dir
        self.config = config
        self.metrics_path = run_dir / "metrics.jsonl"
        self.enabled = bool(config["tracking"].get("enabled", True))
        self.tensorboard = None
        self.mlflow = None
        self.mlflow_run = None
        self.wandb = None
        if not self.enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise TrackingDependencyError("tensorboard is required; install requirements-dyphrag.txt") from error
        try:
            import mlflow
        except ImportError as error:
            raise TrackingDependencyError("mlflow is required; install requirements-dyphrag.txt") from error
        self.tensorboard = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
        self.mlflow = mlflow
        mlflow_root = run_dir.parent / "mlruns"
        mlflow.set_tracking_uri(mlflow_root.resolve().as_uri())
        mlflow.set_experiment("DyPH-RAG")
        run_id_path = run_dir / "mlflow_run_id.txt"
        if run_id_path.is_file():
            self.mlflow_run = mlflow.start_run(run_id=run_id_path.read_text(encoding="utf-8").strip())
            is_new_run = False
        else:
            self.mlflow_run = mlflow.start_run(run_name=run_dir.name)
            run_id_path.write_text(self.mlflow_run.info.run_id, encoding="utf-8")
            is_new_run = True
        safe_params = {key: str(value)[:500] for key, value in _flatten(config).items()}
        if is_new_run:
            mlflow.log_params(safe_params)
        if config["tracking"].get("wandb", False):
            try:
                import wandb
            except ImportError as error:
                raise TrackingDependencyError("W&B was configured but wandb is not installed") from error
            self.wandb = wandb
            wandb.init(project="DyPH-RAG", config=safe_params, dir=str(run_dir), mode=config["tracking"].get("wandb_mode", "offline"))
        else:
            self.wandb = None

    def log(self, step: int, metrics: dict[str, float], split: str) -> None:
        record = {"step": step, "split": split, **{key: float(value) for key, value in metrics.items()}}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=True) + "\n")
        for key, value in metrics.items():
            if value == value and self.enabled:
                name = f"{split}/{key}"
                assert self.tensorboard is not None
                assert self.mlflow is not None
                self.tensorboard.add_scalar(name, value, step)
                self.mlflow.log_metric(name, value, step=step)
        if self.wandb is not None:
            self.wandb.log({f"{split}/{key}": value for key, value in metrics.items()}, step=step)

    def finish(self, summary_path: Path) -> None:
        if not self.enabled:
            return
        assert self.tensorboard is not None
        assert self.mlflow is not None
        self.tensorboard.flush()
        self.tensorboard.close()
        self.mlflow.log_artifact(str(summary_path))
        self.mlflow.end_run()
        if self.wandb is not None:
            self.wandb.finish()
