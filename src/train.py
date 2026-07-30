"""Uniform DyPH-RAG experiment entry point.

Example:
    python -m src.train experiment=full_dyphrag dataset=synthetic seed=42
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from tqdm import tqdm

from .dyphrag.config import PROJECT_ROOT, load_config, save_resolved
from .dyphrag.data import DatasetBundle, load_dataset
from .dyphrag.metrics import binary_metrics, bootstrap_confidence_intervals, retrieval_evidence_metrics
from .dyphrag.model import DyPHRAGModel
from .dyphrag.privacy import anonymous_ref, configure_logging
from .dyphrag.retrieval import RetrievalSystem
from .dyphrag.runtime import (
    capture_rng_state,
    restore_rng_state,
    run_metadata,
    set_deterministic_seed,
    utc_now,
    write_json,
)
from .dyphrag.tracking import RunTracker


LOGGER = logging.getLogger("dyphrag.train")


class ControlledInterruption(RuntimeError):
    """Used by the acceptance test to exercise exact resume."""


@dataclass
class TrainState:
    epoch: int = 0
    position: int = 0
    global_step: int = 0
    best_metric: float = float("-inf")


def _checkpoint_payload(model: DyPHRAGModel, optimizer: torch.optim.Optimizer, state: TrainState) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state": state.__dict__,
        "rng_state": capture_rng_state(),
    }


def save_checkpoint(path: Path, model: DyPHRAGModel, optimizer: torch.optim.Optimizer, state: TrainState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(_checkpoint_payload(model, optimizer, state), temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path, model: DyPHRAGModel, optimizer: torch.optim.Optimizer, device: torch.device) -> TrainState:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng_state(payload["rng_state"])
    return TrainState(**payload["state"])


def _epoch_order(size: int, seed: int, epoch: int) -> list[int]:
    order = list(range(size))
    random.Random(seed + epoch * 1_000_003).shuffle(order)
    return order


def evaluate(
    model: DyPHRAGModel,
    examples: Sequence[Any],
    run_dir: Path,
    split: str,
    step: int,
    seed: int,
    bootstrap_samples: int,
    run_salt: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    patient_refs: list[str] = []
    traces: list[dict[str, Any]] = []
    started = time.perf_counter()
    trace_path = run_dir / "retrieval_traces" / f"{split}_step{step}.jsonl"
    evidence_path = run_dir / "evidence_exports" / f"{split}_step{step}.csv"
    with trace_path.open("w", encoding="utf-8") as trace_file, evidence_path.open("w", encoding="utf-8", newline="") as evidence_file:
        writer = csv.DictWriter(
            evidence_file,
            fieldnames=[
                "example_ref",
                "evidence_ref",
                "source",
                "source_ref",
                "citation_id",
                "source_type",
                "version",
                "evidence_time",
                "polarity",
                "score",
                "reliability",
                "applicability",
                "evidence_grade",
                "has_provenance",
                "provenance_locator",
                "population",
                "license",
                "token_count",
                "expected_polarity",
                "differential_target_ref",
            ],
        )
        writer.writeheader()
        with torch.no_grad():
            for index, example in enumerate(examples):
                output = model(example)
                if example.label in (0, 1):
                    labels.append(example.label)
                    probabilities.append(float(output.probability))
                    patient_refs.append(example.patient_id)
                safe_trace = {
                    "example_ref": anonymous_ref(example.patient_id, run_salt),
                    "target_ref": anonymous_ref(example.target_disease, run_salt),
                    "prediction": output.prediction,
                    "probability": round(float(output.probability), 8),
                    "abstention_probability": round(float(output.abstention_probability), 8),
                    **output.trace,
                }
                traces.append(safe_trace)
                trace_file.write(json.dumps(safe_trace, sort_keys=True) + "\n")
                for item in output.evidence:
                    safe = item.safe_trace()
                    writer.writerow({"example_ref": safe_trace["example_ref"], **safe})
    elapsed = time.perf_counter() - started
    if not labels:
        raise ValueError(f"No known binary labels are available for {split} evaluation")
    metrics = binary_metrics(labels, probabilities)
    metrics["known_label_coverage"] = len(labels) / max(len(examples), 1)
    metrics["abstention_rate"] = sum(trace["prediction"] == "insufficient_evidence" for trace in traces) / max(len(traces), 1)
    intervals = bootstrap_confidence_intervals(labels, probabilities, patient_refs, bootstrap_samples, seed)
    for key, bounds in intervals.items():
        metrics[f"{key}_ci_low"] = bounds[0]
        metrics[f"{key}_ci_high"] = bounds[1]
    metrics.update(retrieval_evidence_metrics(traces))
    evidence_items = [item for trace in traces for item in trace["evidence"]]
    labeled_polarities = [item for item in evidence_items if item.get("expected_polarity") is not None]
    metrics["evidence_polarity_accuracy"] = (
        sum(item["polarity"] == item["expected_polarity"] for item in labeled_polarities)
        / max(len(labeled_polarities), 1)
    )
    metrics["evidence_precision"] = (
        sum(item["polarity"] != "neutral" for item in evidence_items) / max(len(evidence_items), 1)
    )
    metrics["evidence_recall"] = metrics["evidence_precision"]
    metrics["inference_latency_ms"] = elapsed * 1000.0 / max(len(examples), 1)
    metrics["peak_vram_bytes"] = float(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0.0
    for source in ("self", "peer", "medical"):
        metrics[f"retrieval_cost_items_{source}"] = float(
            sum(item.get("source") == source for item in evidence_items)
        )
    return metrics, traces


def fit_posthoc_calibration(
    model: DyPHRAGModel,
    examples: Sequence[Any],
    max_iterations: int = 50,
    minimum_coverage: float = 0.8,
) -> dict[str, float]:
    """Fit temperature and selective threshold on validation data only."""
    model.eval()
    logits: list[torch.Tensor] = []
    labels: list[int] = []
    abstentions: list[float] = []
    with torch.no_grad():
        for example in examples:
            if example.label not in (0, 1):
                continue
            output = model(example)
            logits.append(output.logits.detach())
            labels.append(int(example.label))
            abstentions.append(float(output.abstention_probability.detach()))
    if not labels:
        raise ValueError("Post-hoc calibration requires known validation labels")
    stacked_logits = torch.stack(logits)
    targets = torch.tensor(labels, device=stacked_logits.device)
    log_temperature = torch.zeros((), device=stacked_logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=max(int(max_iterations), 1),
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = torch.nn.functional.cross_entropy(stacked_logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(0.05, 20.0))
    calibrated = torch.softmax(stacked_logits / temperature, dim=-1)[:, 1]
    predictions = (calibrated >= 0.5).to(dtype=torch.int64)
    correct = (predictions == targets).to(dtype=torch.float32).cpu().tolist()
    minimum_coverage = min(max(float(minimum_coverage), 0.0), 1.0)
    candidates = sorted({0.0, 1.0, *abstentions})
    best_threshold = 1.0
    best_objective = float("-inf")
    best_coverage = 1.0
    for threshold in candidates:
        retained = [index for index, value in enumerate(abstentions) if value <= threshold]
        coverage = len(retained) / len(labels)
        if coverage + 1e-12 < minimum_coverage or not retained:
            continue
        selective_accuracy = sum(correct[index] for index in retained) / len(retained)
        objective = selective_accuracy - 0.05 * (1.0 - coverage)
        if objective > best_objective or (
            objective == best_objective and coverage > best_coverage
        ):
            best_objective = objective
            best_threshold = threshold
            best_coverage = coverage
    model.set_posthoc_calibration(temperature, best_threshold)
    return {
        "temperature": temperature,
        "abstention_threshold": best_threshold,
        "validation_coverage": best_coverage,
        "minimum_coverage": minimum_coverage,
        "selection_objective": best_objective,
        "fitted_on_validation_only": 1.0,
        "validation_examples": float(len(labels)),
    }


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config["seed"])
    set_deterministic_seed(seed)
    run_dir = Path(config["output_dir"]).expanduser()
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "retrieval_traces").mkdir(exist_ok=True)
    (run_dir / "evidence_exports").mkdir(exist_ok=True)
    (run_dir / "indices").mkdir(exist_ok=True)
    existing_summary = run_dir / "summary.json"
    if bool(config["runtime"]["resume"]) and existing_summary.is_file():
        previous = json.loads(existing_summary.read_text(encoding="utf-8"))
        if previous.get("status") == "succeeded" and (run_dir / "checkpoints" / "best.ckpt").is_file():
            LOGGER.info("successful run already complete; skipping", extra={"fields": {"global_step": previous.get("global_step")}})
            return previous
    save_resolved(config, run_dir / "resolved_config.yaml")
    bundle: DatasetBundle = load_dataset(config, seed)
    retrieval = RetrievalSystem(bundle.train, config, seed)
    write_json(
        run_dir / "indices" / "cohort_manifest.json",
        {
            "split_hash": bundle.manifest.split_hash,
            "peer_index_hash": retrieval.peer_retriever.corpus_hash,
            "medical_index_hash": retrieval.medical_retriever.corpus_hash,
            "retrieval_corpus_hash": retrieval.corpus_hash,
            "cohort_hash": bundle.cohort_hash,
            "patient_count": len(bundle.train),
            "features": ["diagnosis", "numeric_trajectory", "medication_trajectory", "demographic_context", "disease_stage", "treatment_response"],
        },
    )
    torch.save(torch.stack(retrieval.peer_retriever.vectors), run_dir / "indices" / "cohort_vectors.pt")
    metadata = run_metadata(PROJECT_ROOT, config, bundle.manifest.split_hash, retrieval.corpus_hash)
    write_json(run_dir / "run_metadata.json", metadata)
    device_name = str(config["runtime"]["device"])
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = DyPHRAGModel(config, retrieval).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]))
    tracker = RunTracker(run_dir, config)
    state = TrainState()
    last_path = run_dir / "checkpoints" / "last.ckpt"
    best_path = run_dir / "checkpoints" / "best.ckpt"
    calibrated_path = run_dir / "checkpoints" / "calibrated.ckpt"
    if bool(config["runtime"]["resume"]) and last_path.is_file():
        state = load_checkpoint(last_path, model, optimizer, device)
        LOGGER.info("resumed checkpoint", extra={"fields": {"global_step": state.global_step, "epoch": state.epoch, "position": state.position}})
    started = time.perf_counter()
    run_salt = bundle.manifest.split_hash[:16]
    interrupted = False
    try:
        epochs = int(config["training"]["epochs"])
        checkpoint_every = int(config["training"]["checkpoint_every_steps"])
        interrupt_after = int(config["runtime"]["interrupt_after_steps"])
        while state.epoch < epochs:
            order = _epoch_order(len(bundle.train), seed, state.epoch)
            progress = tqdm(
                range(state.position, len(order)),
                desc=f"epoch {state.epoch + 1}/{epochs}",
                leave=False,
                disable=not bool(config["runtime"]["progress"]),
            )
            for position in progress:
                example = bundle.train[order[position]]
                model.train()
                optimizer.zero_grad(set_to_none=True)
                output = model(example, example.label)
                if output.loss is None:
                    raise RuntimeError("Training loss was not produced")
                output.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["max_grad_norm"]))
                optimizer.step()
                state.global_step += 1
                state.position = position + 1
                loss_value = float(output.loss.detach())
                tracker.log(state.global_step, {"loss": loss_value}, "train")
                progress.set_postfix(loss=f"{loss_value:.4f}")
                if checkpoint_every > 0 and state.global_step % checkpoint_every == 0:
                    save_checkpoint(last_path, model, optimizer, state)
                if interrupt_after > 0 and state.global_step >= interrupt_after:
                    save_checkpoint(last_path, model, optimizer, state)
                    raise ControlledInterruption(f"forced interruption after step {state.global_step}")
            state.epoch += 1
            state.position = 0
            validation, _ = evaluate(
                model,
                bundle.validation,
                run_dir,
                "validation",
                state.global_step,
                seed,
                int(config["evaluation"]["bootstrap_samples"]),
                run_salt,
            )
            tracker.log(state.global_step, validation, "validation")
            monitor = float(validation[config["training"]["monitor"]])
            if monitor == monitor and monitor > state.best_metric:
                state.best_metric = monitor
                save_checkpoint(best_path, model, optimizer, state)
            save_checkpoint(last_path, model, optimizer, state)
        if not best_path.is_file():
            save_checkpoint(best_path, model, optimizer, state)
        training_seconds = time.perf_counter() - started
        load_checkpoint(best_path, model, optimizer, device)
        calibration = fit_posthoc_calibration(
            model,
            bundle.validation,
            int(config["evaluation"].get("calibration_max_iterations", 50)),
            float(config["evaluation"].get("minimum_selective_coverage", 0.8)),
        )
        save_checkpoint(calibrated_path, model, optimizer, state)
        test_metrics, _ = evaluate(
            model,
            bundle.test,
            run_dir,
            "test",
            state.global_step,
            seed,
            int(config["evaluation"]["bootstrap_samples"]),
            run_salt,
        )
        tracker.log(state.global_step, test_metrics, "test")
        total_runtime_seconds = time.perf_counter() - started
        peak_vram = float(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0.0
        summary = {
            "status": "succeeded",
            "experiment": config["experiment"],
            "dataset": config["dataset"],
            "seed": seed,
            "split_hash": bundle.manifest.split_hash,
            "cohort_hash": bundle.cohort_hash,
            "retrieval_corpus_hash": retrieval.corpus_hash,
            "best_checkpoint": str(calibrated_path),
            "uncalibrated_model_selection_checkpoint": str(best_path),
            "calibration": calibration,
            "best_metric": state.best_metric,
            "global_step": state.global_step,
            "metrics": test_metrics,
            "efficiency": {
                "training_seconds": training_seconds,
                "total_runtime_seconds": total_runtime_seconds,
                "gpu_hours": total_runtime_seconds / 3600.0 if device.type == "cuda" else 0.0,
                "peak_vram_bytes": peak_vram,
                "peak_vram_gib": peak_vram / (1024.0**3),
                "inference_latency_ms": test_metrics["inference_latency_ms"],
                "index_size_items": len(bundle.train),
                "cpu_peak_rss_bytes": __import__("psutil").Process().memory_info().rss,
            },
            "end_timestamp": utc_now(),
        }
    except (ControlledInterruption, KeyboardInterrupt) as error:
        interrupted = True
        save_checkpoint(last_path, model, optimizer, state)
        summary = {
            "status": "interrupted",
            "reason": type(error).__name__,
            "seed": seed,
            "global_step": state.global_step,
            "last_checkpoint": str(last_path),
            "end_timestamp": utc_now(),
        }
    summary_path = run_dir / "summary.json"
    write_json(summary_path, summary)
    tracker.finish(summary_path)
    if interrupted:
        raise ControlledInterruption("run interrupted after writing an exact-resume checkpoint")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging()
    arguments = list(sys.argv[1:] if argv is None else argv)
    config: dict[str, Any] | None = None
    try:
        config = load_config(arguments)
        summary = run_training(config)
    except ControlledInterruption as error:
        LOGGER.warning(str(error))
        return 130
    except Exception as error:
        LOGGER.exception("run failed")
        if config is not None:
            run_dir = Path(config["output_dir"]).expanduser()
            if not run_dir.is_absolute():
                run_dir = PROJECT_ROOT / run_dir
            run_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                run_dir / "summary.json",
                {
                    "status": "failed",
                    "experiment": config.get("experiment"),
                    "dataset": config.get("dataset"),
                    "seed": config.get("seed"),
                    "reason": type(error).__name__,
                    "message": str(error),
                    "end_timestamp": utc_now(),
                },
            )
        return 1
    LOGGER.info("run completed", extra={"fields": {"status": summary["status"], "global_step": summary["global_step"]}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
