"""Resumable parallel experiment matrix launcher for DyPH-RAG."""

from __future__ import annotations

import csv
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scipy.stats import ttest_rel

from .dyphrag.config import PROJECT_ROOT
from .dyphrag.runtime import utc_now, write_json


PRINCIPAL = [
    "baseline_ehr_gru",
    "baseline_ehr_transformer",
    "baseline_static_heterogeneous_graph",
    "baseline_visit_hypergraph",
    "baseline_graphcare_adapted",
    "baseline_kare_adapted",
    "baseline_static_vector_rag",
    "baseline_disease_retrieval",
    "baseline_dynamic_no_polarity",
    "full_dyphrag",
]
ABLATIONS = [
    "ablation_no_self",
    "ablation_no_peer",
    "ablation_no_medical",
    "ablation_no_prior",
    "ablation_uniform_router",
    "ablation_no_disease_conditioning",
    "ablation_no_time_conditioning",
    "ablation_fixed_topk",
    "ablation_single_shot",
    "ablation_frozen_retriever",
    "ablation_random_retrieval",
    "ablation_no_numeric_hyperedges",
    "ablation_no_cohort_hyperedges",
    "ablation_visit_only",
    "ablation_static_hypergraph",
    "ablation_add_only",
    "ablation_reweight_only",
    "ablation_no_deactivation",
    "ablation_ordinary_graph_expansion",
    "ablation_no_temporal_encoding",
    "ablation_numeric_bucketing",
    "ablation_no_refute",
    "ablation_no_differential",
    "ablation_no_polarity_classifier",
    "ablation_no_consistency_loss",
    "ablation_no_source_reliability",
    "ablation_no_abstention",
    "ablation_no_iterative_query",
    "ablation_shuffled_evidence_negative_control",
]
ENCODER_CONTROLS = [
    "baseline_hgnn",
    "baseline_hgat",
    "ablation_clique_graph",
    "ablation_star_graph",
    "numeric_fourier",
    "numeric_spline",
    "numeric_monotonic",
    "numeric_continuous_mlp",
    "router_sparse_gumbel",
    "router_heuristic",
]
ALL_EXPERIMENTS = [*PRINCIPAL, *ABLATIONS, *ENCODER_CONTROLS]
SMOKE = ["full_dyphrag", "baseline_ehr_gru", "ablation_no_peer"]


@dataclass(frozen=True)
class Job:
    key: str
    experiment: str
    seed: int
    run_dir: Path
    gpu: str | None


def _experiments(groups: str) -> list[str]:
    selected: list[str] = []
    for group in groups.split(","):
        group = group.strip()
        if group == "all":
            selected.extend(ALL_EXPERIMENTS)
        elif group == "principal":
            selected.extend(PRINCIPAL)
        elif group == "smoke":
            selected.extend(SMOKE)
        elif group:
            selected.append(group)
    return list(dict.fromkeys(selected))


def _available_gpus() -> list[str]:
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured and configured != "-1":
        return [value.strip() for value in configured.split(",") if value.strip()]
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True)
        return [line.strip() for line in output.splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []


def _monitor_gpu(path: Path, stop: threading.Event, interval: float) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "gpu", "utilization_percent", "memory_used_mib", "memory_total_mib"])
        while not stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                for line in output.splitlines():
                    writer.writerow([utc_now(), *[part.strip() for part in line.split(",")]])
                handle.flush()
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass
            stop.wait(interval)


def _run_job(
    job: Job,
    dataset: str,
    retry_count: int,
    extra_overrides: list[str],
    state: dict[str, Any],
    lock: threading.Lock,
    state_path: Path,
    gpu_locks: dict[str, threading.Lock],
    stop_requested: threading.Event,
) -> int:
    job.run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "src.train",
        f"experiment={job.experiment}",
        f"dataset={dataset}",
        f"seed={job.seed}",
        f"output_dir={job.run_dir}",
        "runtime.resume=true",
        *extra_overrides,
    ]
    environment = os.environ.copy()
    if job.gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = job.gpu
    gpu_lock = gpu_locks.get(job.gpu) if job.gpu is not None else None
    if gpu_lock is not None:
        while not stop_requested.is_set():
            if gpu_lock.acquire(timeout=0.5):
                break
        else:
            with lock:
                state[job.key].update({"status": "interrupted", "end": utc_now()})
                write_json(state_path, state)
            return 130
    launch_attempt = 0
    try:
        while launch_attempt <= retry_count and not stop_requested.is_set():
            launch_attempt += 1
            with lock:
                attempts = int(state[job.key].get("attempts", 0)) + 1
                state[job.key].update({"status": "running", "attempts": attempts, "start": utc_now(), "command": command})
                write_json(state_path, state)
            started = time.monotonic()
            with (job.run_dir / "stdout.log").open("a", encoding="utf-8") as stdout, (job.run_dir / "stderr.log").open("a", encoding="utf-8") as stderr:
                process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment, stdout=stdout, stderr=stderr)
                last_update = 0.0
                while process.poll() is None:
                    if stop_requested.wait(0.5):
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        break
                    if time.monotonic() - last_update >= 1.0:
                        metrics_path = job.run_dir / "metrics.jsonl"
                        progress: dict[str, Any] = {"elapsed_seconds": round(time.monotonic() - started, 1)}
                        if metrics_path.is_file():
                            lines = metrics_path.read_text(encoding="utf-8").splitlines()
                            if lines:
                                try:
                                    record = json.loads(lines[-1])
                                    progress.update(
                                        {
                                            "step": record.get("step", 0),
                                            "split": record.get("split", ""),
                                            "last_metric": record.get("loss", record.get("auprc", "")),
                                        }
                                    )
                                except json.JSONDecodeError:
                                    pass
                        with lock:
                            state[job.key].update(progress)
                            write_json(state_path, state)
                        last_update = time.monotonic()
                completed_code = process.wait()
            status = (
                "interrupted"
                if stop_requested.is_set() or completed_code in {130, -2, -15}
                else ("succeeded" if completed_code == 0 else "failed")
            )
            with lock:
                state[job.key].update({"status": status, "returncode": completed_code, "end": utc_now()})
                write_json(state_path, state)
            if completed_code == 0 or status == "interrupted":
                return 130 if status == "interrupted" else 0
        return 130 if stop_requested.is_set() else completed_code
    finally:
        if gpu_lock is not None:
            gpu_lock.release()


def _render(state: dict[str, Any]) -> Any:
    try:
        from rich.table import Table
    except ImportError:
        return " | ".join(f"{key}:{value['status']}" for key, value in sorted(state.items()))
    table = Table(title="DyPH-RAG experiment matrix")
    table.add_column("job")
    table.add_column("status")
    table.add_column("attempts")
    table.add_column("gpu")
    table.add_column("step/split")
    table.add_column("last")
    table.add_column("seconds")
    for key, value in sorted(state.items()):
        table.add_row(
            key,
            value["status"],
            str(value.get("attempts", 0)),
            str(value.get("gpu", "cpu")),
            f"{value.get('step', 0)}/{value.get('split', '')}",
            str(value.get("last_metric", ""))[:10],
            str(value.get("elapsed_seconds", "")),
        )
    return table


def aggregate(root: Path, state: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for key, value in sorted(state.items()):
        summary_path = Path(value["run_dir"]) / "summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        efficiency = summary.get("efficiency", {})
        rows.append(
            {
                "job": key,
                "experiment": value["experiment"],
                "seed": value["seed"],
                "status": summary.get("status"),
                **summary.get("metrics", {}),
                "gpu_hours": efficiency.get("gpu_hours"),
                "peak_vram_gib": efficiency.get("peak_vram_gib"),
                "training_seconds": efficiency.get("training_seconds"),
                "inference_latency_ms": efficiency.get("inference_latency_ms", summary.get("metrics", {}).get("inference_latency_ms")),
            }
        )
    write_json(root / "aggregate.json", rows)
    fields = sorted({key for row in rows for key in row}) if rows else ["job", "status"]
    with (root / "aggregate.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    statistics_rows: list[dict[str, Any]] = []
    for experiment in sorted({row["experiment"] for row in rows}):
        successful = [row for row in rows if row["experiment"] == experiment and row["status"] == "succeeded"]
        result: dict[str, Any] = {"experiment": experiment, "seeds": len(successful)}
        numeric_keys = sorted(
            {
                key
                for row in successful
                for key, value in row.items()
                if key not in {"seed"}
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and not (isinstance(value, float) and math.isnan(value))
            }
        )
        for key in numeric_keys:
            values = [
                float(row[key])
                for row in successful
                if isinstance(row.get(key), (int, float))
                and not (isinstance(row.get(key), float) and math.isnan(row[key]))
            ]
            if values:
                result[f"{key}_mean"] = statistics.fmean(values)
                result[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        statistics_rows.append(result)
    write_json(root / "aggregate_stats.json", statistics_rows)
    stats_fields = sorted({key for row in statistics_rows for key in row}) if statistics_rows else ["experiment", "seeds"]
    with (root / "aggregate_stats.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=stats_fields)
        writer.writeheader()
        writer.writerows(statistics_rows)
    comparisons: list[dict[str, Any]] = []
    full = {int(row["seed"]): row for row in rows if row["experiment"] == "full_dyphrag" and row["status"] == "succeeded"}
    for experiment in sorted({row["experiment"] for row in rows} - {"full_dyphrag"}):
        candidate = {int(row["seed"]): row for row in rows if row["experiment"] == experiment and row["status"] == "succeeded"}
        common = sorted(full.keys() & candidate.keys())
        for metric in ("auroc", "auprc", "brier", "ece"):
            left = [float(full[seed][metric]) for seed in common if metric in full[seed] and metric in candidate[seed]]
            right = [float(candidate[seed][metric]) for seed in common if metric in full[seed] and metric in candidate[seed]]
            if left:
                statistic, pvalue = ttest_rel(left, right) if len(left) >= 2 else (float("nan"), float("nan"))
                comparisons.append(
                    {
                        "experiment": experiment,
                        "metric": metric,
                        "paired_seeds": len(left),
                        "mean_delta_full_minus_candidate": sum(a - b for a, b in zip(left, right)) / len(left),
                        "paired_t": float(statistic),
                        "pvalue": float(pvalue),
                    }
                )
    write_json(root / "paired_comparisons.json", comparisons)
    lines = ["# DyPH-RAG matrix summary", "", f"Generated: {utc_now()}", "", "| Job | Status | AUROC | AUPRC |", "|---|---:|---:|---:|"]
    lines.extend(
        f"| {row['job']} | {row['status']} | {row.get('auroc', '')} | {row.get('auprc', '')} |" for row in rows
    )
    lines.extend(
        [
            "",
            "## Multi-seed mean ± standard deviation",
            "",
            "| Experiment | Seeds | AUROC | AUPRC | Brier | ECE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        "| {experiment} | {seeds} | {auroc_mean:.6f} ± {auroc_std:.6f} | "
        "{auprc_mean:.6f} ± {auprc_std:.6f} | {brier_mean:.6f} ± {brier_std:.6f} | "
        "{ece_mean:.6f} ± {ece_std:.6f} |".format(
            experiment=row["experiment"],
            seeds=row["seeds"],
            auroc_mean=row.get("auroc_mean", float("nan")),
            auroc_std=row.get("auroc_std", float("nan")),
            auprc_mean=row.get("auprc_mean", float("nan")),
            auprc_std=row.get("auprc_std", float("nan")),
            brier_mean=row.get("brier_mean", float("nan")),
            brier_std=row.get("brier_std", float("nan")),
            ece_mean=row.get("ece_mean", float("nan")),
            ece_std=row.get("ece_std", float("nan")),
        )
        for row in statistics_rows
    )
    (root / "aggregate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with tarfile.open(root / "reproducibility_bundle.tar.gz", "w:gz") as archive:
        package_paths = [
            root / "aggregate.json",
            root / "aggregate.csv",
            root / "aggregate_stats.json",
            root / "aggregate_stats.csv",
            root / "aggregate.md",
            root / "paired_comparisons.json",
            state_path(root),
        ]
        for path in package_paths:
            if path.is_file():
                archive.add(path, arcname=path.name)
        for value in state.values():
            run_dir = Path(value["run_dir"])
            for name in ("resolved_config.yaml", "summary.json", "run_metadata.json", "metrics.jsonl"):
                path = run_dir / name
                if path.is_file():
                    archive.add(path, arcname=f"{run_dir.name}/{name}")


def state_path(root: Path) -> Path:
    return root / "state.json"


def main() -> int:
    root = Path(os.environ.get("DYPHRAG_MATRIX_DIR", PROJECT_ROOT / "results" / "matrix")).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    experiments = _experiments(os.environ.get("EXPERIMENT_GROUPS", "principal"))
    seeds = [int(value) for value in os.environ.get("SEEDS", "42,43,44").split(",") if value.strip()]
    max_parallel = max(1, int(os.environ.get("MAX_PARALLEL", "1")))
    retry_count = max(0, int(os.environ.get("RETRY_COUNT", "1")))
    fail_fast = os.environ.get("FAIL_FAST", "0") == "1"
    dataset = os.environ.get("DYPHRAG_DATASET", "synthetic")
    extra_overrides = shlex.split(os.environ.get("DYPHRAG_MATRIX_OVERRIDES", ""))
    gpus = _available_gpus()
    path = state_path(root)
    state = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    jobs: list[Job] = []
    for index, (experiment, seed) in enumerate((experiment, seed) for experiment in experiments for seed in seeds):
        key = f"{experiment}_seed{seed}"
        run_dir = root / "runs" / key
        gpu = gpus[index % len(gpus)] if gpus else None
        state.setdefault(key, {"status": "pending", "attempts": 0, "experiment": experiment, "seed": seed, "run_dir": str(run_dir), "gpu": gpu or "cpu"})
        summary_path = run_dir / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") == "succeeded" and (run_dir / "checkpoints" / "best.ckpt").is_file():
                state[key]["status"] = "succeeded"
        if state[key]["status"] == "running":
            state[key]["status"] = "interrupted"
        if state[key]["status"] != "succeeded":
            jobs.append(Job(key, experiment, seed, run_dir, gpu))
    write_json(path, state)
    stop_monitor = threading.Event()
    monitor = threading.Thread(target=_monitor_gpu, args=(root / "gpu_monitor.csv", stop_monitor, 2.0), daemon=True)
    monitor.start()
    lock = threading.Lock()
    gpu_locks = {gpu: threading.Lock() for gpu in gpus}
    stop_requested = threading.Event()
    futures: dict[Future[int], Job] = {}
    try:
        from rich.live import Live
        live_context = Live(_render(state), refresh_per_second=2)
    except ImportError:
        live_context = None
    executor = ThreadPoolExecutor(max_workers=max_parallel)
    try:
        for job in jobs:
            futures[
                executor.submit(
                    _run_job,
                    job,
                    dataset,
                    retry_count,
                    extra_overrides,
                    state,
                    lock,
                    path,
                    gpu_locks,
                    stop_requested,
                )
            ] = job
        if live_context:
            live_context.start()
        try:
            while futures:
                completed, _ = wait(futures, timeout=0.5)
                if live_context:
                    live_context.update(_render(state))
                for future in completed:
                    try:
                        code = future.result()
                    except Exception:
                        code = 1
                    futures.pop(future)
                    if fail_fast and code not in {0, 130}:
                        stop_requested.set()
        except KeyboardInterrupt:
            stop_requested.set()
            for pending in futures:
                pending.cancel()
    finally:
        if live_context:
            live_context.stop()
        executor.shutdown(wait=True, cancel_futures=True)
        stop_monitor.set()
        monitor.join(timeout=3)
    aggregate(root, state)
    selected = [value for value in state.values() if value["experiment"] in experiments and value["seed"] in seeds]
    if any(value["status"] == "failed" for value in selected):
        return 1
    return 130 if any(value["status"] == "interrupted" for value in selected) else 0


if __name__ == "__main__":
    raise SystemExit(main())
