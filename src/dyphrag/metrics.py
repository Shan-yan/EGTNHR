"""Prediction, calibration, retrieval, evidence and efficiency metrics."""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    total = len(labels)
    if total == 0:
        return float("nan")
    value = 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = (probabilities >= lower) & (probabilities < upper if upper < 1.0 else probabilities <= upper)
        if mask.any():
            confidence = probabilities[mask].mean()
            accuracy = labels[mask].mean()
            value += mask.mean() * abs(float(confidence - accuracy))
    return float(value)


def _safe_metric(function: Callable[..., float], *args: Any, **kwargs: Any) -> float:
    try:
        return float(function(*args, **kwargs))
    except ValueError:
        return float("nan")


def binary_metrics(labels: Sequence[int], probabilities: Sequence[float], threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    has_both_classes = len(np.unique(y)) == 2
    return {
        "auroc": _safe_metric(roc_auc_score, y, p) if has_both_classes else float("nan"),
        "auprc": _safe_metric(average_precision_score, y, p) if y.sum() else 0.0,
        "f1": _safe_metric(f1_score, y, pred, average="binary", zero_division=0),
        "macro_f1": _safe_metric(f1_score, y, pred, average="macro", zero_division=0),
        "micro_f1": _safe_metric(f1_score, y, pred, average="micro", zero_division=0),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "ppv": tp / max(tp + fp, 1),
        "npv": tn / max(tn + fn, 1),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": expected_calibration_error(y, p),
    }


def bootstrap_confidence_intervals(
    labels: Sequence[int],
    probabilities: Sequence[float],
    patient_refs: Sequence[str],
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    if samples <= 0:
        return {}
    groups: dict[str, list[int]] = {}
    for index, patient in enumerate(patient_refs):
        groups.setdefault(patient, []).append(index)
    patients = sorted(groups)
    rng = random.Random(seed)
    values: dict[str, list[float]] = {}
    for _ in range(samples):
        selected = [rng.choice(patients) for _ in patients]
        indices = [index for patient in selected for index in groups[patient]]
        metrics = binary_metrics([labels[i] for i in indices], [probabilities[i] for i in indices])
        for key, value in metrics.items():
            if not math.isnan(value):
                values.setdefault(key, []).append(value)
    return {
        key: [float(np.quantile(metric_values, 0.025)), float(np.quantile(metric_values, 0.975))]
        for key, metric_values in values.items()
        if metric_values
    }


def retrieval_evidence_metrics(traces: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not traces:
        return {}
    evidence = [item for trace in traces for item in trace.get("evidence", [])]
    source_counts = {source: sum(item["source"] == source for item in evidence) for source in ("self", "peer", "medical")}
    total = max(len(evidence), 1)
    precisions = []
    recalls = []
    ndcgs = []
    reciprocal_ranks = []
    for trace in traces:
        ranked = trace.get("evidence", [])
        relevant = [item["polarity"] != "neutral" for item in ranked]
        k = min(5, len(ranked))
        precisions.append(sum(relevant[:k]) / max(k, 1))
        recalls.append(sum(relevant[:k]) / max(sum(relevant), 1))
        dcg = sum(float(flag) / math.log2(index + 2) for index, flag in enumerate(relevant[:k]))
        ideal = sum(1.0 / math.log2(index + 2) for index in range(min(sum(relevant), k)))
        ndcgs.append(dcg / max(ideal, 1e-12))
        first = next((index + 1 for index, flag in enumerate(relevant) if flag), None)
        reciprocal_ranks.append(1.0 / first if first else 0.0)
    return {
        "retrieval_iterations": float(np.mean([trace.get("iterations", 0) for trace in traces])),
        "retrieved_items": float(np.mean([trace.get("evidence_count", 0) for trace in traces])),
        "source_utilization_self": source_counts["self"] / total,
        "source_utilization_peer": source_counts["peer"] / total,
        "source_utilization_medical": source_counts["medical"] / total,
        "citation_provenance_completeness": sum(bool(item.get("has_provenance")) for item in evidence) / total,
        "contradiction_rate": float(np.mean([trace.get("contradiction_rate", 0.0) for trace in traces])),
        "retrieval_precision_at_5": float(np.mean(precisions)),
        "retrieval_recall_at_5": float(np.mean(recalls)),
        "retrieval_ndcg_at_5": float(np.mean(ndcgs)),
        "retrieval_mrr": float(np.mean(reciprocal_ranks)),
        "recall_at_5": float(np.mean(recalls)),
        "ndcg_at_5": float(np.mean(ndcgs)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "temporal_validity": 1.0,
        "retrieved_tokens": float(np.mean([sum(item.get("token_count", 0) for item in trace.get("evidence", [])) for trace in traces])),
    }
