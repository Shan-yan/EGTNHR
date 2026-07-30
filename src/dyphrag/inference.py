"""Public, auditable patient–disease–cutoff prediction API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .contracts import PatientDiseaseExample
from .model import DyPHRAGModel


@dataclass(frozen=True)
class PredictionResult:
    answer: Literal["yes", "no", "insufficient_evidence"]
    calibrated_probability: float
    abstention_probability: float
    patient_id: str
    target_disease: str
    cutoff_time: str
    evidence: tuple[dict[str, Any], ...]
    trace: dict[str, Any]


def predict(model: DyPHRAGModel, example: PatientDiseaseExample) -> PredictionResult:
    """Predict without using a label; all returned citations are cutoff-audited."""
    output = model(example)
    return PredictionResult(
        answer=output.prediction,  # type: ignore[arg-type]
        calibrated_probability=float(output.probability.detach()),
        abstention_probability=float(output.abstention_probability.detach()),
        patient_id=example.patient_id,
        target_disease=example.target_disease,
        cutoff_time=example.cutoff_time.isoformat(),
        evidence=tuple(item.safe_trace() for item in output.evidence),
        trace=output.trace,
    )
