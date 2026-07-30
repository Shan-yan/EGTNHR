"""Typed, cutoff-aware data contracts used by every DyPH-RAG component."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Mapping, Sequence


class ContractError(ValueError):
    """Raised when an input violates a DyPH-RAG data contract."""


class EvidencePolarity(str, Enum):
    SUPPORT = "support"
    REFUTE = "refute"
    DIFFERENTIAL = "differential"
    NEUTRAL = "neutral"


class EvidenceSource(str, Enum):
    SELF = "self"
    PEER = "peer"
    MEDICAL = "medical"


ALLOWED_EVENT_UNITS = {
    "visit",
    "event",
    "episode",
    "numeric_trend_window",
    "patient-state_hyperedge",
    # Backward-compatible spellings used by the first local prototype.
    "temporal_episode",
    "numeric_trend",
    "patient_state_hyperedge",
}


def parse_time(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NumericMeasurement:
    concept_id: str
    raw_value: float
    normalized_value: float
    unit: str
    canonical_unit: str
    reference_low: float | None = None
    reference_high: float | None = None
    abnormal_direction: Literal["low", "normal", "high", "unknown"] = "unknown"
    context: str = ""
    measurement_condition: str = ""
    measurement_time: datetime | None = None
    time_since_prior_hours: float | None = None
    slope: float | None = None
    baseline_deviation: float | None = None
    missing: bool = False
    observed: bool = True

    def __post_init__(self) -> None:
        if self.measurement_time is not None:
            object.__setattr__(self, "measurement_time", parse_time(self.measurement_time))
        if self.missing and self.observed:
            object.__setattr__(self, "observed", False)


@dataclass(frozen=True)
class ClinicalEvent:
    event_id: str
    timestamp: datetime
    event_type: Literal["diagnosis", "medication", "procedure", "lab", "vital", "context"]
    concept_id: str
    text: str = ""
    visit_id: str | None = None
    unit_type: str = "event"
    numeric: NumericMeasurement | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", parse_time(self.timestamp))
        if self.unit_type not in ALLOWED_EVENT_UNITS:
            raise ContractError(f"Unsupported retrieval unit: {self.unit_type}")


@dataclass(frozen=True)
class PatientDiseaseExample:
    patient_id: str
    target_disease: str
    cutoff_time: datetime
    label: int
    events: Sequence[ClinicalEvent]
    split: Literal["train", "validation", "test"]
    context: Mapping[str, float] = field(default_factory=dict)
    label_rule: str = "synthetic_documented_rule"

    def __post_init__(self) -> None:
        object.__setattr__(self, "cutoff_time", parse_time(self.cutoff_time))
        if self.label not in (0, 1, 2):
            raise ContractError("label must be no=0, yes=1, or unknown=2")
        if not self.patient_id or not self.target_disease:
            raise ContractError("patient_id and target_disease are required")

    @property
    def prior_events(self) -> tuple[ClinicalEvent, ...]:
        return tuple(event for event in self.events if event.timestamp <= self.cutoff_time)

    @property
    def sample_key(self) -> tuple[str, str, str]:
        return self.patient_id, self.target_disease, self.cutoff_time.isoformat()

    @property
    def sample_hash(self) -> str:
        return stable_hash(self.sample_key)


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    source: EvidenceSource
    source_id: str
    timestamp: datetime
    text: str
    score: float
    polarity: EvidencePolarity = EvidencePolarity.NEUTRAL
    patient_id: str | None = None
    split: str | None = None
    source_type: str = "ehr"
    version: str = "1"
    population: str = "unspecified"
    evidence_grade: str = "ungraded"
    reliability: float = 1.0
    applicability: float = 1.0
    provenance_span: str = ""
    license: str = "internal-research"
    supersedes: tuple[str, ...] = ()
    contradicts: tuple[str, ...] = ()
    differential_disease: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", parse_time(self.timestamp))
        if not 0.0 <= float(self.reliability) <= 1.0:
            raise ContractError("evidence reliability must be in [0, 1]")
        if not 0.0 <= float(self.applicability) <= 1.0:
            raise ContractError("evidence applicability must be in [0, 1]")

    def safe_trace(self) -> dict[str, Any]:
        """Return provenance diagnostics without patient IDs or clinical text."""
        external = self.source == EvidenceSource.MEDICAL
        return {
            "evidence_ref": stable_hash(self.evidence_id)[:16],
            "source": self.source.value,
            "source_ref": stable_hash(self.source_id)[:16],
            "citation_id": self.source_id if external else "",
            "source_type": self.source_type,
            "version": self.version,
            "evidence_time": self.timestamp.isoformat(),
            "score": round(float(self.score), 6),
            "polarity": self.polarity.value,
            "reliability": float(self.reliability),
            "applicability": float(self.applicability),
            "evidence_grade": self.evidence_grade,
            "has_provenance": bool(self.provenance_span),
            "provenance_locator": self.provenance_span if external else "",
            "population": self.population if external else "",
            "license": self.license if external else "",
            "token_count": len(self.text.split()),
            "expected_polarity": self.metadata.get("polarity_label"),
            "differential_target_ref": (
                stable_hash(self.differential_disease)[:16]
                if self.differential_disease
                else ""
            ),
        }


@dataclass(frozen=True)
class SplitManifest:
    dataset_name: str
    dataset_version: str
    train_patients: tuple[str, ...]
    validation_patients: tuple[str, ...]
    test_patients: tuple[str, ...]
    labeling_rule: str

    @property
    def split_hash(self) -> str:
        return stable_hash(asdict(self))

    def patient_split(self, patient_id: str) -> str | None:
        for split in ("train", "validation", "test"):
            if patient_id in getattr(self, f"{split}_patients"):
                return split
        return None
