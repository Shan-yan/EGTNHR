"""Fail-loud leakage and split audits."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .contracts import EvidenceItem, EvidenceSource, PatientDiseaseExample, SplitManifest


class LeakageError(RuntimeError):
    """Raised when future, target or held-out information leaks into a query."""


FORBIDDEN_FIELDS = {
    "label",
    "target_label",
    "ground_truth",
    "outcome",
    "mortality_outcome",
    "readmission_outcome",
    "post_discharge_outcome",
    "discharge_diagnosis",
    "label_at_cutoff",
    "future_label",
}

POST_CUTOFF_FLAGS = {
    "after_discharge",
    "post_discharge",
    "future_event",
    "future_test",
    "available_after_cutoff",
}


def _walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.add(str(key).lower())
            keys.update(_walk_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.update(_walk_keys(item))
    return keys


def audit_manifest(manifest: SplitManifest) -> None:
    groups = [set(manifest.train_patients), set(manifest.validation_patients), set(manifest.test_patients)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise LeakageError("A patient occurs in more than one split")


def audit_examples(examples: Iterable[PatientDiseaseExample], manifest: SplitManifest) -> None:
    audit_manifest(manifest)
    for example in examples:
        expected = manifest.patient_split(example.patient_id)
        if expected != example.split:
            raise LeakageError("Example split disagrees with the immutable split manifest")
        for event in example.events:
            if event.timestamp > example.cutoff_time:
                raise LeakageError("A model input contains an event after its prediction cutoff")
            forbidden = _walk_keys(event.metadata) & FORBIDDEN_FIELDS
            if forbidden:
                raise LeakageError(f"Forbidden target/outcome fields in an event: {sorted(forbidden)}")
            if event.event_type == "diagnosis" and bool(event.metadata.get("at_discharge")):
                raise LeakageError("A discharge diagnosis entered model input")
            flagged = sorted(flag for flag in POST_CUTOFF_FLAGS if bool(event.metadata.get(flag)))
            if flagged:
                raise LeakageError(f"A post-cutoff/discharge event entered model input: {flagged}")
            available_time = event.metadata.get("available_time")
            if available_time is not None:
                from .contracts import parse_time

                if parse_time(available_time) > example.cutoff_time:
                    raise LeakageError("An event was only available after the prediction cutoff")
            if event.numeric is not None and event.numeric.measurement_time is not None:
                if event.numeric.measurement_time > example.cutoff_time:
                    raise LeakageError("A numeric measurement is later than the prediction cutoff")


def audit_retrieval(query: PatientDiseaseExample, evidence: Iterable[EvidenceItem]) -> None:
    for item in evidence:
        if item.timestamp > query.cutoff_time:
            raise LeakageError("Retrieved evidence is later than the prediction cutoff")
        if item.source == EvidenceSource.PEER:
            if item.split != "train":
                raise LeakageError("Peer evidence did not come from the training split")
            if item.patient_id == query.patient_id:
                raise LeakageError("The target patient was returned as its own peer")
        forbidden = _walk_keys(item.metadata) & FORBIDDEN_FIELDS
        if forbidden:
            raise LeakageError(f"Forbidden fields in retrieved evidence: {sorted(forbidden)}")
        lowered = item.text.lower()
        markers = (
            "target_label",
            "ground_truth",
            "post_discharge_outcome",
            "future_label",
            "label_at_cutoff",
        )
        if any(marker in lowered for marker in markers):
            raise LeakageError("Retrieved text contains a target/outcome marker")
