"""Dataset adapters and a privacy-safe synthetic disease-pair benchmark."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .contracts import ClinicalEvent, NumericMeasurement, PatientDiseaseExample, SplitManifest, stable_hash
from .leakage import audit_examples


@dataclass(frozen=True)
class DatasetBundle:
    train: tuple[PatientDiseaseExample, ...]
    validation: tuple[PatientDiseaseExample, ...]
    test: tuple[PatientDiseaseExample, ...]
    manifest: SplitManifest

    @property
    def corpus_hash(self) -> str:
        """Hash the exact cutoff-safe training retrieval corpus, not only its shape."""
        train_corpus = {
            "manifest": self.manifest.split_hash,
            "examples": [
                {
                    "sample": example.sample_hash,
                    "events": [
                        {
                            "id": event.event_id,
                            "time": event.timestamp.isoformat(),
                            "type": event.event_type,
                            "concept": event.concept_id,
                            "unit": event.unit_type,
                        }
                        for event in example.prior_events
                    ],
                }
                for example in self.train
            ],
        }
        return stable_hash(train_corpus)

    @property
    def cohort_hash(self) -> str:
        return stable_hash(
            {
                "split_hash": self.manifest.split_hash,
                "train_patients": sorted({example.patient_id for example in self.train}),
            }
        )


def _synthetic_example(index: int, split: str, seed: int) -> PatientDiseaseExample:
    rng = random.Random(seed * 1009 + index)
    patient_id = f"synthetic-{split}-{index:04d}"
    target = ("disease_alpha", "disease_beta")[index % 2]
    label = (index // 2 + index) % 2
    cutoff = datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=index * 3)
    risk_concept = "risk_alpha" if target == "disease_alpha" else "risk_beta"
    protective_concept = "protective_alpha" if target == "disease_alpha" else "protective_beta"
    events: list[ClinicalEvent] = []
    for offset in range(4):
        when = cutoff - timedelta(days=30 - offset * 7)
        concept = risk_concept if (label == 1 and offset >= 1) else protective_concept
        value = 0.7 + 0.1 * offset + rng.uniform(-0.03, 0.03) if label else 0.2 + rng.uniform(-0.03, 0.03)
        events.append(
            ClinicalEvent(
                event_id=f"{patient_id}-e{offset}",
                timestamp=when,
                event_type="lab" if offset % 2 else "diagnosis",
                concept_id=concept,
                text=f"documented {concept}",
                visit_id=f"{patient_id}-v{offset // 2}",
                unit_type=("visit", "event", "numeric_trend", "patient_state_hyperedge")[offset],
                numeric=NumericMeasurement(
                    concept_id="synthetic_marker",
                    raw_value=value * 100,
                    normalized_value=value,
                    unit="synthetic_unit",
                    canonical_unit="synthetic_unit",
                    reference_low=20.0,
                    reference_high=60.0,
                    abnormal_direction="high" if value > 0.6 else "normal",
                    context="synthetic fasting state",
                    measurement_condition="fasting",
                    measurement_time=when,
                    time_since_prior_hours=168.0 if offset else None,
                    slope=(0.1 / 168.0) if label and offset else 0.0,
                    baseline_deviation=value - 0.4,
                ),
                metadata={"synthetic": True, "documented_before_cutoff": True},
            )
        )
    return PatientDiseaseExample(
        patient_id=patient_id,
        target_disease=target,
        cutoff_time=cutoff,
        label=label,
        events=tuple(events),
        split=split,  # type: ignore[arg-type]
        context={"age_normalized": (index % 8) / 8.0, "history_length": 4.0},
        label_rule="positive iff repeated disease-specific synthetic risk evidence is documented before cutoff",
    )


def synthetic_bundle(seed: int = 42, train_size: int = 24, validation_size: int = 8, test_size: int = 8) -> DatasetBundle:
    cursor = 0
    splits: dict[str, tuple[PatientDiseaseExample, ...]] = {}
    for name, size in (("train", train_size), ("validation", validation_size), ("test", test_size)):
        splits[name] = tuple(_synthetic_example(cursor + i, name, seed) for i in range(size))
        cursor += size
    manifest = SplitManifest(
        dataset_name="dyphrag_synthetic",
        dataset_version="1.0",
        train_patients=tuple(x.patient_id for x in splits["train"]),
        validation_patients=tuple(x.patient_id for x in splits["validation"]),
        test_patients=tuple(x.patient_id for x in splits["test"]),
        labeling_rule="fully synthetic documented pre-cutoff risk rule",
    )
    bundle = DatasetBundle(splits["train"], splits["validation"], splits["test"], manifest)
    audit_examples((*bundle.train, *bundle.validation, *bundle.test), manifest)
    return bundle


def _event_from_json(value: dict[str, Any]) -> ClinicalEvent:
    numeric = dict(value.get("numeric") or {})
    if numeric and numeric.get("measurement_time") is None:
        numeric["measurement_time"] = value["timestamp"]
    return ClinicalEvent(
        event_id=str(value["event_id"]),
        timestamp=value["timestamp"],
        event_type=value["event_type"],
        concept_id=str(value["concept_id"]),
        text=str(value.get("text", "")),
        visit_id=value.get("visit_id"),
        unit_type=value.get("unit_type", "event"),
        numeric=NumericMeasurement(**numeric) if numeric else None,
        metadata=value.get("metadata", {}),
    )


def load_prepared_mimic(dataset: str, path: Path, version: str) -> DatasetBundle:
    """Load cutoff-safe JSONL produced inside an authorized MIMIC environment.

    Each line follows :class:`PatientDiseaseExample`. Raw MIMIC tables are never
    copied into run artifacts. A separate immutable manifest is derived here and
    audited before any retrieval index is constructed.
    """
    if dataset not in {"mimiciii", "mimiciv"}:
        raise ValueError(f"Unsupported clinical dataset: {dataset}")
    if not path.is_file():
        raise FileNotFoundError(
            f"Prepared {dataset} JSONL not found at {path}. Set DYPHRAG_MIMIC3_JSONL or DYPHRAG_MIMIC4_JSONL."
        )
    grouped: dict[str, list[PatientDiseaseExample]] = {"train": [], "validation": [], "test": []}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            raw = json.loads(line)
            split = raw.get("split")
            if split not in grouped:
                raise ValueError(f"Invalid split on line {line_number}")
            label_rule = str(raw.get("label_rule", "")).strip()
            if not label_rule:
                raise ValueError(f"A documented label_rule is required on line {line_number}")
            grouped[split].append(
                PatientDiseaseExample(
                    patient_id=str(raw["patient_id"]),
                    target_disease=str(raw["target_disease"]),
                    cutoff_time=raw["cutoff_time"],
                    label=int(raw["label"]),
                    events=tuple(_event_from_json(item) for item in raw["events"]),
                    split=split,
                    context=raw.get("context", {}),
                    label_rule=label_rule,
                )
            )
    if any(not values for values in grouped.values()):
        raise ValueError("Prepared MIMIC data must contain non-empty train/validation/test splits")
    manifest = SplitManifest(
        dataset_name=dataset,
        dataset_version=version,
        train_patients=tuple(sorted({x.patient_id for x in grouped["train"]})),
        validation_patients=tuple(sorted({x.patient_id for x in grouped["validation"]})),
        test_patients=tuple(sorted({x.patient_id for x in grouped["test"]})),
        labeling_rule=";".join(sorted({x.label_rule for values in grouped.values() for x in values})),
    )
    bundle = DatasetBundle(tuple(grouped["train"]), tuple(grouped["validation"]), tuple(grouped["test"]), manifest)
    audit_examples((*bundle.train, *bundle.validation, *bundle.test), manifest)
    return bundle


def load_dataset(config: dict[str, Any], seed: int) -> DatasetBundle:
    dataset = str(config["dataset"])
    data_cfg = config["data"]
    if dataset == "synthetic":
        return synthetic_bundle(
            seed=seed,
            train_size=int(data_cfg["train_size"]),
            validation_size=int(data_cfg["validation_size"]),
            test_size=int(data_cfg["test_size"]),
        )
    env_name = "DYPHRAG_MIMIC3_JSONL" if dataset == "mimiciii" else "DYPHRAG_MIMIC4_JSONL"
    configured = data_cfg.get("prepared_jsonl") or os.environ.get(env_name)
    if not configured:
        raise FileNotFoundError(f"Set {env_name} to an authorized, cutoff-safe prepared JSONL file")
    bundle = load_prepared_mimic(dataset, Path(configured).expanduser(), str(data_cfg["version"]))
    limits = (
        data_cfg.get("max_train_examples"),
        data_cfg.get("max_validation_examples"),
        data_cfg.get("max_test_examples"),
    )
    if any(value is not None for value in limits):
        train_limit, validation_limit, test_limit = (
            len(split) if value is None else int(value)
            for split, value in zip((bundle.train, bundle.validation, bundle.test), limits)
        )
        bundle = DatasetBundle(
            bundle.train[:train_limit],
            bundle.validation[:validation_limit],
            bundle.test[:test_limit],
            bundle.manifest,
        )
        audit_examples((*bundle.train, *bundle.validation, *bundle.test), bundle.manifest)
    return bundle
