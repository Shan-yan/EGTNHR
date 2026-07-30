#!/usr/bin/env python3
"""Prepare cutoff-safe, pseudonymous MIMIC disease-pair JSONL for DyPH-RAG.

The operational task is admission-time prediction of diagnosis codes eventually
recorded for the index admission. Diagnosis rows are labels only and never
become model or retrieval features. A negative means "not coded in this
admission", not clinically disease-free.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import logging
import random
import re
import secrets
import stat
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


LOGGER = logging.getLogger("dyphrag.prepare_mimic")
CODE_CLEANER = re.compile(r"[^A-Z0-9]+")


@dataclass(frozen=True)
class Admission:
    subject_id: str
    admission_id: str
    admit_time: datetime
    discharge_time: datetime
    admission_type: str


@dataclass(frozen=True)
class Demographics:
    gender: str
    birth_time: datetime | None = None
    anchor_age: float | None = None
    anchor_year: int | None = None


def parse_time(value: str | None) -> datetime | None:
    value = (value or "").strip()
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    for candidate in (normalized, normalized.replace(" ", "T")):
        try:
            result = datetime.fromisoformat(candidate)
            if result.tzinfo is None:
                result = result.replace(tzinfo=timezone.utc)
            return result.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def rows(path: Path) -> Iterator[dict[str, str]]:
    csv.field_size_limit(16 * 1024 * 1024)
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        yield from csv.DictReader(handle)


def load_or_create_salt(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        value = path.read_text(encoding="utf-8").strip()
        if len(value) < 32:
            raise ValueError("The pseudonym salt file is unexpectedly short")
        return bytes.fromhex(value)
    value = secrets.token_bytes(32)
    path.write_text(value.hex(), encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return value


def pseudonym(salt: bytes, namespace: str, value: str) -> str:
    digest = hmac.new(salt, f"{namespace}:{value}".encode(), hashlib.sha256).hexdigest()
    return digest[:24]


def normalize_code(value: str, prefix_length: int) -> str:
    return CODE_CLEANER.sub("", value.upper())[:prefix_length]


def safe_concept(value: str, limit: int = 80) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.:/+-]+", "_", value.strip())
    return cleaned[:limit] or "UNKNOWN"


def dataset_paths(root: Path, dataset: str) -> dict[str, Path]:
    if dataset == "mimiciii":
        names = {
            "admissions": "ADMISSIONS.csv",
            "diagnoses": "DIAGNOSES_ICD.csv",
            "patients": "PATIENTS.csv",
            "prescriptions": "PRESCRIPTIONS.csv",
            "procedures": "PROCEDURES_ICD.csv",
        }
    else:
        names = {
            "admissions": "admissions.csv",
            "diagnoses": "diagnoses_icd.csv",
            "patients": "patients.csv",
            "prescriptions": "prescriptions.csv",
            "procedures": "procedures_icd.csv",
        }
    result = {key: root / name for key, name in names.items()}
    missing = [str(path) for path in result.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required MIMIC tables: {missing}")
    return result


def load_demographics(path: Path, dataset: str) -> dict[str, Demographics]:
    result = {}
    for row in rows(path):
        subject = row["SUBJECT_ID"] if dataset == "mimiciii" else row["subject_id"]
        if dataset == "mimiciii":
            result[subject] = Demographics(row.get("GENDER", ""), parse_time(row.get("DOB")))
        else:
            anchor_age = float(row["anchor_age"]) if row.get("anchor_age") else None
            anchor_year = int(row["anchor_year"]) if row.get("anchor_year") else None
            result[subject] = Demographics(row.get("gender", ""), None, anchor_age, anchor_year)
    return result


def load_admissions(path: Path, dataset: str) -> dict[str, list[Admission]]:
    grouped: dict[str, list[Admission]] = defaultdict(list)
    for row in rows(path):
        upper = dataset == "mimiciii"
        subject = row["SUBJECT_ID"] if upper else row["subject_id"]
        admission = row["HADM_ID"] if upper else row["hadm_id"]
        admit = parse_time(row.get("ADMITTIME" if upper else "admittime"))
        discharge = parse_time(row.get("DISCHTIME" if upper else "dischtime"))
        if admit is None or discharge is None or discharge < admit:
            continue
        grouped[subject].append(
            Admission(
                subject,
                admission,
                admit,
                discharge,
                row.get("ADMISSION_TYPE" if upper else "admission_type", "UNKNOWN"),
            )
        )
    for admissions in grouped.values():
        admissions.sort(key=lambda item: (item.admit_time, item.discharge_time, item.admission_id))
    return grouped


def patient_splits(
    admissions: dict[str, list[Admission]],
    max_patients: int | None,
    strategy: str,
    seed: int,
) -> dict[str, str]:
    eligible = [subject for subject, visits in admissions.items() if len(visits) >= 2]
    if strategy == "temporal":
        eligible.sort(key=lambda subject: (admissions[subject][0].admit_time, subject))
    else:
        random.Random(seed).shuffle(eligible)
    if max_patients is not None:
        eligible = eligible[:max_patients]
    train_end = int(len(eligible) * 0.8)
    validation_end = int(len(eligible) * 0.9)
    return {
        subject: "train" if index < train_end else ("validation" if index < validation_end else "test")
        for index, subject in enumerate(eligible)
    }


def index_admissions(admissions: dict[str, list[Admission]], splits: dict[str, str]) -> tuple[dict[str, str], set[str]]:
    admission_split = {}
    all_feature_admissions: set[str] = set()
    for subject in splits:
        visits = admissions[subject]
        all_feature_admissions.update(visit.admission_id for visit in visits[:-1])
        for visit in visits[1:]:
            admission_split[visit.admission_id] = splits[subject]
    return admission_split, all_feature_admissions


def diagnosis_vocabulary(
    path: Path,
    dataset: str,
    admission_split: dict[str, str],
    prefix_length: int,
    top_k: int,
    min_train_positive: int,
) -> tuple[str, ...]:
    per_admission: dict[str, set[str]] = defaultdict(set)
    upper = dataset == "mimiciii"
    for row in rows(path):
        admission = row["HADM_ID"] if upper else row["hadm_id"]
        if admission_split.get(admission) != "train":
            continue
        code = normalize_code(row["ICD9_CODE"] if upper else row["icd_code"], prefix_length)
        if code:
            version = "ICD9" if upper or row.get("icd_version") == "9" else "ICD10"
            per_admission[admission].add(f"{version}_CATEGORY:{code}")
    prevalence = Counter(code for codes in per_admission.values() for code in codes)
    return tuple(code for code, count in prevalence.most_common(top_k) if count >= min_train_positive)


def diagnosis_labels(
    path: Path,
    dataset: str,
    admission_split: dict[str, str],
    vocabulary: set[str],
    prefix_length: int,
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    upper = dataset == "mimiciii"
    for row in rows(path):
        admission = row["HADM_ID"] if upper else row["hadm_id"]
        if admission not in admission_split:
            continue
        code = normalize_code(row["ICD9_CODE"] if upper else row["icd_code"], prefix_length)
        version = "ICD9" if upper or row.get("icd_version") == "9" else "ICD10"
        target = f"{version}_CATEGORY:{code}"
        if target in vocabulary:
            result[admission].add(target)
    return result


def load_procedures(
    path: Path,
    dataset: str,
    eligible_admissions: set[str],
    cap: int,
) -> dict[str, dict[str, datetime | None]]:
    result: dict[str, dict[str, datetime | None]] = defaultdict(dict)
    upper = dataset == "mimiciii"
    for row in rows(path):
        admission = row["HADM_ID"] if upper else row["hadm_id"]
        if admission not in eligible_admissions or len(result[admission]) >= cap:
            continue
        raw = row.get("ICD9_CODE" if upper else "icd_code", "")
        version = "ICD9" if upper or row.get("icd_version") == "9" else "ICD10"
        concept = f"PROC_{version}:{safe_concept(raw)}"
        result[admission].setdefault(concept, parse_time(row.get("chartdate")) if not upper else None)
    return result


def load_medications(
    path: Path,
    dataset: str,
    eligible_admissions: set[str],
    cap: int,
) -> dict[str, dict[str, datetime | None]]:
    result: dict[str, dict[str, datetime | None]] = defaultdict(dict)
    upper = dataset == "mimiciii"
    for index, row in enumerate(rows(path), 1):
        admission = row["HADM_ID"] if upper else row["hadm_id"]
        if admission in eligible_admissions:
            code = (
                row.get("FORMULARY_DRUG_CD" if upper else "formulary_drug_cd")
                or row.get("NDC" if upper else "ndc")
                or row.get("DRUG_NAME_GENERIC" if upper else "drug")
                or ""
            )
            concept = f"MED:{safe_concept(code)}"
            if concept != "MED:UNKNOWN" and (concept in result[admission] or len(result[admission]) < cap):
                timestamp = parse_time(row.get("STARTDATE" if upper else "starttime"))
                current = result[admission].get(concept)
                if current is None or (timestamp is not None and timestamp < current):
                    result[admission][concept] = timestamp
        if index % 2_000_000 == 0:
            LOGGER.info("prescription scan progress", extra={"rows_scanned": index})
    return result


def age_at(demographics: Demographics | None, cutoff: datetime) -> float:
    if demographics is None:
        return 0.0
    if demographics.birth_time is not None:
        years = (cutoff - demographics.birth_time).total_seconds() / (365.2425 * 86400.0)
        return min(max(years, 0.0), 90.0)
    if demographics.anchor_age is not None and demographics.anchor_year is not None:
        return min(max(demographics.anchor_age + cutoff.year - demographics.anchor_year, 0.0), 90.0)
    return 0.0


def event(
    salt: bytes,
    admission: Admission,
    order: int,
    timestamp: datetime,
    event_type: str,
    concept_id: str,
    unit_type: str,
    source_table: str,
    imputed: bool,
) -> dict[str, Any]:
    visit_ref = f"v_{pseudonym(salt, 'visit', admission.admission_id)}"
    event_ref = pseudonym(salt, "event", f"{admission.admission_id}:{order}:{event_type}:{concept_id}")
    return {
        "event_id": f"e_{event_ref}",
        "timestamp": timestamp.isoformat(),
        "event_type": event_type,
        "concept_id": concept_id,
        "text": concept_id,
        "visit_id": visit_ref,
        "unit_type": unit_type,
        "metadata": {
            "source_table": source_table,
            "historical_completed_admission": True,
            "time_imputed_to_discharge": imputed,
        },
    }


def build_events(
    salt: bytes,
    historical: list[Admission],
    cutoff: datetime,
    medications: dict[str, dict[str, datetime | None]],
    procedures: dict[str, dict[str, datetime | None]],
    max_events: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    order = 0
    for admission in historical:
        if admission.discharge_time > cutoff:
            continue
        result.append(
            event(
                salt,
                admission,
                order,
                admission.discharge_time,
                "context",
                f"ADMISSION_TYPE:{safe_concept(admission.admission_type)}",
                "visit",
                "admissions",
                False,
            )
        )
        order += 1
        for concept, recorded_time in sorted(procedures.get(admission.admission_id, {}).items()):
            if recorded_time is not None and recorded_time > cutoff:
                continue
            when = recorded_time if recorded_time is not None else admission.discharge_time
            result.append(event(salt, admission, order, when, "procedure", concept, "event", "procedures_icd", recorded_time is None))
            order += 1
        for concept, recorded_time in sorted(medications.get(admission.admission_id, {}).items()):
            if recorded_time is not None and recorded_time > cutoff:
                continue
            when = recorded_time if recorded_time is not None else admission.discharge_time
            result.append(event(salt, admission, order, when, "medication", concept, "temporal_episode", "prescriptions", recorded_time is None))
            order += 1
    result.sort(key=lambda item: (item["timestamp"], item["event_id"]))
    return result[-max_events:]


def write_dataset(
    output: Path,
    salt: bytes,
    dataset: str,
    admissions: dict[str, list[Admission]],
    demographics: dict[str, Demographics],
    splits: dict[str, str],
    vocabulary: tuple[str, ...],
    labels: dict[str, set[str]],
    medications: dict[str, dict[str, datetime | None]],
    procedures: dict[str, dict[str, datetime | None]],
    negative_ratio: int,
    max_positive_per_admission: int,
    max_events: int,
    seed: int,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    label_rule = (
        "Admission-time ICD category coding prediction: positive iff the target category is recorded in the index "
        "admission diagnoses table; negative iff it is not recorded. Negative denotes coding absence only, not "
        "clinical disease absence. Features use completed pre-index admissions only; diagnosis rows are labels only."
    )
    with output.open("w", encoding="utf-8") as handle:
        for subject in sorted(splits):
            split = splits[subject]
            patient_ref = f"p_{pseudonym(salt, 'patient', subject)}"
            visits = admissions[subject]
            for index_admission_position in range(1, len(visits)):
                index_admission = visits[index_admission_position]
                positives = sorted(labels.get(index_admission.admission_id, set()))
                if not positives:
                    continue
                positives = positives[:max_positive_per_admission]
                negative_candidates = [target for target in vocabulary if target not in labels[index_admission.admission_id]]
                rng_seed = int(pseudonym(salt, "sampling", f"{subject}:{index_admission.admission_id}:{seed}"), 16)
                rng = random.Random(rng_seed)
                rng.shuffle(negative_candidates)
                targets = [(target, 1) for target in positives]
                targets.extend((target, 0) for target in negative_candidates[: len(positives) * negative_ratio])
                historical = [visit for visit in visits[:index_admission_position] if visit.discharge_time <= index_admission.admit_time]
                events = build_events(salt, historical, index_admission.admit_time, medications, procedures, max_events)
                if not events:
                    continue
                demo = demographics.get(subject)
                context = {
                    "age_normalized": age_at(demo, index_admission.admit_time) / 90.0,
                    "sex_female": float(bool(demo and demo.gender.upper() == "F")),
                    "history_length": float(len(historical)),
                    "missing_numerical_tables": 1.0,
                }
                for target, label in targets:
                    record = {
                        "patient_id": patient_ref,
                        "target_disease": target,
                        "cutoff_time": index_admission.admit_time.isoformat(),
                        "label": label,
                        "label_rule": label_rule,
                        "split": split,
                        "context": context,
                        "events": events,
                    }
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    counts[f"{split}_examples"] += 1
                    counts[f"{split}_positive"] += label
    corpus_digest = hashlib.sha256()
    with output.open("rb") as corpus:
        for chunk in iter(lambda: corpus.read(1024 * 1024), b""):
            corpus_digest.update(chunk)
    digest = corpus_digest.hexdigest()
    split_payload = {
        split: sorted(pseudonym(salt, "patient", subject) for subject, assigned in splits.items() if assigned == split)
        for split in ("train", "validation", "test")
    }
    return {
        "dataset": dataset,
        "dataset_version": "MIMIC-III v1.4" if dataset == "mimiciii" else "MIMIC-IV local core",
        "task": "admission_time_disease_code_prediction",
        "label_rule": label_rule,
        "split_strategy": "patient_disjoint",
        "split_hash": hashlib.sha256(json.dumps(split_payload, sort_keys=True).encode()).hexdigest(),
        "corpus_hash": digest,
        "patient_counts": Counter(splits.values()),
        "example_counts": counts,
        "target_disease_count": len(vocabulary),
        "target_vocabulary_hash": hashlib.sha256(json.dumps(vocabulary).encode()).hexdigest(),
        "numerical_state_status": "blocked: LABEVENTS/CHARTEVENTS not present in supplied directory",
        "identifier_status": "raw patient/admission identifiers replaced with keyed HMAC pseudonyms before output",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mimiciii", "mimiciv"), default="mimiciii")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--salt-file", type=Path, required=True)
    parser.add_argument("--split-strategy", choices=("temporal", "random"), default="temporal")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-patients", type=int)
    parser.add_argument("--top-k-diseases", type=int, default=50)
    parser.add_argument("--min-train-positive", type=int, default=25)
    parser.add_argument("--code-prefix-length", type=int, default=3)
    parser.add_argument("--negative-ratio", type=int, default=1)
    parser.add_argument("--max-positive-per-admission", type=int, default=2)
    parser.add_argument("--max-events", type=int, default=128)
    parser.add_argument("--max-medications-per-admission", type=int, default=24)
    parser.add_argument("--max-procedures-per-admission", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    paths = dataset_paths(args.root, args.dataset)
    salt = load_or_create_salt(args.salt_file)
    demographics = load_demographics(paths["patients"], args.dataset)
    admissions = load_admissions(paths["admissions"], args.dataset)
    splits = patient_splits(admissions, args.max_patients, args.split_strategy, args.seed)
    if len(splits) < 30:
        raise ValueError("Fewer than 30 patients with repeated admissions are available")
    admission_split, feature_admissions = index_admissions(admissions, splits)
    vocabulary = diagnosis_vocabulary(
        paths["diagnoses"],
        args.dataset,
        admission_split,
        args.code_prefix_length,
        args.top_k_diseases,
        args.min_train_positive,
    )
    if not vocabulary:
        raise ValueError("No target diseases meet the configured training prevalence threshold")
    labels = diagnosis_labels(
        paths["diagnoses"], args.dataset, admission_split, set(vocabulary), args.code_prefix_length
    )
    procedures = load_procedures(
        paths["procedures"], args.dataset, feature_admissions, args.max_procedures_per_admission
    )
    medications = load_medications(
        paths["prescriptions"], args.dataset, feature_admissions, args.max_medications_per_admission
    )
    manifest = write_dataset(
        args.output,
        salt,
        args.dataset,
        admissions,
        demographics,
        splits,
        vocabulary,
        labels,
        medications,
        procedures,
        args.negative_ratio,
        args.max_positive_per_admission,
        args.max_events,
        args.seed,
    )
    manifest["split_strategy"] = args.split_strategy
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=dict), encoding="utf-8")
    LOGGER.info(
        "prepared dataset complete: patients=%d examples=%d targets=%d",
        sum(manifest["patient_counts"].values()),
        sum(value for key, value in manifest["example_counts"].items() if key.endswith("_examples")),
        manifest["target_disease_count"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
