"""Prepare KARE EHR artifacts from a local MIMIC-III or MIMIC-IV copy."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path


RESOURCE_DIR = Path(__file__).resolve().parent / "resources"


def load_mappings(resource_dir: Path = RESOURCE_DIR):
    def read_mapping(filename: str, *, level3_only: bool = False):
        result = {}
        with (resource_dir / filename).open(newline="") as csvfile:
            for row in csv.DictReader(csvfile):
                if not level3_only or row.get("level") == "3.0":
                    result[row["code"]] = row["name"].lower()
        return result

    return (
        read_mapping("CCSCM.csv"),
        read_mapping("CCSPROC.csv"),
        read_mapping("ATC.csv", level3_only=True),
    )


def load_dataset(dataset: str, root: Path, *, dev: bool, refresh_cache: bool):
    from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset

    if dataset == "mimic3":
        return MIMIC3Dataset(
            root=str(root),
            tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
            code_mapping={
                "NDC": ("ATC", {"target_kwargs": {"level": 3}}),
                "ICD9CM": "CCSCM",
                "ICD9PROC": "CCSPROC",
            },
            dev=dev,
            refresh_cache=refresh_cache,
        )
    return MIMIC4Dataset(
        root=str(root),
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        code_mapping={
            "NDC": ("ATC", {"target_kwargs": {"level": 3}}),
            "ICD9CM": "CCSCM",
            "ICD9PROC": "CCSPROC",
            "ICD10CM": "CCSCM",
            "ICD10PROC": "CCSPROC",
        },
        dev=dev,
        refresh_cache=refresh_cache,
    )


def assign_task(dataset: str, ds, task: str):
    try:
        from .utils import (
            drug_recommendation_mimic3_fn,
            drug_recommendation_mimic4_fn,
            length_of_stay_prediction_mimic3_fn,
            length_of_stay_prediction_mimic4_fn,
            mortality_prediction_mimic3_fn,
            mortality_prediction_mimic4_fn,
            readmission_prediction_mimic3_fn,
            readmission_prediction_mimic4_fn,
        )
    except ImportError:
        from utils import (  # type: ignore
            drug_recommendation_mimic3_fn,
            drug_recommendation_mimic4_fn,
            length_of_stay_prediction_mimic3_fn,
            length_of_stay_prediction_mimic4_fn,
            mortality_prediction_mimic3_fn,
            mortality_prediction_mimic4_fn,
            readmission_prediction_mimic3_fn,
            readmission_prediction_mimic4_fn,
        )

    task_functions = {
        ("mimic3", "drugrec"): drug_recommendation_mimic3_fn,
        ("mimic4", "drugrec"): drug_recommendation_mimic4_fn,
        ("mimic3", "mortality"): mortality_prediction_mimic3_fn,
        ("mimic4", "mortality"): mortality_prediction_mimic4_fn,
        ("mimic3", "readmission"): readmission_prediction_mimic3_fn,
        ("mimic4", "readmission"): readmission_prediction_mimic4_fn,
        ("mimic3", "lenofstay"): length_of_stay_prediction_mimic3_fn,
        ("mimic4", "lenofstay"): length_of_stay_prediction_mimic4_fn,
    }
    return ds.set_task(task_functions[(dataset, task)])


def expand_and_map(values, mapping):
    if not values:
        return []
    flat = [item for group in values for item in group] if isinstance(values[0], list) else values
    return [mapping.get(str(item), str(item)) for item in flat]


def process_dataset(sample_dataset, condition_dict, procedure_dict, drug_dict):
    patient_data = defaultdict(dict)
    for patient, indices in sample_dataset.patient_to_index.items():
        for history_index, sample_index in enumerate(indices):
            patient_key = f"{patient}_{history_index}"
            patient_data[patient_key]["label"] = sample_dataset.samples[sample_index].get("label")
            for visit_index, previous_index in enumerate(indices[: history_index + 1]):
                sample = sample_dataset.samples[previous_index]
                patient_data[patient_key][f"visit {visit_index}"] = {
                    "conditions": expand_and_map(sample["conditions"], condition_dict),
                    "procedures": expand_and_map(sample["procedures"], procedure_dict),
                    "drugs": expand_and_map(sample["drugs"], drug_dict),
                }
    return dict(patient_data)


def balanced_subset(patient_data: dict, max_samples: int, seed: int) -> dict:
    if max_samples <= 0 or len(patient_data) <= max_samples:
        return patient_data
    keys = list(patient_data)
    random.Random(seed).shuffle(keys)
    target_positive = max_samples // 2
    selected = []
    selected_set = set()
    for key in keys:
        patient_id = key.rsplit("_", 1)[0]
        if patient_data[key]["label"] == 1 and patient_id not in selected_set:
            selected.append(key)
            selected_set.add(patient_id)
            if len(selected) >= target_positive:
                break
    for key in keys:
        if len(selected) >= max_samples:
            break
        patient_id = key.rsplit("_", 1)[0]
        if patient_data[key]["label"] == 0 and patient_id not in selected_set:
            selected.append(key)
            selected_set.add(patient_id)
    # Extremely imbalanced/dev cohorts may not contain enough of one class.
    for key in keys:
        if len(selected) >= max_samples:
            break
        patient_id = key.rsplit("_", 1)[0]
        if patient_id not in selected_set:
            selected.append(key)
            selected_set.add(patient_id)
    return {key: patient_data[key] for key in selected}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mimic3", "mimic4"), required=True)
    parser.add_argument("--task", choices=("mortality", "readmission", "drugrec", "lenofstay"), required=True)
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/ehr_data"))
    parser.add_argument("--dev", action="store_true", help="Use PyHealth's small development subset")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0, help="Balanced aggregate cap; 0 keeps all samples")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = args.mimic_root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"MIMIC directory does not exist: {root}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset} from {root}")
    ds = load_dataset(args.dataset, root, dev=args.dev, refresh_cache=args.refresh_cache)
    sample_dataset = assign_task(args.dataset, ds, args.task)
    print(f"Task samples before aggregate selection: {len(sample_dataset)}")

    mappings = load_mappings()
    patient_data = process_dataset(sample_dataset, *mappings)
    patient_data = balanced_subset(patient_data, args.max_samples, args.seed)

    pkl_path = output_dir / f"{args.dataset}_{args.task}.pkl"
    json_path = output_dir / f"pateint_{args.dataset}_{args.task}.json"
    with pkl_path.open("wb") as file:
        pickle.dump(sample_dataset, file)
    with json_path.open("w") as file:
        json.dump(patient_data, file, indent=2)
    print(f"Wrote {pkl_path}")
    print(f"Wrote {json_path} ({len(patient_data)} aggregate samples)")


if __name__ == "__main__":
    main()
