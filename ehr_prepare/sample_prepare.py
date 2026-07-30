"""Split a prepared KARE cohort by patient and export JSON sample lists."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mimic3", "mimic4"), required=True)
    parser.add_argument("--task", choices=("mortality", "readmission", "drugrec", "lenofstay"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/ehr_data"))
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=528)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if abs(sum(args.ratios) - 1.0) > 1e-9:
        raise SystemExit("--ratios must sum to 1.0")
    data_dir = args.data_dir.expanduser().resolve()
    dataset_path = data_dir / f"{args.dataset}_{args.task}.pkl"
    aggregate_path = data_dir / f"pateint_{args.dataset}_{args.task}.json"
    for path in (dataset_path, aggregate_path):
        if not path.is_file():
            raise SystemExit(f"Missing input: {path}")

    try:
        from .spliter import split_by_patient
    except ImportError:
        from spliter import split_by_patient  # type: ignore

    with dataset_path.open("rb") as file:
        sample_dataset = pickle.load(file)
    with aggregate_path.open() as file:
        aggregate_samples = json.load(file)
    split_datasets = split_by_patient(sample_dataset, args.ratios, seed=args.seed)
    split_patient_ids = [
        {sample["patient_id"] for sample in split_dataset}
        for split_dataset in split_datasets
    ]

    outputs = [[], [], []]
    for aggregate_id, visits in aggregate_samples.items():
        patient_id, visit_id = aggregate_id.rsplit("_", 1)
        sample = {
            "visit_id": visit_id,
            "patient_id": patient_id,
            "conditions": [],
            "procedures": [],
            "drugs": [],
            "label": visits["label"],
        }
        visit_keys = sorted(
            (key for key in visits if key.startswith("visit ")),
            key=lambda key: int(key.split()[-1]),
        )
        for key in visit_keys:
            sample["conditions"].append(visits[key]["conditions"])
            sample["procedures"].append(visits[key]["procedures"])
            sample["drugs"].append(visits[key]["drugs"])
        for split_index, patient_ids in enumerate(split_patient_ids):
            if patient_id in patient_ids:
                outputs[split_index].append(sample)
                break

    for split_name, output in zip(("train", "val", "test"), outputs):
        path = data_dir / f"{args.dataset}_{args.task}_samples_{split_name}.json"
        with path.open("w") as file:
            json.dump(output, file, indent=2)
        print(f"Wrote {path} ({len(output)} samples)")


if __name__ == "__main__":
    main()
