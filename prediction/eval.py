"""Evaluate KARE prediction JSON without author-specific filesystem paths."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


LABEL_RE = re.compile(r"(?:#\s*Prediction\s*#\s*)?\b([01])\b", re.IGNORECASE | re.DOTALL)


def extract_label(value) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    text = str(value).strip()
    if text in {"0", "1"}:
        return int(text)
    marker = re.split(r"#\s*Prediction\s*#", text, flags=re.IGNORECASE)
    search_text = marker[-1] if len(marker) > 1 else text
    match = LABEL_RE.search(search_text)
    return int(match.group(1)) if match else None


def evaluate_records(records: dict | list, missing_policy: str = "error") -> dict:
    iterable = records.values() if isinstance(records, dict) else records
    y_true, y_pred = [], []
    missing = 0
    for index, record in enumerate(iterable):
        truth = extract_label(record.get("ground_truth", record.get("label", record.get("output"))))
        prediction = extract_label(
            record.get("reasoning_and_prediction", record.get("prediction", record.get("pred")))
        )
        if truth is None:
            raise ValueError(f"Could not parse ground-truth label at record {index}")
        if prediction is None:
            missing += 1
            if missing_policy == "error":
                raise ValueError(f"Could not parse prediction at record {index}")
            if missing_policy == "skip":
                continue
            prediction = 0
        y_true.append(truth)
        y_pred.append(prediction)
    if not y_true:
        raise ValueError("No evaluable records")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {
        "samples": len(y_true),
        "missing_predictions": missing,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_positive": precision_score(y_true, y_pred, zero_division=0),
        "recall_positive": recall_score(y_true, y_pred, zero_division=0),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="Prediction JSON file")
    parser.add_argument("--missing-policy", choices=("error", "skip", "zero"), default="error")
    parser.add_argument("--output", type=Path, help="Optional metrics JSON output")
    args = parser.parse_args(argv)
    with args.result.open() as file:
        records = json.load(file)
    metrics = evaluate_records(records, missing_policy=args.missing_policy)
    rendered = json.dumps(metrics, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
