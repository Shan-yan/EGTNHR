#!/usr/bin/env python3
"""Offline smoke test for the files and deterministic code shipped with KARE."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prediction.eval import evaluate_records


def main() -> int:
    kg_path = ROOT / "graph/kg_raw.txt"
    valid_triples = 0
    with kg_path.open() as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            if len(line.rstrip("\n").split("\t")) != 3:
                raise AssertionError(f"Malformed KG triple at line {line_number}")
            valid_triples += 1
    if valid_triples < 1000:
        raise AssertionError("Included knowledge graph is unexpectedly small")

    summary_path = ROOT / "data_examples/community_summary_examples/example_community_summary.json"
    summaries = json.loads(summary_path.read_text())
    if len(summaries) != 10_000 or not all(isinstance(item, dict) for item in summaries):
        raise AssertionError("Community-summary example schema/count changed")

    retrieval_counts = {}
    for task in ("mortality", "readmission"):
        path = ROOT / f"data_examples/retrieved_knowledge_examples/retrieved_knowledge_example_{task}.json"
        retrieval = json.loads(path.read_text())
        if len(retrieval) != 2_000 or not all(isinstance(value, list) for value in retrieval.values()):
            raise AssertionError(f"Invalid {task} retrieval example")
        retrieval_counts[task] = len(retrieval)

    synthetic = {
        "a": {"ground_truth": 0, "reasoning_and_prediction": "# Prediction #\n0"},
        "b": {"ground_truth": 1, "reasoning_and_prediction": "# Prediction #\n1"},
        "c": {"ground_truth": "1", "prediction": "0"},
        "d": {"ground_truth": "# Prediction #\n0", "prediction": "1"},
    }
    metrics = evaluate_records(synthetic)
    if metrics["accuracy"] != 0.5 or metrics["confusion_matrix"] != [[1, 1], [1, 1]]:
        raise AssertionError("Evaluation smoke fixture produced unexpected metrics")

    print(f"PASS: {valid_triples:,} KG triples")
    print(f"PASS: {len(summaries):,} community summaries")
    print(f"PASS: retrieval examples {retrieval_counts}")
    print("PASS: evaluation parser and metrics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
