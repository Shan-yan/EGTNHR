"""Retrieve label-matched and label-mismatched similar patients with FAISS."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm


def retrieve(contexts: dict, patient_data: dict, embeddings: dict, top_k: int, max_context_chars: int):
    patient_ids = [patient_id for patient_id in embeddings if patient_id in contexts and patient_id in patient_data]
    if not patient_ids:
        raise ValueError("No shared patient IDs across contexts, labels, and embeddings")
    matrix = np.asarray([embeddings[patient_id] for patient_id in patient_ids], dtype="float32")
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    output = {}
    search_k = min(len(patient_ids), max(100, top_k * 20))
    for row, patient_id in enumerate(tqdm(patient_ids, desc="Similar patients")):
        _, neighbors = index.search(matrix[row : row + 1], search_k)
        same, different = [], []
        label = patient_data[patient_id]["label"]
        for neighbor_row in neighbors[0]:
            if neighbor_row < 0:
                continue
            neighbor_id = patient_ids[int(neighbor_row)]
            if neighbor_id == patient_id or neighbor_id.rsplit("_", 1)[0] == patient_id.rsplit("_", 1)[0]:
                continue
            if len(contexts[neighbor_id]) > max_context_chars:
                continue
            bucket = same if patient_data[neighbor_id]["label"] == label else different
            if len(bucket) < top_k:
                bucket.append(contexts[neighbor_id] + f"\n\nLabel:\n{patient_data[neighbor_id]['label']}\n")
            if len(same) >= top_k and len(different) >= top_k:
                break
        output[patient_id] = {"positive": same or ["None"], "negative": different or ["None"]}
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--patient-data", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-context-chars", type=int, default=20_000)
    args = parser.parse_args(argv)
    with args.contexts.open() as file:
        contexts = json.load(file)
    with args.patient_data.open() as file:
        patient_data = json.load(file)
    with args.embeddings.open("rb") as file:
        embeddings = pickle.load(file)
    output = retrieve(contexts, patient_data, embeddings, args.top_k, args.max_context_chars)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"Wrote {args.output} ({len(output)} patients)")


if __name__ == "__main__":
    main()
