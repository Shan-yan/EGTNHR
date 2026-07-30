"""Embed deterministic patient contexts with the configured OpenAI account."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import pickle
from pathlib import Path

from tqdm import tqdm

from apis.gpt_emb_api import get_embedding


def embed_contexts(data: dict, workers: int, max_chars: int) -> dict:
    def retrieve(item):
        patient_id, text = item
        return patient_id, get_embedding(str(text)[:max_chars])

    embeddings = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(retrieve, item): item[0] for item in data.items()}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Patients"):
            patient_id, embedding = future.result()
            embeddings[patient_id] = embedding
    return embeddings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=30_000)
    args = parser.parse_args(argv)
    with args.input.open() as file:
        contexts = json.load(file)
    embeddings = embed_contexts(contexts, args.workers, args.max_chars)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        pickle.dump(embeddings, file)
    print(f"Wrote {args.output} ({len(embeddings)} embeddings)")


if __name__ == "__main__":
    main()
