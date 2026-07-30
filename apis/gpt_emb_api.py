"""OpenAI embedding helpers."""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example and export it first.")
    return OpenAI(api_key=api_key)


def get_embedding(text: str, model: str = "text-embedding-3-large") -> list[float]:
    text = text.replace("\n", " ")
    return _client().embeddings.create(input=[text], model=model).data[0].embedding


def generate_embeddings(texts, model: str = "text-embedding-3-large"):
    return {text: get_embedding(text, model=model) for text in texts}
