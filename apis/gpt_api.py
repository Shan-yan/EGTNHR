"""OpenAI chat-completion helper."""

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


def get_gpt_response(prompt: str, model: str = "gpt-4o", seed: int = 44) -> str:
    return _client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        seed=seed,
    ).choices[0].message.content
