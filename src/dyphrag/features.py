"""Deterministic local feature functions; no model download or network access."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable

import torch


def hashed_vector(tokens: Iterable[str], dimension: int, device: torch.device | None = None) -> torch.Tensor:
    vector = torch.zeros(dimension, dtype=torch.float32, device=device)
    for token in tokens:
        digest = hashlib.sha256(token.lower().encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dimension
        sign = 1.0 if digest[4] % 2 else -1.0
        vector[index] += sign
    norm = vector.norm(p=2)
    return vector / norm if norm > 0 else vector


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = float(left.norm() * right.norm())
    return float(torch.dot(left, right) / denominator) if denominator else 0.0


def temporal_decay(hours: float, half_life_hours: float) -> float:
    return math.exp(-math.log(2.0) * max(hours, 0.0) / max(half_life_hours, 1e-6))

