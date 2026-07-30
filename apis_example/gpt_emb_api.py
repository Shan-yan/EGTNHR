"""Compatibility imports; use :mod:`apis.gpt_emb_api` in new code."""

from apis.gpt_emb_api import generate_embeddings, get_embedding

__all__ = ["generate_embeddings", "get_embedding"]
