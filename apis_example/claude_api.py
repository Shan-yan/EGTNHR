"""Compatibility imports; use :mod:`apis.claude_api` in new code."""

from apis.claude_api import BEDROCK_MODEL_NAME_MAP, get_claude_response

__all__ = ["BEDROCK_MODEL_NAME_MAP", "get_claude_response"]
