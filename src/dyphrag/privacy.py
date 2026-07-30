"""Privacy-safe logging helpers. Raw EHR text and identifiers never leave memory."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Mapping


SENSITIVE_KEYS = {
    "patient_id",
    "subject_id",
    "hadm_id",
    "visit_id",
    "text",
    "prompt",
    "note",
    "name",
    "dob",
}


def anonymous_ref(value: str, run_salt: str) -> str:
    return hashlib.sha256(f"{run_salt}:{value}".encode()).hexdigest()[:16]


def sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if str(key).lower() in SENSITIVE_KEYS else sanitize(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "fields"):
            payload["fields"] = sanitize(record.fields)
        return json.dumps(payload, sort_keys=True, default=str)


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)

