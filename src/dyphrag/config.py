"""Offline YAML configuration composition with Hydra-style dot-list overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"


class ConfigurationError(ValueError):
    pass


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"Configuration does not exist: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ConfigurationError(f"Top-level YAML must be a mapping: {path}")
    return value


def deep_merge(base: MutableMapping[str, Any], update: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _parse_scalar(value: str) -> Any:
    parsed = yaml.safe_load(value)
    return parsed


def _set_dot(config: MutableMapping[str, Any], path: str, value: Any) -> None:
    cursor = config
    parts = path.split(".")
    for part in parts[:-1]:
        node = cursor.setdefault(part, {})
        if not isinstance(node, MutableMapping):
            raise ConfigurationError(f"Cannot override nested value below {part}")
        cursor = node
    cursor[parts[-1]] = value


def load_config(overrides: Iterable[str], config_root: Path = CONFIG_ROOT) -> dict[str, Any]:
    config = _load_yaml(config_root / "config.yaml")
    normal: list[tuple[str, str]] = []
    groups = {"experiment": config.get("experiment", "full_dyphrag"), "dataset": config.get("dataset", "synthetic")}
    for token in overrides:
        if "=" not in token:
            raise ConfigurationError(f"Expected key=value override, got {token!r}")
        key, value = token.split("=", 1)
        if key in groups:
            groups[key] = value
        else:
            normal.append((key, value))
    for group, name in groups.items():
        fragment = _load_yaml(config_root / group / f"{name}.yaml")
        deep_merge(config, fragment)
        config[group] = name
    for key, value in normal:
        _set_dot(config, key, _parse_scalar(value))
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    from .baselines import capabilities_for

    capabilities_for(str(config["model"]["mode"]))
    if config.get("task") != "disease_pair_binary":
        raise ConfigurationError("DyPH-RAG currently supports task=disease_pair_binary")
    hidden = int(config["model"]["hidden_dim"])
    if hidden <= 0 or hidden % 4:
        raise ConfigurationError("model.hidden_dim must be positive and divisible by four")
    if int(config["retrieval"]["max_iterations"]) < 1:
        raise ConfigurationError("retrieval.max_iterations must be at least one")
    if int(config["retrieval"]["max_total_k"]) < 0:
        raise ConfigurationError("retrieval.max_total_k cannot be negative")
    if not 0.0 <= float(config["evidence"]["abstention_threshold"]) <= 1.0:
        raise ConfigurationError("evidence.abstention_threshold must be in [0, 1]")
    for key in (
        "stop_threshold",
        "evidence_sufficiency_threshold",
        "source_gate_threshold",
        "retrieval_gate_threshold",
    ):
        if not 0.0 <= float(config["router"][key]) <= 1.0:
            raise ConfigurationError(f"router.{key} must be in [0, 1]")
    if not 0.0 <= float(config["evaluation"]["minimum_selective_coverage"]) <= 1.0:
        raise ConfigurationError("evaluation.minimum_selective_coverage must be in [0, 1]")


def save_resolved(config: Mapping[str, Any], path: Path) -> None:
    path.write_text(yaml.safe_dump(dict(config), sort_keys=True), encoding="utf-8")
