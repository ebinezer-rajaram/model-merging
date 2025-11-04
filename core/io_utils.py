"""Utility helpers for configuration and filesystem tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def ensure_dir(path: Path) -> Path:
    """Create directory if absent and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionaries, overriding base with override values."""
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML configuration from disk."""
    with path.open("r") as handle:
        return yaml.safe_load(handle) or {}


def load_config(config_path: Path, base_config_path: Path | None = None) -> Dict[str, Any]:
    """Load task configuration merged with base defaults."""
    config_path = config_path.resolve()
    config_data = load_yaml(config_path)

    if base_config_path is None:
        base_config_path = config_path.parent / "base.yaml"

    if base_config_path.exists():
        base_data = load_yaml(base_config_path)
        return _deep_merge(base_data, config_data)

    return config_data


def dump_json(data: Dict[str, Any], path: Path) -> None:
    """Write JSON data to disk with indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
