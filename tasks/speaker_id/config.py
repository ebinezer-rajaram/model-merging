"""Configuration helpers for the speaker identification task."""

from pathlib import Path
from typing import Dict

TASK_NAME = "speaker_id"
DEFAULT_CONFIG_FILE = "speaker_id.yaml"


def get_config_path(package_root: Path, config_name: str | None = None) -> Path:
    """Resolve configuration path for the speaker identification task."""
    config_dir = package_root / "configs"
    filename = config_name or DEFAULT_CONFIG_FILE
    return config_dir / filename


def get_artifact_directories(package_root: Path) -> Dict[str, Path]:
    """Return canonical artifact directories for the speaker identification task."""
    base = package_root / "artifacts" / TASK_NAME
    return {
        "base": base,
        "adapters": base / "adapters",
        "vectors": base / "vectors",
        "metrics": base / "metrics",
        "models": base / "models",
        "datasets": base / "datasets",
    }


__all__ = [
    "TASK_NAME",
    "DEFAULT_CONFIG_FILE",
    "get_config_path",
    "get_artifact_directories",
]
