"""Configuration helpers for the emotion recognition task."""

from pathlib import Path
from typing import Dict

TASK_NAME = "emotion"
DEFAULT_CONFIG_FILE = "emotion.yaml"


def get_config_path(package_root: Path, config_name: str | None = None) -> Path:
    """Resolve the config file path for the emotion task."""
    config_dir = package_root / "configs"
    filename = config_name or DEFAULT_CONFIG_FILE
    return config_dir / filename


def get_artifact_directories(package_root: Path) -> Dict[str, Path]:
    """Return standard artifact directories for the emotion task."""
    base = package_root / "artifacts" / TASK_NAME
    return {
        "base": base,
        "adapters": base / "adapters",
        "vectors": base / "vectors",
        "metrics": base / "metrics",
        "models": base / "models",
        "datasets": base / "datasets",
    }
