"""Configuration helpers for the Speech Translation task."""

from pathlib import Path
from typing import Dict, Optional

from core.tasks.config import BaseTaskConfig

TASK_NAME = "st"
DEFAULT_CONFIG_FILE = "st.yaml"


class STTaskConfig(BaseTaskConfig):
    """Speech Translation task configuration with language-specific artifact directories."""

    TASK_NAME = "st"
    DEFAULT_CONFIG_FILE = "st.yaml"

    @classmethod
    def get_artifact_directories(
        cls, package_root: Path, language: Optional[str] = None
    ) -> Dict[str, Path]:
        """Return artifact directories for ST task with optional language-specific subdirectories.

        Args:
            package_root: Root directory of the package
            language: Optional language pair (e.g., "en_de", "en_fr") to create language-specific subdirs

        Returns:
            Dictionary with paths for adapters, datasets, metrics, etc.
        """
        if language:
            # Language-specific artifact directories
            base = package_root / "artifacts" / cls.TASK_NAME / language
        else:
            # Default behavior without language
            base = package_root / "artifacts" / cls.TASK_NAME

        return {
            "base": base,
            "adapters": base / "adapters",
            "vectors": base / "vectors",
            "metrics": base / "metrics",
            "models": base / "models",
            "datasets": base / "datasets",
        }


def get_config_path(package_root: Path, config_name: str | None = None) -> Path:
    """Resolve configuration path for the ST task."""
    return STTaskConfig.get_config_path(package_root, config_name)


def get_artifact_directories(
    package_root: Path, language: Optional[str] = None
) -> Dict[str, Path]:
    """Return artifact directories for ST task with optional language-specific subdirectories."""
    return STTaskConfig.get_artifact_directories(package_root, language)


__all__ = [
    "TASK_NAME",
    "DEFAULT_CONFIG_FILE",
    "STTaskConfig",
    "get_config_path",
    "get_artifact_directories",
]
