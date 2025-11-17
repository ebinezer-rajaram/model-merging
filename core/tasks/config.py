"""Base configuration utilities for all tasks."""

from pathlib import Path
from typing import Dict


class BaseTaskConfig:
    """Base class for task configurations.

    Usage:
        class MyTaskConfig(BaseTaskConfig):
            TASK_NAME = "my_task"
            DEFAULT_CONFIG_FILE = "my_task.yaml"
    """

    TASK_NAME: str = None
    DEFAULT_CONFIG_FILE: str = None

    @classmethod
    def get_config_path(cls, package_root: Path, config_name: str | None = None) -> Path:
        """Resolve configuration path for the task."""
        if cls.TASK_NAME is None:
            raise NotImplementedError("TASK_NAME must be set in subclass")
        config_dir = package_root / "configs"
        filename = config_name or cls.DEFAULT_CONFIG_FILE
        return config_dir / filename

    @classmethod
    def get_artifact_directories(cls, package_root: Path) -> Dict[str, Path]:
        """Return canonical artifact directories for the task."""
        if cls.TASK_NAME is None:
            raise NotImplementedError("TASK_NAME must be set in subclass")
        base = package_root / "artifacts" / cls.TASK_NAME
        return {
            "base": base,
            "adapters": base / "adapters",
            "vectors": base / "vectors",
            "metrics": base / "metrics",
            "models": base / "models",
            "datasets": base / "datasets",
        }


def create_simple_task_config(task_name: str, default_config_file: str):
    """Factory function to create simple task config helpers.

    Returns a tuple of (TASK_NAME, DEFAULT_CONFIG_FILE, get_config_path, get_artifact_directories)
    that can be directly exported from a task's config.py module.

    Example:
        # In tasks/my_task/config.py:
        from tasks.base.config import create_simple_task_config

        TASK_NAME, DEFAULT_CONFIG_FILE, get_config_path, get_artifact_directories = (
            create_simple_task_config("my_task", "my_task.yaml")
        )

        __all__ = ["TASK_NAME", "DEFAULT_CONFIG_FILE", "get_config_path", "get_artifact_directories"]
    """

    class _TaskConfig(BaseTaskConfig):
        TASK_NAME = task_name
        DEFAULT_CONFIG_FILE = default_config_file

    return (
        _TaskConfig.TASK_NAME,
        _TaskConfig.DEFAULT_CONFIG_FILE,
        _TaskConfig.get_config_path,
        _TaskConfig.get_artifact_directories,
    )


__all__ = ["BaseTaskConfig", "create_simple_task_config"]
