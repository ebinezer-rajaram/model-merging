"""Configuration helpers for the VocalSound classification task."""

from core.tasks.config import create_simple_task_config

TASK_NAME, DEFAULT_CONFIG_FILE, get_config_path, get_artifact_directories = (
    create_simple_task_config("vocalsound", "vocalsound.yaml")
)

__all__ = [
    "TASK_NAME",
    "DEFAULT_CONFIG_FILE",
    "get_config_path",
    "get_artifact_directories",
]
