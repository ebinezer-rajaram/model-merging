"""Configuration helpers for the keyword spotting task."""

from core.tasks.config import create_simple_task_config

# Create configuration using the factory function
TASK_NAME, DEFAULT_CONFIG_FILE, get_config_path, get_artifact_directories = (
    create_simple_task_config("kws", "kws.yaml")
)

__all__ = [
    "TASK_NAME",
    "DEFAULT_CONFIG_FILE",
    "get_config_path",
    "get_artifact_directories",
]
