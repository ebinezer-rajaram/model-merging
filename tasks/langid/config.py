"""Configuration helpers for the language identification task."""

from core.tasks.config import create_simple_task_config

# Create configuration using the factory function
TASK_NAME, DEFAULT_CONFIG_FILE, get_config_path, get_artifact_directories = (
    create_simple_task_config("langid", "langid.yaml")
)

__all__ = [
    "TASK_NAME",
    "DEFAULT_CONFIG_FILE",
    "get_config_path",
    "get_artifact_directories",
]
