"""Configuration helpers for the emotion recognition task."""

from tasks.base.config import create_simple_task_config

# Create configuration using the factory function - eliminates all boilerplate
TASK_NAME, DEFAULT_CONFIG_FILE, get_config_path, get_artifact_directories = (
    create_simple_task_config("emotion", "emotion.yaml")
)

__all__ = [
    "TASK_NAME",
    "DEFAULT_CONFIG_FILE",
    "get_config_path",
    "get_artifact_directories",
]
