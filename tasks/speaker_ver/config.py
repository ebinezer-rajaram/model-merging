"""Configuration helper for speaker verification task."""

from core.tasks.config import create_simple_task_config

TASK_NAME, DEFAULT_CONFIG_FILE, get_config_path, get_artifact_directories = (
    create_simple_task_config("speaker_ver", "speaker_ver.yaml")
)
