"""Task registry for centralized task configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class TaskInfo:
    """Information about a registered task."""

    name: str
    default_config_file: str
    artifact_subdirs: List[str]
    default_adapter_subdir: str

    def get_artifact_dirs(self, base_dir: Path) -> dict[str, Path]:
        """Get dictionary of artifact directories for this task."""
        return {
            subdir: base_dir / subdir
            for subdir in self.artifact_subdirs
        }


# Centralized task registry
TASK_REGISTRY = {
    "asr": TaskInfo(
        name="asr",
        default_config_file="asr.yaml",
        artifact_subdirs=["adapters", "vectors", "metrics", "models", "datasets"],
        default_adapter_subdir="qwen2_5_omni_lora_asr_10h",
    ),
    "emotion": TaskInfo(
        name="emotion",
        default_config_file="emotion.yaml",
        artifact_subdirs=["adapters", "vectors", "metrics", "models", "datasets"],
        default_adapter_subdir="qwen2_5_omni_lora_emotion",
    ),
    "intent": TaskInfo(
        name="intent",
        default_config_file="intent.yaml",
        artifact_subdirs=["adapters", "vectors", "metrics", "models", "datasets"],
        default_adapter_subdir="qwen2_5_omni_lora_intent",
    ),
    "speaker_id": TaskInfo(
        name="speaker_id",
        default_config_file="speaker_id.yaml",
        artifact_subdirs=["adapters", "vectors", "metrics", "models", "datasets"],
        default_adapter_subdir="qwen2_5_omni_lora_speaker_id",
    ),
    "speech_qa": TaskInfo(
        name="speech_qa",
        default_config_file="speech_qa.yaml",
        artifact_subdirs=["adapters", "vectors", "metrics", "models", "datasets"],
        default_adapter_subdir="qwen2_5_omni_lora_speech_qa",
    ),
    "kws": TaskInfo(
        name="kws",
        default_config_file="kws.yaml",
        artifact_subdirs=["adapters", "vectors", "metrics", "models", "datasets"],
        default_adapter_subdir="qwen2_5_omni_lora_kws",
    ),
}


def get_task_info(task_name: str) -> TaskInfo:
    """
    Get task information from the registry.

    Args:
        task_name: Name of the task

    Returns:
        TaskInfo object

    Raises:
        ValueError: If task_name is not in registry
    """
    if task_name not in TASK_REGISTRY:
        valid_tasks = ", ".join(TASK_REGISTRY.keys())
        raise ValueError(
            f"Unknown task '{task_name}'. Valid tasks are: {valid_tasks}"
        )
    return TASK_REGISTRY[task_name]


def list_tasks() -> List[str]:
    """Get list of all registered task names."""
    return list(TASK_REGISTRY.keys())
