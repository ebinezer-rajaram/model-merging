"""Centralized configuration management for speech merging tasks."""

from .schemas import (
    LoraConfig,
    ModelConfig,
    DatasetConfig,
    TrainingConfig,
    EvaluationConfig,
    ArtifactsConfig,
    MetricsConfig,
    TaskConfig,
)
from .loader import load_task_config, get_config_path
from .registry import TASK_REGISTRY, get_task_info

__all__ = [
    "LoraConfig",
    "ModelConfig",
    "DatasetConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "ArtifactsConfig",
    "MetricsConfig",
    "TaskConfig",
    "load_task_config",
    "get_config_path",
    "TASK_REGISTRY",
    "get_task_info",
]
