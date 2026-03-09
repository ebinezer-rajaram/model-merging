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
from .multitask_schema import (
    MultiTaskArtifactsConfig,
    MultiTaskConfig,
    MultiTaskLoggingConfig,
    MultiTaskMetricsConfig,
    MultiTaskSamplingConfig,
    MultiTaskSpec,
    MultiTaskTrainingConfig,
    parse_multitask_config,
)
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
    "MultiTaskConfig",
    "MultiTaskArtifactsConfig",
    "MultiTaskLoggingConfig",
    "MultiTaskMetricsConfig",
    "MultiTaskSamplingConfig",
    "MultiTaskSpec",
    "MultiTaskTrainingConfig",
    "parse_multitask_config",
    "TASK_REGISTRY",
    "get_task_info",
]
