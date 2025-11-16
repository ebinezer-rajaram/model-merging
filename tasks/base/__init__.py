"""Base utilities for speech tasks.

This module provides shared functionality across all task implementations:
- Configuration utilities
- Dataset loading helpers
- Data collators
- Metric computation utilities

Adding a new task is now much simpler - just inherit from base classes
and customize the task-specific behavior.
"""

from .collator import BaseAudioTextCollator, BaseClassificationCollator, BaseGenerationCollator
from .config import BaseTaskConfig, create_simple_task_config
from .dataset import (
    DEFAULT_AUDIO_COLUMN,
    FALLBACK_AUDIO_COLUMNS,
    add_duration_to_dataset,
    apply_split_percentages,
    assemble_splits,
    cache_and_sample_splits,
    prepare_classification_dataset,
)
from .metrics import compute_classification_metrics

__all__ = [
    # Config
    "BaseTaskConfig",
    "create_simple_task_config",
    # Dataset
    "DEFAULT_AUDIO_COLUMN",
    "FALLBACK_AUDIO_COLUMNS",
    "add_duration_to_dataset",
    "apply_split_percentages",
    "assemble_splits",
    "cache_and_sample_splits",
    "prepare_classification_dataset",
    # Collators
    "BaseAudioTextCollator",
    "BaseClassificationCollator",
    "BaseGenerationCollator",
    # Metrics
    "compute_classification_metrics",
]
