"""Unified experiment results collection package.

Public API:
    ResultsAggregator  — top-level entry point; runs all collectors.
    SingleTaskCollector, MTLCollector, MergeCollector  — individual collectors.
    ExperimentResult, ExperimentMetadata, TaskEvalResult  — schema dataclasses.
    DATAFRAME_COLUMNS, ANALYSIS_COLUMNS  — column name lists.
    to_wide_df  — pivot long-format DataFrame to wide format.
"""

from .aggregator import ANALYSIS_COLUMNS, DATAFRAME_COLUMNS, ResultsAggregator
from .base import ResultsCollector
from .merge_collector import MergeCollector
from .mtl_collector import MTLCollector
from .schema import (
    ExperimentMetadata,
    ExperimentResult,
    InterferenceDeltaInfo,
    TaskEvalResult,
    TrainingHyperparameters,
)
from .single_task_collector import SingleTaskCollector
from .utils import HELD_OUT_TASKS, KNOWN_TASKS, TASK_METRICS, to_wide_df

__all__ = [
    # Aggregator
    "ResultsAggregator",
    "DATAFRAME_COLUMNS",
    "ANALYSIS_COLUMNS",
    # Collectors
    "ResultsCollector",
    "SingleTaskCollector",
    "MTLCollector",
    "MergeCollector",
    # Schema
    "ExperimentResult",
    "ExperimentMetadata",
    "TaskEvalResult",
    "InterferenceDeltaInfo",
    "TrainingHyperparameters",
    # Utilities
    "to_wide_df",
    "KNOWN_TASKS",
    "HELD_OUT_TASKS",
    "TASK_METRICS",
]
