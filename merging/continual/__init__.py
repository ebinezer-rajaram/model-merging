"""Continual merge utilities for reusable compressed artifacts."""

from merging.continual.engine import (
    ContinualMergeResult,
    continual_merge_sources_to_artifact,
    materialize_existing_merge_to_artifact,
)
from merging.continual.policy import ContinualMergePolicy

__all__ = [
    "ContinualMergePolicy",
    "ContinualMergeResult",
    "continual_merge_sources_to_artifact",
    "materialize_existing_merge_to_artifact",
]
