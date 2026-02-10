"""Adapter merging methods and utilities.

This package provides various methods for merging LoRA adapters trained on different tasks.
Each merging method is implemented in its own module for extensibility.

Available Methods:
    - uniform: Simple averaging of adapter weights
    - uniform_delta: Simple averaging in task-vector (delta) space (in-memory only)
    - weighted: Weighted averaging with configurable lambda parameter
    - task_vector: Merge in full weight space via task vectors
    - weighted_delta: Weighted merge in task-vector (delta) space
    - ties: Paper-core TIES merge in task-vector (delta) space (in-memory only)

Usage:
    from merging import merge_uniform, merge_weighted
    from merging.runtime.utils import resolve_best_adapter

    # Resolve best adapters for tasks
    adapter1, meta1 = resolve_best_adapter("asr")
    adapter2, meta2 = resolve_best_adapter("emotion")

    # Merge using uniform averaging
    merged_path = merge_uniform([adapter1, adapter2], output_path)

    # Merge using weighted averaging
    merged_path = merge_weighted(adapter1, adapter2, lambda_weight=0.7, output_path=output_path)
"""

from merging.methods.uniform import merge_uniform
from merging.methods.weighted import merge_weighted
from merging.engine.registry import get_merge_method, list_merge_methods
from merging.engine import builtin_methods as _methods  # registers built-ins
from merging.evaluation.evaluate import evaluate_merged_adapter
from merging.runtime.utils import (
    create_merge_output_path,
    load_adapter_weights,
    resolve_best_adapter,
    save_merged_adapter,
)

__all__ = [
    "merge_uniform",
    "merge_weighted",
    "create_merge_output_path",
    "evaluate_merged_adapter",
    "load_adapter_weights",
    "resolve_best_adapter",
    "save_merged_adapter",
    "get_merge_method",
    "list_merge_methods",
]
