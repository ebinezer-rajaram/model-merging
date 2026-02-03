"""Register built-in merge methods."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch

from experiments.extract_vector import extract_task_vector_from_lora
from merging.techniques.task_vector import merge_uniform_via_task_vectors
from merging.core.registry import MergeMethod, MergeOutput, build_merge_metadata, register_merge_method
from merging.techniques.uniform import merge_adapters_uniform
from merging.core.utils import compute_delta_from_lora_weights, load_adapter_weights
from merging.techniques.weighted import merge_adapters_weighted, merge_task_vectors_weighted


def _uniform_in_memory(
    *,
    adapter_paths: List[Path],
    source_metadata: List[Dict],
    merge_mode: str,
    lambda_weight: Optional[float],
) -> MergeOutput:
    all_weights = [load_adapter_weights(p) for p in adapter_paths]
    merged_weights = merge_adapters_uniform(all_weights, merge_mode=merge_mode)
    merged_delta = compute_delta_from_lora_weights(merged_weights, adapter_paths[0])
    metadata = build_merge_metadata(
        method="uniform",
        merge_mode=merge_mode,
        num_adapters=len(adapter_paths),
        source_metadata=source_metadata,
        num_parameters=len(merged_delta),
    )
    return MergeOutput(merged_delta=merged_delta, merged_weights=merged_weights, metadata=metadata)


def _weighted_in_memory(
    *,
    adapter_paths: List[Path],
    source_metadata: List[Dict],
    merge_mode: str,
    lambda_weight: Optional[float],
) -> MergeOutput:
    if lambda_weight is None:
        raise ValueError("weighted requires lambda_weight.")
    weights1 = load_adapter_weights(adapter_paths[0])
    weights2 = load_adapter_weights(adapter_paths[1])
    merged_weights = merge_adapters_weighted(
        weights1,
        weights2,
        lambda_weight=lambda_weight,
        merge_mode=merge_mode,
    )
    merged_delta = compute_delta_from_lora_weights(merged_weights, adapter_paths[0])
    weighted_metadata = [dict(source_metadata[0]), dict(source_metadata[1])]
    weighted_metadata[0]["weight"] = lambda_weight
    weighted_metadata[1]["weight"] = 1.0 - lambda_weight
    metadata = build_merge_metadata(
        method="weighted",
        merge_mode=merge_mode,
        num_adapters=len(adapter_paths),
        source_metadata=weighted_metadata,
        num_parameters=len(merged_delta),
        lambda_weight=lambda_weight,
    )
    return MergeOutput(merged_delta=merged_delta, merged_weights=merged_weights, metadata=metadata)


def _task_vector_in_memory(
    *,
    adapter_paths: List[Path],
    source_metadata: List[Dict],
    merge_mode: str,
    lambda_weight: Optional[float],
) -> MergeOutput:
    task_vectors = [extract_task_vector_from_lora(p) for p in adapter_paths]
    merged_delta = merge_adapters_uniform(task_vectors, merge_mode=merge_mode)
    metadata = build_merge_metadata(
        method="task_vector",
        merge_mode=merge_mode,
        num_adapters=len(adapter_paths),
        source_metadata=source_metadata,
        num_parameters=len(merged_delta),
    )
    return MergeOutput(merged_delta=merged_delta, merged_weights=None, metadata=metadata)


def _weighted_delta_in_memory(
    *,
    adapter_paths: List[Path],
    source_metadata: List[Dict],
    merge_mode: str,
    lambda_weight: Optional[float],
) -> MergeOutput:
    if lambda_weight is None:
        raise ValueError("weighted_delta requires lambda_weight.")
    tv1 = extract_task_vector_from_lora(adapter_paths[0])
    tv2 = extract_task_vector_from_lora(adapter_paths[1])
    merged_delta = merge_task_vectors_weighted(
        tv1,
        tv2,
        lambda_weight=lambda_weight,
        merge_mode=merge_mode,
    )
    weighted_metadata = [dict(source_metadata[0]), dict(source_metadata[1])]
    weighted_metadata[0]["weight"] = lambda_weight
    weighted_metadata[1]["weight"] = 1.0 - lambda_weight
    metadata = build_merge_metadata(
        method="weighted_delta",
        merge_mode=merge_mode,
        num_adapters=len(adapter_paths),
        source_metadata=weighted_metadata,
        num_parameters=len(merged_delta),
        lambda_weight=lambda_weight,
    )
    return MergeOutput(merged_delta=merged_delta, merged_weights=None, metadata=metadata)


def _task_vector_save(
    *,
    adapter_paths: List[Path],
    output_path: Path,
    merge_mode: str,
    show_progress: bool,
) -> Path:
    merge_uniform_via_task_vectors(
        adapter_paths=adapter_paths,
        output_path=output_path,
        merge_mode=merge_mode,
        show_progress=show_progress,
    )
    return output_path


def register_builtin_methods() -> None:
    register_merge_method(
        MergeMethod(
            name="uniform",
            requires_lambda=False,
            min_adapters=2,
            max_adapters=None,
            saveable=True,
            merge_in_memory=_uniform_in_memory,
        )
    )
    register_merge_method(
        MergeMethod(
            name="weighted",
            requires_lambda=True,
            min_adapters=2,
            max_adapters=2,
            saveable=True,
            merge_in_memory=_weighted_in_memory,
        )
    )
    register_merge_method(
        MergeMethod(
            name="task_vector",
            requires_lambda=False,
            min_adapters=2,
            max_adapters=None,
            saveable=True,
            merge_in_memory=_task_vector_in_memory,
            save_fn=_task_vector_save,
        )
    )
    register_merge_method(
        MergeMethod(
            name="weighted_delta",
            requires_lambda=True,
            min_adapters=2,
            max_adapters=2,
            saveable=False,
            merge_in_memory=_weighted_delta_in_memory,
        )
    )


# Register on import
register_builtin_methods()
