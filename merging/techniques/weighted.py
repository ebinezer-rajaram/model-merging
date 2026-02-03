"""Weighted adapter merging method with configurable lambda parameter.

This module implements weighted averaging of two LoRA adapters:
    W_merged = λ * W_1 + (1 - λ) * W_2

where λ (lambda) is a weight parameter between 0 and 1 that controls
the contribution of each adapter. This allows fine-grained control over
how much each task influences the merged adapter.

Special cases:
    - λ = 0.0: Returns adapter 2 only
    - λ = 0.5: Equal weighting (equivalent to uniform merge)
    - λ = 1.0: Returns adapter 1 only

Usage:
    from merging import merge_weighted
    from merging.core.utils import resolve_best_adapter, create_merge_output_path

    # Resolve adapters
    adapter1, meta1 = resolve_best_adapter("asr")
    adapter2, meta2 = resolve_best_adapter("emotion")

    # Create output path with lambda in name
    output_path = create_merge_output_path(
        "weighted",
        ["asr", "emotion"],
        {"lambda": 0.7}
    )

    # Merge with 70% ASR, 30% emotion
    merged_path = merge_weighted(
        adapter1_path=adapter1,
        adapter2_path=adapter2,
        lambda_weight=0.7,
        output_path=output_path,
        source_metadata=[meta1, meta2],
    )
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch

from merging.core.utils import load_adapter_weights, save_merged_adapter


def merge_adapters_weighted(
    adapter1_weights: Dict[str, torch.Tensor],
    adapter2_weights: Dict[str, torch.Tensor],
    lambda_weight: float,
    merge_mode: str = "common",
) -> Dict[str, torch.Tensor]:
    """Merge two adapters using weighted averaging.

    Computes: W_merged = λ * W_1 + (1 - λ) * W_2

    Args:
        adapter1_weights: First adapter weight dictionary
        adapter2_weights: Second adapter weight dictionary
        lambda_weight: Weight for first adapter (0.0 to 1.0)
        merge_mode: How to handle different parameters:
            - "common": Only merge parameters present in BOTH adapters (default)
            - "strict": Require both adapters to have identical parameters

    Returns:
        Merged adapter weights

    Raises:
        ValueError: If lambda not in [0, 1] or strict mode has mismatched params
    """
    # Validate lambda
    if not 0.0 <= lambda_weight <= 1.0:
        raise ValueError(
            f"lambda_weight must be between 0.0 and 1.0, got {lambda_weight}"
        )

    # Handle edge cases
    if lambda_weight == 0.0:
        print("⚠️  lambda=0.0: Returning second adapter unchanged")
        return adapter2_weights.copy()
    if lambda_weight == 1.0:
        print("⚠️  lambda=1.0: Returning first adapter unchanged")
        return adapter1_weights.copy()

    # Find common and unique parameters
    keys1 = set(adapter1_weights.keys())
    keys2 = set(adapter2_weights.keys())
    common_keys = keys1 & keys2
    unique_to_1 = keys1 - keys2
    unique_to_2 = keys2 - keys1

    if merge_mode == "strict":
        # Verify both adapters have the same keys
        if keys1 != keys2:
            raise ValueError(
                f"Adapters have different parameters.\n"
                f"Unique to adapter 1: {unique_to_1}\n"
                f"Unique to adapter 2: {unique_to_2}\n"
                f"Use merge_mode='common' to merge only common parameters."
            )
        keys_to_merge = keys1
    else:  # common mode
        keys_to_merge = common_keys

        if unique_to_1 or unique_to_2:
            total_unique = len(unique_to_1) + len(unique_to_2)
            print(f"⚠️  Warning: {total_unique} parameters not common to both adapters")
            print(f"   Merging {len(common_keys)} common parameters")
            print(f"   Excluding {len(unique_to_1)} from adapter 1, {len(unique_to_2)} from adapter 2")

    # Compute weighted average
    merged_weights = {}
    weight2 = 1.0 - lambda_weight

    print(f"🧮 Computing weighted average with λ={lambda_weight:.3f}")
    print(f"   Weights: {lambda_weight:.1%} adapter 1, {weight2:.1%} adapter 2")
    print(f"   Parameters to merge: {len(keys_to_merge)}")

    for key in keys_to_merge:
        # W_merged = λ * W_1 + (1 - λ) * W_2
        merged_weights[key] = (
            lambda_weight * adapter1_weights[key] +
            weight2 * adapter2_weights[key]
        )

    print(f"✅ Weighted merge complete: {len(merged_weights)} parameters merged")

    return merged_weights


def merge_task_vectors_weighted(
    task_vector1: Dict[str, torch.Tensor],
    task_vector2: Dict[str, torch.Tensor],
    lambda_weight: float,
    merge_mode: str = "common",
) -> Dict[str, torch.Tensor]:
    """Merge two task vectors (delta weights) using weighted averaging.

    Computes: delta_W_merged = λ * delta_W_1 + (1 - λ) * delta_W_2

    Args:
        task_vector1: First task vector dict (base_key -> delta_W)
        task_vector2: Second task vector dict (base_key -> delta_W)
        lambda_weight: Weight for first adapter (0.0 to 1.0)
        merge_mode: How to handle different parameters:
            - "common": Only merge parameters present in BOTH vectors (default)
            - "strict": Require identical parameter sets and shapes

    Returns:
        Merged task vector dict
    """
    if not 0.0 <= lambda_weight <= 1.0:
        raise ValueError(
            f"lambda_weight must be between 0.0 and 1.0, got {lambda_weight}"
        )

    if lambda_weight == 0.0:
        print("⚠️  lambda=0.0: Returning second task vector unchanged")
        return task_vector2.copy()
    if lambda_weight == 1.0:
        print("⚠️  lambda=1.0: Returning first task vector unchanged")
        return task_vector1.copy()

    keys1 = set(task_vector1.keys())
    keys2 = set(task_vector2.keys())
    common_keys = keys1 & keys2
    unique_to_1 = keys1 - keys2
    unique_to_2 = keys2 - keys1

    if merge_mode == "strict":
        if keys1 != keys2:
            raise ValueError(
                f"Task vectors have different parameters.\n"
                f"Unique to vector 1: {unique_to_1}\n"
                f"Unique to vector 2: {unique_to_2}\n"
                f"Use merge_mode='common' to merge only common parameters."
            )
        keys_to_merge = keys1
    else:
        keys_to_merge = common_keys
        if unique_to_1 or unique_to_2:
            total_unique = len(unique_to_1) + len(unique_to_2)
            print(f"⚠️  Warning: {total_unique} parameters not common to both vectors")
            print(f"   Merging {len(common_keys)} common parameters")
            print(f"   Excluding {len(unique_to_1)} from vector 1, {len(unique_to_2)} from vector 2")

    merged_vectors: Dict[str, torch.Tensor] = {}
    weight2 = 1.0 - lambda_weight

    print(f"🧮 Computing weighted average of task vectors with λ={lambda_weight:.3f}")
    print(f"   Weights: {lambda_weight:.1%} vector 1, {weight2:.1%} vector 2")
    print(f"   Parameters to merge: {len(keys_to_merge)}")

    for key in keys_to_merge:
        v1 = task_vector1[key]
        v2 = task_vector2[key]
        if v1.shape != v2.shape:
            message = f"Shape mismatch for {key}: {v1.shape} vs {v2.shape}"
            if merge_mode == "strict":
                raise ValueError(message)
            print(f"⚠️  Skipping {key}: {message}")
            continue
        merged_vectors[key] = (lambda_weight * v1) + (weight2 * v2)

    print(f"✅ Weighted task vector merge complete: {len(merged_vectors)} parameters merged")
    return merged_vectors


def merge_weighted(
    adapter1_path: Path,
    adapter2_path: Path,
    lambda_weight: float,
    output_path: Path,
    source_metadata: Optional[list[Dict]] = None,
    merge_mode: str = "common",
    register_run: bool = True,
    show_progress: bool = True,
) -> Path:
    """Perform weighted averaging merge of two adapters.

    This is the high-level API for weighted merging. It handles:
    1. Loading adapter weights
    2. Computing weighted average
    3. Creating metadata with lambda parameter
    4. Saving merged adapter
    5. Registering the run

    Args:
        adapter1_path: Path to first adapter directory
        adapter2_path: Path to second adapter directory
        lambda_weight: Weight for first adapter (0.0 to 1.0)
        output_path: Directory to save merged adapter (should be a run directory)
        source_metadata: Optional list of [meta1, meta2] for source adapters
        merge_mode: How to handle different parameters ("common" or "strict")
        register_run: Whether to register this run with RunManager
        show_progress: Whether to print progress messages

    Returns:
        Path to the saved merged adapter

    Raises:
        ValueError: If lambda not in [0, 1]

    Example:
        >>> from merging import merge_weighted
        >>> from merging.core.utils import resolve_best_adapter, create_merge_output_path
        >>>
        >>> # Resolve best adapters
        >>> adapter1, meta1 = resolve_best_adapter("asr")
        >>> adapter2, meta2 = resolve_best_adapter("emotion")
        >>>
        >>> # Create output path with lambda in name
        >>> output_path = create_merge_output_path(
        ...     "weighted",
        ...     ["asr", "emotion"],
        ...     {"lambda": 0.7}
        ... )
        >>>
        >>> # Merge with 70% ASR, 30% emotion
        >>> merged = merge_weighted(
        ...     adapter1_path=adapter1,
        ...     adapter2_path=adapter2,
        ...     lambda_weight=0.7,
        ...     output_path=output_path,
        ...     source_metadata=[meta1, meta2],
        ... )
    """
    if show_progress:
        print(f"\n🔀 Merging 2 adapters using weighted averaging")
        print(f"   Method: weighted")
        print(f"   Lambda (λ): {lambda_weight}")
        print(f"   Mode: {merge_mode}")
        print(f"   1. {adapter1_path} (weight: {lambda_weight:.1%})")
        print(f"   2. {adapter2_path} (weight: {1-lambda_weight:.1%})")

    # Load adapter weights
    if show_progress:
        print(f"\n📥 Loading adapter 1: {adapter1_path.name}")
    weights1 = load_adapter_weights(adapter1_path)
    if show_progress:
        print(f"   Loaded {len(weights1)} parameters")

    if show_progress:
        print(f"\n📥 Loading adapter 2: {adapter2_path.name}")
    weights2 = load_adapter_weights(adapter2_path)
    if show_progress:
        print(f"   Loaded {len(weights2)} parameters")

    # Merge using weighted averaging
    merged_weights = merge_adapters_weighted(
        weights1,
        weights2,
        lambda_weight,
        merge_mode=merge_mode,
    )

    # Build metadata
    metadata = {
        "merge_method": "weighted",
        "lambda": lambda_weight,
        "merge_mode": merge_mode,
        "num_adapters": 2,
        "timestamp": datetime.now().isoformat(),
        "source_adapters": source_metadata if source_metadata else [
            {"path": str(adapter1_path), "weight": lambda_weight},
            {"path": str(adapter2_path), "weight": 1.0 - lambda_weight},
        ],
        "num_parameters": len(merged_weights),
        "parameter_names_sample": list(merged_weights.keys())[:5],
    }

    # Ensure weights are in metadata if provided
    if source_metadata:
        metadata["source_adapters"][0]["weight"] = lambda_weight
        metadata["source_adapters"][1]["weight"] = 1.0 - lambda_weight

    # Save merged adapter
    if show_progress:
        print(f"\n💾 Saving merged adapter to {output_path}")

    save_merged_adapter(
        weights=merged_weights,
        output_path=output_path,
        reference_adapter_path=adapter1_path,  # Use first adapter as reference
        metadata=metadata,
        register_run=register_run,
    )

    if show_progress:
        print(f"\n✅ Weighted merge complete!")
        print(f"   Output: {output_path}")
        print(f"   Lambda: {lambda_weight}")

    return output_path


__all__ = [
    "merge_weighted",
    "merge_adapters_weighted",
    "merge_task_vectors_weighted",
]
