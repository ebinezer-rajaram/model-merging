"""Uniform (simple averaging) adapter merging method.

This module implements uniform averaging of multiple LoRA adapters:
    W_merged = (W_1 + W_2 + ... + W_n) / n

This is the simplest merging approach and often works surprisingly well.
It treats all source adapters equally without any weighting.

Usage:
    from merging import merge_uniform
    from merging.runtime.utils import resolve_best_adapter, create_merge_output_path

    # Resolve adapters
    adapter1, meta1 = resolve_best_adapter("asr")
    adapter2, meta2 = resolve_best_adapter("emotion")

    # Create output path
    output_path = create_merge_output_path("uniform", ["asr", "emotion"])

    # Merge
    merged_path = merge_uniform(
        adapter_paths=[adapter1, adapter2],
        output_path=output_path,
        source_metadata=[meta1, meta2],
    )
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch

from merging.runtime.utils import load_adapter_weights, save_merged_adapter


def merge_adapters_uniform(
    adapter_weights: List[Dict[str, torch.Tensor]],
    merge_mode: str = "common",
) -> Dict[str, torch.Tensor]:
    """Merge multiple adapters using uniform averaging.

    Computes simple average: W_merged = (W_1 + W_2 + ... + W_n) / n

    Args:
        adapter_weights: List of adapter weight dictionaries
        merge_mode: How to handle different parameters across adapters:
            - "common": Only merge parameters present in ALL adapters (default)
            - "strict": Require all adapters to have identical parameters

    Returns:
        Merged adapter weights

    Raises:
        ValueError: If no adapters provided or strict mode has mismatched params
    """
    if not adapter_weights:
        raise ValueError("No adapters provided for merging")

    if len(adapter_weights) == 1:
        print("⚠️  Warning: Only one adapter provided, returning it unchanged")
        return adapter_weights[0]

    # Find common and unique parameters
    all_keys = [set(adapter.keys()) for adapter in adapter_weights]
    common_keys = set.intersection(*all_keys)
    all_unique_keys = set.union(*all_keys)

    if merge_mode == "strict":
        # Verify all adapters have the same keys
        reference_keys = all_keys[0]
        for i, adapter_keys in enumerate(all_keys[1:], start=1):
            if adapter_keys != reference_keys:
                missing = reference_keys - adapter_keys
                extra = adapter_keys - reference_keys
                raise ValueError(
                    f"Adapter {i} has different parameters than adapter 0.\n"
                    f"Missing: {missing}\nExtra: {extra}\n"
                    f"Use merge_mode='common' to merge only common parameters."
                )
        keys_to_merge = reference_keys
    else:  # common mode
        keys_to_merge = common_keys
        unique_keys = all_unique_keys - common_keys

        if unique_keys:
            print(f"⚠️  Warning: {len(unique_keys)} parameters are not common across all adapters")
            print(f"   Merging {len(common_keys)} common parameters")
            print(f"   Excluding {len(unique_keys)} unique parameters")

    # Average all common parameters
    merged_weights = {}
    num_adapters = len(adapter_weights)

    print(f"🧮 Computing uniform average of {num_adapters} adapters...")
    print(f"   Parameters to merge: {len(keys_to_merge)}")

    for key in keys_to_merge:
        # Sum all adapter weights for this parameter
        summed = torch.zeros_like(adapter_weights[0][key])
        for adapter in adapter_weights:
            summed += adapter[key]

        # Compute average
        merged_weights[key] = summed / num_adapters

    print(f"✅ Uniform merge complete: {len(merged_weights)} parameters merged")

    return merged_weights


def merge_uniform(
    adapter_paths: List[Path],
    output_path: Path,
    source_metadata: Optional[List[Dict]] = None,
    merge_mode: str = "common",
    register_run: bool = True,
    show_progress: bool = True,
) -> Path:
    """Perform uniform averaging merge of multiple adapters.

    This is the high-level API for uniform merging. It handles:
    1. Loading adapter weights
    2. Computing uniform average
    3. Creating metadata
    4. Saving merged adapter
    5. Registering the run

    Args:
        adapter_paths: List of paths to adapter directories
        output_path: Directory to save merged adapter (should be a run directory)
        source_metadata: Optional list of metadata dicts for source adapters
        merge_mode: How to handle different parameters ("common" or "strict")
        register_run: Whether to register this run with RunManager
        show_progress: Whether to print progress messages

    Returns:
        Path to the saved merged adapter

    Example:
        >>> from merging import merge_uniform
        >>> from merging.runtime.utils import resolve_best_adapter, create_merge_output_path
        >>>
        >>> # Resolve best adapters
        >>> adapter1, meta1 = resolve_best_adapter("asr")
        >>> adapter2, meta2 = resolve_best_adapter("emotion")
        >>>
        >>> # Create output path
        >>> output_path = create_merge_output_path("uniform", ["asr", "emotion"])
        >>>
        >>> # Merge
        >>> merged = merge_uniform(
        ...     adapter_paths=[adapter1, adapter2],
        ...     output_path=output_path,
        ...     source_metadata=[meta1, meta2],
        ... )
    """
    if show_progress:
        print(f"\n🔀 Merging {len(adapter_paths)} adapters using uniform averaging")
        print(f"   Method: uniform")
        print(f"   Mode: {merge_mode}")
        for i, path in enumerate(adapter_paths, 1):
            print(f"   {i}. {path}")

    # Load all adapter weights
    all_weights = []
    for i, adapter_path in enumerate(adapter_paths, 1):
        if show_progress:
            print(f"\n📥 Loading adapter {i}/{len(adapter_paths)}: {adapter_path.name}")
        weights = load_adapter_weights(adapter_path)
        if show_progress:
            print(f"   Loaded {len(weights)} parameters")
        all_weights.append(weights)

    # Merge using uniform averaging
    merged_weights = merge_adapters_uniform(all_weights, merge_mode=merge_mode)

    # Build metadata
    metadata = {
        "merge_method": "uniform",
        "merge_mode": merge_mode,
        "num_adapters": len(adapter_paths),
        "timestamp": datetime.now().isoformat(),
        "source_adapters": source_metadata if source_metadata else [
            {"path": str(p)} for p in adapter_paths
        ],
        "num_parameters": len(merged_weights),
        "parameter_names_sample": list(merged_weights.keys())[:5],  # First 5 for reference
    }

    # Save merged adapter
    if show_progress:
        print(f"\n💾 Saving merged adapter to {output_path}")

    save_merged_adapter(
        weights=merged_weights,
        output_path=output_path,
        reference_adapter_path=adapter_paths[0],  # Use first adapter as reference
        metadata=metadata,
        register_run=register_run,
    )

    if show_progress:
        print(f"\n✅ Uniform merge complete!")
        print(f"   Output: {output_path}")

    return output_path


__all__ = [
    "merge_uniform",
    "merge_adapters_uniform",
]
