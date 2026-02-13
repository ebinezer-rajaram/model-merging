"""Single-scalar merge in task-vector (delta) space.

Formula:
    delta_merged = scale * (delta_1 + delta_2 + ... + delta_n)
"""

from __future__ import annotations

from typing import Dict, List

import torch


def merge_task_vectors_uniform_scalar(
    task_vectors: List[Dict[str, torch.Tensor]],
    *,
    scale: float = 1.0,
    merge_mode: str = "common",
) -> Dict[str, torch.Tensor]:
    """Merge N task vectors using one global scalar over their sum.

    Args:
        task_vectors: List of task vector dicts (base_key -> delta tensor)
        scale: Global scalar multiplier applied after summation.
        merge_mode: "common" or "strict" key-handling behavior.

    Returns:
        Merged task-vector dictionary.
    """
    if len(task_vectors) < 2:
        raise ValueError("uniform_scalar_delta requires at least 2 task vectors.")
    if merge_mode not in {"common", "strict"}:
        raise ValueError(f"Unsupported merge_mode='{merge_mode}'.")

    scale = float(scale)

    key_sets = [set(tv.keys()) for tv in task_vectors]
    common_keys = set.intersection(*key_sets)
    all_keys = set.union(*key_sets)

    if merge_mode == "strict":
        reference = key_sets[0]
        for i, keys in enumerate(key_sets[1:], start=1):
            if keys != reference:
                raise ValueError(
                    f"Task vectors have different parameters in strict mode.\n"
                    f"Missing in vector {i}: {sorted(reference - keys)[:10]}\n"
                    f"Extra in vector {i}: {sorted(keys - reference)[:10]}"
                )
        keys_to_merge = reference
    else:
        keys_to_merge = common_keys
        unique = all_keys - common_keys
        if unique:
            print(
                f"⚠️  uniform_scalar_delta: {len(unique)} parameters are not common across all vectors; "
                f"merging {len(keys_to_merge)} common parameters."
            )

    merged_vectors: Dict[str, torch.Tensor] = {}
    print(
        f"🧮 uniform_scalar_delta: merging {len(keys_to_merge)} parameters "
        f"with scale={scale:g}"
    )

    for key in keys_to_merge:
        shapes = [tv[key].shape for tv in task_vectors]
        if len(set(shapes)) != 1:
            if merge_mode == "strict":
                raise ValueError(f"Shape mismatch for {key}: {shapes}")
            print(f"⚠️  uniform_scalar_delta: skipping {key} due to shape mismatch {shapes}")
            continue

        out = torch.zeros_like(task_vectors[0][key])
        for vector in task_vectors:
            out = out + vector[key]
        merged_vectors[key] = scale * out

    print(f"✅ uniform_scalar_delta complete: {len(merged_vectors)} merged parameters.")
    return merged_vectors


__all__ = ["merge_task_vectors_uniform_scalar"]
