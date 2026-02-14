"""DARE merge in task-vector (delta) space."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from experiments.extract_vector import extract_task_vector_from_lora
from merging.config.specs import merge_spec_from_legacy_args
from merging.engine.registry import MergeOutput, build_merge_metadata
from merging.transforms.registry import apply_transforms


TensorDict = Dict[str, torch.Tensor]


def _validate_dare_params(params: Optional[Dict[str, object]]) -> Tuple[float, int]:
    raw = params or {}

    drop_rate_value = raw.get("drop_rate", 0.9)
    if not isinstance(drop_rate_value, (int, float)) or isinstance(drop_rate_value, bool):
        raise ValueError(
            "dare param 'drop_rate' must be a float/int in [0,1), "
            f"got {type(drop_rate_value).__name__}"
        )
    drop_rate = float(drop_rate_value)
    if not 0.0 <= drop_rate < 1.0:
        raise ValueError(f"dare param 'drop_rate' must be in [0,1), got {drop_rate}")

    seed_value = raw.get("seed", 42)
    if not isinstance(seed_value, int) or isinstance(seed_value, bool):
        raise ValueError(f"dare param 'seed' must be an int, got {type(seed_value).__name__}")
    return drop_rate, int(seed_value)


def _resolve_keys_to_merge(task_vectors: List[TensorDict], merge_mode: str) -> Tuple[List[str], int]:
    if merge_mode not in {"common", "strict"}:
        raise ValueError(f"Unsupported merge_mode='{merge_mode}'.")

    key_sets = [set(tv.keys()) for tv in task_vectors]
    if merge_mode == "strict":
        reference = key_sets[0]
        for i, keys in enumerate(key_sets[1:], start=1):
            if keys != reference:
                raise ValueError(
                    f"Task vectors have different parameters in strict mode.\n"
                    f"Missing in vector {i}: {sorted(reference - keys)[:10]}\n"
                    f"Extra in vector {i}: {sorted(keys - reference)[:10]}"
                )
        return sorted(reference), 0

    common_keys = set.intersection(*key_sets) if key_sets else set()
    all_keys = set.union(*key_sets) if key_sets else set()
    missing_count = len(all_keys - common_keys)
    if missing_count:
        print(
            f"⚠️  dare: {missing_count} parameters are not common across all vectors; "
            f"merging {len(common_keys)} common parameters."
        )
    return sorted(common_keys), missing_count


def _apply_dare_to_tensor(
    tensor: torch.Tensor,
    *,
    keep_prob: float,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, int, int]:
    if keep_prob >= 1.0:
        out = tensor.detach().clone()
        return out, int(out.numel()), int(out.numel())

    probs_cpu = torch.rand(tensor.shape, generator=generator, device="cpu", dtype=torch.float32)
    keep = (probs_cpu < keep_prob).to(device=tensor.device)
    scale = 1.0 / keep_prob
    out = torch.where(keep, tensor * scale, torch.zeros_like(tensor))
    kept = int(keep.sum().item())
    total = int(keep.numel())
    return out, kept, total


def sparsify_deltas(
    task_vectors: List[TensorDict],
    *,
    drop_rate: float,
    seed: int,
) -> Tuple[List[TensorDict], Dict[str, float]]:
    keep_prob = 1.0 - drop_rate
    if keep_prob <= 0.0:
        raise ValueError("dare requires drop_rate < 1.0.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    sparse_vectors: List[TensorDict] = []
    kept_entries = 0
    total_entries = 0
    for vector in task_vectors:
        sparse: TensorDict = {}
        for key, tensor in vector.items():
            sparse_tensor, kept, total = _apply_dare_to_tensor(
                tensor,
                keep_prob=keep_prob,
                generator=generator,
            )
            sparse[key] = sparse_tensor
            kept_entries += kept
            total_entries += total
        sparse_vectors.append(sparse)

    effective_sparsity = 1.0 - (float(kept_entries) / float(total_entries)) if total_entries > 0 else 0.0
    stats = {
        "drop_rate": float(drop_rate),
        "keep_prob": float(keep_prob),
        "kept_entries": int(kept_entries),
        "total_entries": int(total_entries),
        "effective_sparsity": float(effective_sparsity),
    }
    return sparse_vectors, stats


def fuse_deltas_uniform(
    task_vectors: List[TensorDict],
    *,
    merge_mode: str,
) -> Tuple[TensorDict, Dict[str, int]]:
    if len(task_vectors) < 2:
        raise ValueError("dare requires at least 2 task vectors.")

    keys_to_merge, missing_key_count = _resolve_keys_to_merge(task_vectors, merge_mode)
    print(f"🧮 dare: merge_mode={merge_mode}, mergeable_keys={len(keys_to_merge)}")

    merged: TensorDict = {}
    shape_mismatch_count = 0
    for key in keys_to_merge:
        shapes = [tv[key].shape for tv in task_vectors]
        if len(set(shapes)) != 1:
            if merge_mode == "strict":
                raise ValueError(f"Shape mismatch for {key}: {shapes}")
            print(f"⚠️  dare: skipping {key} due to shape mismatch {shapes}")
            shape_mismatch_count += 1
            continue
        stacked = torch.stack([tv[key] for tv in task_vectors], dim=0)
        merged[key] = stacked.mean(dim=0)

    if shape_mismatch_count:
        print(f"⚠️  dare: skipped {shape_mismatch_count} keys due to shape mismatch.")

    stats = {
        "skipped_missing_key_count": int(missing_key_count),
        "skipped_shape_mismatch_count": int(shape_mismatch_count),
        "merged_parameter_count": int(len(merged)),
    }
    return merged, stats


def merge_dare(
    *,
    adapter_paths: List[Path],
    source_metadata: List[Dict],
    merge_mode: str,
    params: Optional[Dict[str, object]],
) -> MergeOutput:
    if len(adapter_paths) < 2:
        raise ValueError("dare requires at least 2 adapters.")

    spec = merge_spec_from_legacy_args(
        adapters=[str(p) for p in adapter_paths],
        method="dare",
        merge_mode=merge_mode,
        lambda_weight=None,
        params=params,
    )
    drop_rate, seed = _validate_dare_params(spec.method_params)

    print(
        f"🧮 dare: extracting task vectors for {len(adapter_paths)} adapters "
        f"(drop_rate={drop_rate:.4f}, seed={seed})..."
    )
    task_vectors = [apply_transforms(extract_task_vector_from_lora(path), spec.transforms) for path in adapter_paths]
    sparse_vectors, sparsify_stats = sparsify_deltas(task_vectors, drop_rate=drop_rate, seed=seed)
    merged_delta, fuse_stats = fuse_deltas_uniform(sparse_vectors, merge_mode=merge_mode)
    merged_nonzero_entries = int(sum((tensor != 0).sum().item() for tensor in merged_delta.values()))

    metadata = build_merge_metadata(
        method="dare",
        merge_mode=merge_mode,
        num_adapters=len(adapter_paths),
        source_metadata=source_metadata,
        num_parameters=len(merged_delta),
        params={"drop_rate": drop_rate, "seed": seed},
        method_params=spec.method_params,
        lambda_policy=spec.method_params.get("lambda_policy"),
        transforms=[{"name": t.name, "params": dict(t.params)} for t in spec.transforms],
        optimizer=spec.method_params.get("optimizer"),
    )
    metadata["dare_stats"] = {
        **sparsify_stats,
        **fuse_stats,
        "merged_nonzero_entries": merged_nonzero_entries,
    }
    print(
        "✅ dare complete: "
        f"merged_parameters={len(merged_delta)}, "
        f"effective_sparsity={sparsify_stats['effective_sparsity']:.4f}, "
        f"nonzero_entries={merged_nonzero_entries}"
    )
    return MergeOutput(merged_delta=merged_delta, merged_weights=None, metadata=metadata)


__all__ = [
    "merge_dare",
    "sparsify_deltas",
    "fuse_deltas_uniform",
    "_validate_dare_params",
]
