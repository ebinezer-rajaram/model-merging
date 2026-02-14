"""Paper-core TIES merge in task-vector (delta) space."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from experiments.extract_vector import extract_task_vector_from_lora
from merging.config.specs import merge_spec_from_legacy_args
from merging.engine.registry import MergeOutput, build_merge_metadata
from merging.plugins.transforms import apply_transforms


TensorDict = Dict[str, torch.Tensor]


def _validate_ties_params(params: Optional[Dict[str, object]]) -> Tuple[float, float]:
    raw = params or {}

    k_value = raw.get("k", 20.0)
    if not isinstance(k_value, (int, float)) or isinstance(k_value, bool):
        raise ValueError(f"ties param 'k' must be a float/int in [0,100], got {type(k_value).__name__}")
    k_percent = float(k_value)
    if not 0.0 <= k_percent <= 100.0:
        raise ValueError(f"ties param 'k' must be in [0,100], got {k_percent}")

    lambda_value = raw.get("lambda", 1.0)
    if not isinstance(lambda_value, (int, float)) or isinstance(lambda_value, bool):
        raise ValueError(
            f"ties param 'lambda' must be a float/int scaling factor, got {type(lambda_value).__name__}"
        )
    lambda_scale = float(lambda_value)
    return k_percent, lambda_scale


def _flatten_abs_values(task_vector: TensorDict) -> torch.Tensor:
    if not task_vector:
        return torch.empty(0, dtype=torch.float32)
    flat = [tensor.detach().to(dtype=torch.float32).abs().reshape(-1) for tensor in task_vector.values()]
    return torch.cat(flat) if flat else torch.empty(0, dtype=torch.float32)


def _global_topk_threshold(abs_values: torch.Tensor, k_percent: float) -> torch.Tensor:
    if abs_values.numel() == 0 or k_percent <= 0.0:
        return torch.tensor(float("inf"), dtype=torch.float32, device=abs_values.device)
    if k_percent >= 100.0:
        return torch.tensor(0.0, dtype=torch.float32, device=abs_values.device)

    keep_count = int(math.ceil(abs_values.numel() * (k_percent / 100.0)))
    keep_count = max(1, min(keep_count, abs_values.numel()))
    topk_values = torch.topk(abs_values, k=keep_count, largest=True, sorted=False).values
    return torch.min(topk_values)


def _trim_task_vector_global(task_vector: TensorDict, k_percent: float) -> TensorDict:
    if not task_vector:
        return {}
    if k_percent <= 0.0:
        return {key: torch.zeros_like(tensor, dtype=torch.float32) for key, tensor in task_vector.items()}
    if k_percent >= 100.0:
        return {key: tensor.detach().to(dtype=torch.float32).clone() for key, tensor in task_vector.items()}

    threshold = _global_topk_threshold(_flatten_abs_values(task_vector), k_percent)
    trimmed: TensorDict = {}
    for key, tensor in task_vector.items():
        tensor_f32 = tensor.detach().to(dtype=torch.float32)
        keep_mask = tensor_f32.abs() >= threshold
        trimmed[key] = torch.where(keep_mask, tensor_f32, torch.zeros_like(tensor_f32))
    return trimmed


def _elect_sign(trimmed_vectors: List[TensorDict]) -> TensorDict:
    if not trimmed_vectors:
        return {}
    elected: TensorDict = {}
    for key in trimmed_vectors[0].keys():
        stacked = torch.stack([tv[key].to(dtype=torch.float32) for tv in trimmed_vectors], dim=0)
        elected[key] = torch.sign(stacked.sum(dim=0))
    return elected


def _disjoint_mean(trimmed_vectors: List[TensorDict], elected_sign_map: TensorDict) -> TensorDict:
    merged: TensorDict = {}
    for key, elected_sign in elected_sign_map.items():
        stacked = torch.stack([tv[key].to(dtype=torch.float32) for tv in trimmed_vectors], dim=0)
        aligned_mask = (
            (torch.sign(stacked) == elected_sign.unsqueeze(0))
            & (elected_sign.unsqueeze(0) != 0.0)
            & (stacked != 0.0)
        )
        aligned_sum = torch.where(aligned_mask, stacked, torch.zeros_like(stacked)).sum(dim=0)
        aligned_count = aligned_mask.sum(dim=0)
        merged[key] = torch.where(
            aligned_count > 0,
            aligned_sum / aligned_count.to(dtype=torch.float32),
            torch.zeros_like(aligned_sum),
        )
    return merged


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
            f"⚠️  ties: {missing_count} parameters are not common across all vectors; "
            f"merging {len(common_keys)} common parameters."
        )
    return sorted(common_keys), missing_count


def merge_ties(
    *,
    adapter_paths: List[Path],
    source_metadata: List[Dict],
    merge_mode: str,
    params: Optional[Dict[str, object]],
) -> MergeOutput:
    if len(adapter_paths) < 2:
        raise ValueError("ties requires at least 2 adapters.")

    spec = merge_spec_from_legacy_args(
        adapters=[str(p) for p in adapter_paths],
        method="ties",
        merge_mode=merge_mode,
        lambda_weight=None if params is None else params.get("lambda"),
        params=params,
    )
    k_percent, lambda_scale = _validate_ties_params(spec.method_params)

    print(f"🧮 ties: extracting task vectors for {len(adapter_paths)} adapters...")
    task_vectors = [apply_transforms(extract_task_vector_from_lora(path), spec.transforms) for path in adapter_paths]
    keys_to_merge, missing_key_count = _resolve_keys_to_merge(task_vectors, merge_mode)
    print(
        f"🧮 ties: merge_mode={merge_mode}, mergeable_keys={len(keys_to_merge)}, "
        f"trim_k={k_percent:.2f}%, lambda={lambda_scale:.4f}"
    )

    shape_mismatch_count = 0
    for key in list(keys_to_merge):
        shapes = [tv[key].shape for tv in task_vectors]
        if len(set(shapes)) != 1:
            if merge_mode == "strict":
                raise ValueError(f"Shape mismatch for {key}: {shapes}")
            print(f"⚠️  ties: skipping {key} due to shape mismatch {shapes}")
            keys_to_merge.remove(key)
            shape_mismatch_count += 1

    if shape_mismatch_count:
        print(f"⚠️  ties: skipped {shape_mismatch_count} keys due to shape mismatch.")

    mergeable_vectors: List[TensorDict] = [{key: tv[key] for key in keys_to_merge} for tv in task_vectors]
    print(f"🧮 ties: trimming vectors globally to top-{k_percent:.2f}% magnitude entries...")
    trimmed_vectors = [_trim_task_vector_global(tv, k_percent) for tv in mergeable_vectors]
    print("🧮 ties: electing per-parameter signs from trimmed vectors...")
    elected_sign_map = _elect_sign(trimmed_vectors)
    print("🧮 ties: computing disjoint sign-aligned mean...")
    merged_delta_f32 = _disjoint_mean(trimmed_vectors, elected_sign_map)

    merged_delta: TensorDict = {}
    for key, tensor in merged_delta_f32.items():
        scaled = tensor * lambda_scale
        merged_delta[key] = scaled.to(dtype=mergeable_vectors[0][key].dtype)

    total_entries = sum(tv[key].numel() for tv in mergeable_vectors for key in keys_to_merge)
    trimmed_nonzero_entries = sum(
        int((tv[key] != 0.0).sum().item()) for tv in trimmed_vectors for key in keys_to_merge
    )
    trim_density = float(trimmed_nonzero_entries) / float(total_entries) if total_entries > 0 else 0.0

    active_sign_positions = 0
    conflict_positions = 0
    merged_nonzero_entries = 0
    for key in keys_to_merge:
        stacked = torch.stack([tv[key].to(dtype=torch.float32) for tv in trimmed_vectors], dim=0)
        active = (stacked != 0.0).any(dim=0)
        pos_present = (stacked > 0.0).any(dim=0)
        neg_present = (stacked < 0.0).any(dim=0)
        conflict = pos_present & neg_present
        active_sign_positions += int(active.sum().item())
        conflict_positions += int((conflict & active).sum().item())
        merged_nonzero_entries += int((merged_delta_f32[key] != 0.0).sum().item())
    sign_conflict_rate = (
        float(conflict_positions) / float(active_sign_positions) if active_sign_positions > 0 else 0.0
    )

    metadata = build_merge_metadata(
        method="ties",
        merge_mode=merge_mode,
        num_adapters=len(adapter_paths),
        source_metadata=source_metadata,
        num_parameters=len(merged_delta),
        params={"k": k_percent, "lambda": lambda_scale},
        lambda_weight=lambda_scale,
        method_params=spec.method_params,
        lambda_policy=spec.method_params.get("lambda_policy"),
        transforms=[{"name": t.name, "params": dict(t.params)} for t in spec.transforms],
        optimizer=spec.method_params.get("optimizer"),
    )
    metadata["ties_stats"] = {
        "trim_density": trim_density,
        "sign_conflict_rate": sign_conflict_rate,
        "merged_parameter_count": len(merged_delta),
        "merged_nonzero_entries": merged_nonzero_entries,
        "skipped_missing_key_count": missing_key_count,
        "skipped_shape_mismatch_count": shape_mismatch_count,
    }
    print(
        "✅ ties complete: "
        f"merged_parameters={len(merged_delta)}, "
        f"trim_density={trim_density:.4f}, "
        f"sign_conflict_rate={sign_conflict_rate:.4f}, "
        f"nonzero_entries={merged_nonzero_entries}"
    )

    return MergeOutput(merged_delta=merged_delta, merged_weights=None, metadata=metadata)


__all__ = [
    "merge_ties",
    "_validate_ties_params",
    "_flatten_abs_values",
    "_global_topk_threshold",
    "_trim_task_vector_global",
    "_elect_sign",
    "_disjoint_mean",
]
