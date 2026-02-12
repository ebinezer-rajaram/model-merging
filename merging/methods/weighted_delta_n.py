"""N-adapter weighted merge in task-vector (delta) space."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional

import torch

from merging.policies.lambda_policy import extract_layer_index


def _validate_coefficients(
    coeffs: List[float],
    expected_len: int,
    label: str,
    *,
    allow_negative_coefficients: bool = False,
) -> None:
    if len(coeffs) != expected_len:
        raise ValueError(f"{label} must have length={expected_len}, got {len(coeffs)}")
    for idx, value in enumerate(coeffs):
        value_f = float(value)
        if allow_negative_coefficients:
            if not -1.0 <= value_f <= 1.0:
                raise ValueError(f"{label}[{idx}] must be in [-1,1], got {value}")
        elif not 0.0 <= value_f <= 1.0:
            raise ValueError(f"{label}[{idx}] must be in [0,1], got {value}")


def _resolve_coefficients_for_key(
    *,
    key: str,
    num_vectors: int,
    task_coefficients: Optional[List[float]],
    layer_task_coefficients: Optional[Mapping[int, List[float]]],
    default_task_coefficients: Optional[List[float]],
    normalize_coefficients: bool,
    allow_negative_coefficients: bool,
) -> List[float]:
    if layer_task_coefficients:
        layer_idx = extract_layer_index(key)
        if layer_idx is not None and layer_idx in layer_task_coefficients:
            coeffs = [float(x) for x in layer_task_coefficients[layer_idx]]
            _validate_coefficients(
                coeffs,
                num_vectors,
                f"layer_task_coefficients[{layer_idx}]",
                allow_negative_coefficients=allow_negative_coefficients,
            )
        elif default_task_coefficients is not None:
            coeffs = [float(x) for x in default_task_coefficients]
            _validate_coefficients(
                coeffs,
                num_vectors,
                "default_task_coefficients",
                allow_negative_coefficients=allow_negative_coefficients,
            )
        elif task_coefficients is not None:
            coeffs = [float(x) for x in task_coefficients]
            _validate_coefficients(
                coeffs,
                num_vectors,
                "task_coefficients",
                allow_negative_coefficients=allow_negative_coefficients,
            )
        else:
            coeffs = [1.0 / float(num_vectors)] * num_vectors
    elif task_coefficients is not None:
        coeffs = [float(x) for x in task_coefficients]
        _validate_coefficients(
            coeffs,
            num_vectors,
            "task_coefficients",
            allow_negative_coefficients=allow_negative_coefficients,
        )
    else:
        coeffs = [1.0 / float(num_vectors)] * num_vectors

    if normalize_coefficients:
        total = float(sum(coeffs))
        if total <= 0.0:
            raise ValueError("Coefficient sum must be > 0 when normalize_coefficients=true.")
        coeffs = [c / total for c in coeffs]
    return coeffs


def merge_task_vectors_weighted_n(
    task_vectors: List[Dict[str, torch.Tensor]],
    *,
    merge_mode: str = "common",
    task_coefficients: Optional[List[float]] = None,
    normalize_coefficients: bool = True,
    layer_task_coefficients: Optional[Mapping[int, List[float]]] = None,
    default_task_coefficients: Optional[List[float]] = None,
    allow_negative_coefficients: bool = False,
) -> Dict[str, torch.Tensor]:
    if len(task_vectors) < 2:
        raise ValueError("weighted_delta_n requires at least 2 task vectors.")
    if merge_mode not in {"common", "strict"}:
        raise ValueError(f"Unsupported merge_mode='{merge_mode}'.")

    num_vectors = len(task_vectors)
    if task_coefficients is not None:
        _validate_coefficients(
            [float(x) for x in task_coefficients],
            num_vectors,
            "task_coefficients",
            allow_negative_coefficients=allow_negative_coefficients,
        )
    if default_task_coefficients is not None:
        _validate_coefficients(
            [float(x) for x in default_task_coefficients],
            num_vectors,
            "default_task_coefficients",
            allow_negative_coefficients=allow_negative_coefficients,
        )
    if layer_task_coefficients is not None:
        for layer_idx, coeffs in layer_task_coefficients.items():
            if int(layer_idx) < 0:
                raise ValueError(f"Layer index must be >= 0, got {layer_idx}")
            _validate_coefficients(
                [float(x) for x in coeffs],
                num_vectors,
                f"layer_task_coefficients[{layer_idx}]",
                allow_negative_coefficients=allow_negative_coefficients,
            )

    key_sets = [set(tv.keys()) for tv in task_vectors]
    common_keys = set.intersection(*key_sets)
    all_keys = set.union(*key_sets)
    if merge_mode == "strict":
        ref = key_sets[0]
        for i, keys in enumerate(key_sets[1:], start=1):
            if keys != ref:
                raise ValueError(
                    f"Task vectors have different parameters in strict mode.\n"
                    f"Missing in vector {i}: {sorted(ref - keys)[:10]}\n"
                    f"Extra in vector {i}: {sorted(keys - ref)[:10]}"
                )
        keys_to_merge = ref
    else:
        keys_to_merge = common_keys
        if all_keys - common_keys:
            print(
                f"⚠️  weighted_delta_n: {len(all_keys - common_keys)} parameters are not common across all vectors; "
                f"merging {len(keys_to_merge)} common parameters."
            )

    merged_vectors: Dict[str, torch.Tensor] = {}
    policy_type = "layer_task_coefficients" if layer_task_coefficients else "task_coefficients"
    print(f"🧮 weighted_delta_n: merging {len(keys_to_merge)} parameters using {policy_type}")
    for key in keys_to_merge:
        coeffs = _resolve_coefficients_for_key(
            key=key,
            num_vectors=num_vectors,
            task_coefficients=task_coefficients,
            layer_task_coefficients=layer_task_coefficients,
            default_task_coefficients=default_task_coefficients,
            normalize_coefficients=normalize_coefficients,
            allow_negative_coefficients=allow_negative_coefficients,
        )

        shapes = [task_vectors[i][key].shape for i in range(num_vectors)]
        if len(set(shapes)) != 1:
            if merge_mode == "strict":
                raise ValueError(f"Shape mismatch for {key}: {shapes}")
            print(f"⚠️  weighted_delta_n: skipping {key} due to shape mismatch {shapes}")
            continue

        out = torch.zeros_like(task_vectors[0][key])
        for i in range(num_vectors):
            out = out + (coeffs[i] * task_vectors[i][key])
        merged_vectors[key] = out

    print(f"✅ weighted_delta_n complete: {len(merged_vectors)} merged parameters.")
    return merged_vectors


__all__ = ["merge_task_vectors_weighted_n"]
