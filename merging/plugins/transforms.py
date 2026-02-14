"""Pre-merge adapter transform plugin system."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, List, Mapping

import torch

from merging.config.specs import TransformSpec
from merging.policies.lambda_policy import extract_layer_index

TensorDict = Dict[str, torch.Tensor]
TransformFn = Callable[[TensorDict, Mapping[str, object]], TensorDict]

_TRANSFORMS: Dict[str, TransformFn] = {}


def register_transform(name: str, fn: TransformFn) -> None:
    key = name.strip().lower()
    if not key:
        raise ValueError("Transform name must be non-empty.")
    if key in _TRANSFORMS:
        raise ValueError(f"Transform already registered: {key}")
    _TRANSFORMS[key] = fn


def list_transforms() -> List[str]:
    return sorted(_TRANSFORMS.keys())


def get_transform(name: str) -> TransformFn:
    key = name.strip().lower()
    if key not in _TRANSFORMS:
        available = ", ".join(list_transforms())
        raise ValueError(f"Unknown transform '{name}'. Available: {available}")
    return _TRANSFORMS[key]


def apply_transforms(weights: TensorDict, transforms: List[TransformSpec]) -> TensorDict:
    out = dict(weights)
    for spec in transforms:
        transform = get_transform(spec.name)
        out = transform(out, spec.params)
    return out


def _identity_transform(weights: TensorDict, params: Mapping[str, object]) -> TensorDict:
    _ = params
    return dict(weights)


def _validate_ties_transform_params(params: Mapping[str, object]) -> tuple[float, float]:
    k_value = params.get("k", 20.0)
    if not isinstance(k_value, (int, float)) or isinstance(k_value, bool):
        raise ValueError(f"ties transform param 'k' must be a float/int in [0,100], got {type(k_value).__name__}")
    k_percent = float(k_value)
    if not 0.0 <= k_percent <= 100.0:
        raise ValueError(f"ties transform param 'k' must be in [0,100], got {k_percent}")

    lambda_value = params.get("lambda", 1.0)
    if not isinstance(lambda_value, (int, float)) or isinstance(lambda_value, bool):
        raise ValueError(
            "ties transform param 'lambda' must be a float/int scaling factor, "
            f"got {type(lambda_value).__name__}"
        )
    lambda_scale = float(lambda_value)
    return k_percent, lambda_scale


def _global_topk_threshold(abs_values: torch.Tensor, k_percent: float) -> torch.Tensor:
    if abs_values.numel() == 0 or k_percent <= 0.0:
        return torch.tensor(float("inf"), dtype=torch.float32, device=abs_values.device)
    if k_percent >= 100.0:
        return torch.tensor(0.0, dtype=torch.float32, device=abs_values.device)

    keep_count = int(math.ceil(abs_values.numel() * (k_percent / 100.0)))
    keep_count = max(1, min(keep_count, abs_values.numel()))
    topk_values = torch.topk(abs_values, k=keep_count, largest=True, sorted=False).values
    return torch.min(topk_values)


def _ties_transform(weights: TensorDict, params: Mapping[str, object]) -> TensorDict:
    k_percent, lambda_scale = _validate_ties_transform_params(params)
    if not weights:
        return {}
    if k_percent <= 0.0:
        return {
            key: (torch.zeros_like(tensor, dtype=torch.float32) * lambda_scale).to(dtype=tensor.dtype)
            for key, tensor in weights.items()
        }
    if k_percent >= 100.0:
        return {key: (tensor.detach().to(dtype=torch.float32) * lambda_scale).to(dtype=tensor.dtype) for key, tensor in weights.items()}

    flat = [tensor.detach().to(dtype=torch.float32).abs().reshape(-1) for tensor in weights.values()]
    abs_values = torch.cat(flat) if flat else torch.empty(0, dtype=torch.float32)
    threshold = _global_topk_threshold(abs_values, k_percent)

    trimmed: TensorDict = {}
    for key, tensor in weights.items():
        tensor_f32 = tensor.detach().to(dtype=torch.float32)
        keep_mask = tensor_f32.abs() >= threshold
        out = torch.where(keep_mask, tensor_f32, torch.zeros_like(tensor_f32))
        trimmed[key] = (out * lambda_scale).to(dtype=tensor.dtype)
    return trimmed


@dataclass(frozen=True)
class TransformScaffoldNote:
    name: str
    message: str


def _ties_transform_scaffold(weights: TensorDict, params: Mapping[str, object]) -> TensorDict:
    _ = params
    # Scaffold only: keep as no-op until TIES-as-transform is intentionally implemented.
    return dict(weights)


def _validate_layer_l2_normalize_params(params: Mapping[str, object]) -> tuple[float, float, bool]:
    target_norm_value = params.get("target_norm", 1.0)
    if not isinstance(target_norm_value, (int, float)) or isinstance(target_norm_value, bool):
        raise ValueError(
            "layer_l2_normalize param 'target_norm' must be a float/int >= 0, "
            f"got {type(target_norm_value).__name__}"
        )
    target_norm = float(target_norm_value)
    if target_norm < 0.0:
        raise ValueError(f"layer_l2_normalize param 'target_norm' must be >= 0, got {target_norm}")

    eps_value = params.get("eps", 1e-12)
    if not isinstance(eps_value, (int, float)) or isinstance(eps_value, bool):
        raise ValueError(
            "layer_l2_normalize param 'eps' must be a float/int > 0, "
            f"got {type(eps_value).__name__}"
        )
    eps = float(eps_value)
    if eps <= 0.0:
        raise ValueError(f"layer_l2_normalize param 'eps' must be > 0, got {eps}")

    include_non_layer_keys_value = params.get("include_non_layer_keys", True)
    if not isinstance(include_non_layer_keys_value, bool):
        raise ValueError(
            "layer_l2_normalize param 'include_non_layer_keys' must be bool, "
            f"got {type(include_non_layer_keys_value).__name__}"
        )
    include_non_layer_keys = bool(include_non_layer_keys_value)

    return target_norm, eps, include_non_layer_keys


def _layer_l2_normalize_transform(weights: TensorDict, params: Mapping[str, object]) -> TensorDict:
    if not weights:
        return {}

    target_norm, eps, include_non_layer_keys = _validate_layer_l2_normalize_params(params)
    bucketed_keys: Dict[int, List[str]] = {}
    for key in weights.keys():
        layer_idx = extract_layer_index(key)
        if layer_idx is None:
            if not include_non_layer_keys:
                continue
            layer_idx = -1
        bucketed_keys.setdefault(int(layer_idx), []).append(key)

    out: TensorDict = {key: tensor.detach().clone() for key, tensor in weights.items()}
    for keys in bucketed_keys.values():
        l2_sq = 0.0
        for key in keys:
            tensor_f32 = weights[key].detach().to(dtype=torch.float32)
            l2_sq += float(tensor_f32.pow(2).sum().item())
        norm = math.sqrt(l2_sq)
        if norm <= eps:
            continue
        scale = target_norm / norm
        for key in keys:
            tensor_f32 = weights[key].detach().to(dtype=torch.float32)
            out[key] = (tensor_f32 * scale).to(dtype=weights[key].dtype)

    return out


def register_builtin_transforms() -> None:
    if "identity" not in _TRANSFORMS:
        register_transform("identity", _identity_transform)
    if "ties" not in _TRANSFORMS:
        register_transform("ties", _ties_transform)
    if "ties_scaffold" not in _TRANSFORMS:
        register_transform("ties_scaffold", _ties_transform_scaffold)
    if "layer_l2_normalize" not in _TRANSFORMS:
        register_transform("layer_l2_normalize", _layer_l2_normalize_transform)


register_builtin_transforms()


__all__ = [
    "register_transform",
    "get_transform",
    "list_transforms",
    "apply_transforms",
    "TransformScaffoldNote",
]
