"""Typed merge specification models and normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

from merging.runtime.utils import PACKAGE_ROOT


@dataclass(frozen=True)
class TransformSpec:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LambdaPolicySpec:
    type: str
    value: Optional[float] = None
    default: Optional[float] = None
    overrides: Dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizerSpec:
    type: str = "none"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MergeSpec:
    adapters: List[str]
    method: str
    merge_mode: str = "common"
    method_params: Dict[str, Any] = field(default_factory=dict)
    transforms: List[TransformSpec] = field(default_factory=list)
    lambda_policy: Optional[LambdaPolicySpec] = None
    optimizer: Optional[OptimizerSpec] = None


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PACKAGE_ROOT / resolved
    return resolved


def _parse_transform_list(raw: Any) -> List[TransformSpec]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("transforms must be a list.")
    transforms: List[TransformSpec] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("Each transform entry must be a mapping.")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError("Each transform entry requires a non-empty name.")
        params = item.get("params", {})
        if not isinstance(params, Mapping):
            raise ValueError(f"transform '{name}' params must be a mapping.")
        transforms.append(TransformSpec(name=name, params=dict(params)))
    return transforms


def _parse_lambda_policy(raw: Any, fallback_lambda: Optional[float]) -> Optional[LambdaPolicySpec]:
    if raw is None:
        if fallback_lambda is None:
            return None
        return LambdaPolicySpec(type="scalar", value=float(fallback_lambda))

    if not isinstance(raw, Mapping):
        raise ValueError("lambda_policy must be a mapping.")

    policy_type = str(raw.get("type", "")).strip().lower()
    if not policy_type:
        raise ValueError("lambda_policy.type is required.")

    if policy_type == "scalar":
        value = raw.get("value", fallback_lambda)
        if value is None:
            raise ValueError("lambda_policy.type=scalar requires value.")
        return LambdaPolicySpec(type="scalar", value=float(value))

    if policy_type == "per_layer":
        default = raw.get("default", fallback_lambda)
        if default is None:
            raise ValueError("lambda_policy.type=per_layer requires default or --lambda.")
        overrides_raw = raw.get("overrides", {})
        if overrides_raw is None:
            overrides_raw = {}
        if not isinstance(overrides_raw, Mapping):
            raise ValueError("lambda_policy.overrides must be a mapping of layer->lambda.")
        overrides: Dict[int, float] = {}
        for k, v in overrides_raw.items():
            overrides[int(k)] = float(v)
        return LambdaPolicySpec(type="per_layer", default=float(default), overrides=overrides)

    raise ValueError(f"Unsupported lambda_policy.type: {policy_type}")


def _parse_optimizer(raw: Any) -> Optional[OptimizerSpec]:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("optimizer must be a mapping.")
    opt_type = str(raw.get("type", "none")).strip().lower()
    params = raw.get("params", {})
    if not isinstance(params, Mapping):
        raise ValueError("optimizer.params must be a mapping.")
    return OptimizerSpec(type=opt_type, params=dict(params))


def merge_spec_from_legacy_args(
    *,
    adapters: List[str],
    method: str,
    merge_mode: str,
    lambda_weight: Optional[float],
    params: Optional[Dict[str, Any]] = None,
) -> MergeSpec:
    method_params = dict(params or {})
    if lambda_weight is not None and "lambda" not in method_params:
        method_params["lambda"] = float(lambda_weight)

    lambda_policy = _parse_lambda_policy(
        method_params.get("lambda_policy"),
        fallback_lambda=method_params.get("lambda"),
    )
    transforms = _parse_transform_list(method_params.get("transforms"))
    optimizer = _parse_optimizer(method_params.get("optimizer"))

    return MergeSpec(
        adapters=list(adapters),
        method=method,
        merge_mode=merge_mode,
        method_params=method_params,
        transforms=transforms,
        lambda_policy=lambda_policy,
        optimizer=optimizer,
    )


def load_merge_spec(path: str | Path) -> MergeSpec:
    resolved = _resolve_path(path)
    with resolved.open("r") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, Mapping):
        raise ValueError("Merge config must be a mapping.")
    adapters = payload.get("adapters")
    method = payload.get("method")
    if not adapters or not method:
        raise ValueError("Merge config requires adapters and method.")
    if not isinstance(adapters, list):
        raise ValueError("Merge config adapters must be a list.")

    method_params_raw = payload.get("method_params", {})
    if not isinstance(method_params_raw, Mapping):
        raise ValueError("method_params must be a mapping.")
    method_params = dict(method_params_raw)

    lambda_policy = _parse_lambda_policy(
        payload.get("lambda_policy"),
        fallback_lambda=method_params.get("lambda"),
    )
    transforms = _parse_transform_list(payload.get("transforms"))
    optimizer = _parse_optimizer(payload.get("optimizer"))

    return MergeSpec(
        adapters=[str(x) for x in adapters],
        method=str(method),
        merge_mode=str(payload.get("merge_mode", "common")),
        method_params=method_params,
        transforms=transforms,
        lambda_policy=lambda_policy,
        optimizer=optimizer,
    )


def merge_spec_to_params(spec: MergeSpec) -> Dict[str, Any]:
    params = dict(spec.method_params)
    if "lambda" not in params and spec.lambda_policy is not None:
        if spec.lambda_policy.type == "scalar" and spec.lambda_policy.value is not None:
            params["lambda"] = float(spec.lambda_policy.value)
        elif spec.lambda_policy.type == "per_layer" and spec.lambda_policy.default is not None:
            params["lambda"] = float(spec.lambda_policy.default)
    if spec.lambda_policy is not None:
        params["lambda_policy"] = {
            "type": spec.lambda_policy.type,
            "value": spec.lambda_policy.value,
            "default": spec.lambda_policy.default,
            "overrides": dict(spec.lambda_policy.overrides),
        }
    if spec.transforms:
        params["transforms"] = [{"name": t.name, "params": dict(t.params)} for t in spec.transforms]
    if spec.optimizer is not None:
        params["optimizer"] = {"type": spec.optimizer.type, "params": dict(spec.optimizer.params)}
    return params


__all__ = [
    "TransformSpec",
    "LambdaPolicySpec",
    "OptimizerSpec",
    "MergeSpec",
    "load_merge_spec",
    "merge_spec_from_legacy_args",
    "merge_spec_to_params",
]
