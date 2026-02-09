"""Unified merge/sweep config model and normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
import warnings

import yaml

from merging.runtime.utils import PACKAGE_ROOT

_ALLOWED_TOP_LEVEL_KEYS = {
    "adapters",
    "method",
    "merge_mode",
    "method_params",
    "lambda_policy",
    "optimizer",
    "transforms",
    "search",
    "constraint_nonnegative",
    "eval_tasks",
    "split",
    "save_merged",
    "eval_subset",
    "output_dir",
    "compute_missing_interference_baselines",
}

_LEGACY_TOP_LEVEL_KEYS = {
    "grid",       # old sweep shape
    "lambda",     # legacy merge shorthand
    "params",     # legacy merge shorthand
}


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


@dataclass(frozen=True)
class MergeConfig:
    adapters: List[str]
    method: str
    merge_mode: str = "common"
    method_params: Dict[str, Any] = field(default_factory=dict)
    transforms: List[TransformSpec] = field(default_factory=list)
    lambda_policy: Optional[LambdaPolicySpec] = None
    optimizer: Optional[OptimizerSpec] = None

    search: Optional[Dict[str, Any]] = None
    constraint_nonnegative: bool = True
    eval_tasks: Optional[List[str]] = None
    split: str = "test"
    save_merged: bool = False
    eval_subset: Optional[Dict[str, Any]] = None
    output_dir: Optional[Path] = None
    compute_missing_interference_baselines: bool = True

    def to_merge_spec(self) -> MergeSpec:
        return MergeSpec(
            adapters=list(self.adapters),
            method=self.method,
            merge_mode=self.merge_mode,
            method_params=dict(self.method_params),
            transforms=list(self.transforms),
            lambda_policy=self.lambda_policy,
            optimizer=self.optimizer,
        )


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PACKAGE_ROOT / resolved
    return resolved


def _warn_deprecated(message: str) -> None:
    warnings.warn(message, DeprecationWarning, stacklevel=3)


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
            raise ValueError("lambda_policy.type=per_layer requires default or method_params.lambda.")
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


def _validate_top_level_keys(payload: Mapping[str, Any]) -> None:
    allowed = _ALLOWED_TOP_LEVEL_KEYS | _LEGACY_TOP_LEVEL_KEYS
    unknown = sorted(set(payload.keys()) - allowed)
    if unknown:
        known = ", ".join(sorted(allowed))
        raise ValueError(
            "Unknown config key(s): " + ", ".join(unknown) + ". "
            + f"Allowed keys: {known}."
        )


def _coerce_search(raw_search: Any, raw_grid: Any) -> Optional[Dict[str, Any]]:
    if raw_search is not None and not isinstance(raw_search, Mapping):
        raise ValueError("search must be a mapping when provided.")

    search = dict(raw_search) if isinstance(raw_search, Mapping) else None

    if raw_grid is not None:
        _warn_deprecated(
            "Top-level 'grid' is deprecated. Move it under search.grid with search.type='grid'."
        )
        if not isinstance(raw_grid, Mapping):
            raise ValueError("grid must be a mapping of param->values.")
        if search is None:
            search = {"type": "grid", "grid": dict(raw_grid)}
        else:
            search = dict(search)
            if "type" not in search:
                search["type"] = "grid"
            if "grid" in search and search["grid"] != raw_grid:
                raise ValueError("Config provides both grid and search.grid with conflicting values.")
            search["grid"] = dict(raw_grid)

    if search is None:
        return None

    search_type = str(search.get("type", "grid")).strip().lower()
    if search_type not in {"grid", "bayes"}:
        raise ValueError("search.type must be either 'grid' or 'bayes'.")
    search["type"] = search_type

    if search_type == "grid":
        grid = search.get("grid", {})
        if not isinstance(grid, Mapping):
            raise ValueError("search.grid must be a mapping when search.type=grid.")
        search["grid"] = dict(grid)

    return search


def normalize_merge_config(payload: Mapping[str, Any]) -> MergeConfig:
    if not isinstance(payload, Mapping):
        raise ValueError("Merge config must be a mapping.")

    _validate_top_level_keys(payload)

    adapters = payload.get("adapters")
    method = payload.get("method")
    if not adapters or not method:
        raise ValueError("Config requires adapters and method.")
    if not isinstance(adapters, list):
        raise ValueError("adapters must be a list.")

    merge_mode = str(payload.get("merge_mode", "common"))
    if merge_mode not in {"common", "strict"}:
        raise ValueError("merge_mode must be either 'common' or 'strict'.")

    method_params_raw = payload.get("method_params", payload.get("params", {}))
    if "params" in payload:
        _warn_deprecated("Top-level 'params' is deprecated. Use 'method_params'.")
    if not isinstance(method_params_raw, Mapping):
        raise ValueError("method_params must be a mapping.")
    method_params = dict(method_params_raw)

    if "lambda" in payload:
        _warn_deprecated("Top-level 'lambda' is deprecated. Use method_params.lambda.")
        method_params.setdefault("lambda", payload["lambda"])

    transforms = _parse_transform_list(payload.get("transforms"))
    lambda_policy = _parse_lambda_policy(
        payload.get("lambda_policy"),
        fallback_lambda=method_params.get("lambda"),
    )
    optimizer = _parse_optimizer(payload.get("optimizer"))

    search = _coerce_search(payload.get("search"), payload.get("grid"))

    eval_tasks_raw = payload.get("eval_tasks")
    if eval_tasks_raw is not None and not isinstance(eval_tasks_raw, list):
        raise ValueError("eval_tasks must be a list when provided.")

    split = str(payload.get("split", "test"))
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be one of: train, validation, test.")

    eval_subset = payload.get("eval_subset")
    if eval_subset is not None and not isinstance(eval_subset, Mapping):
        raise ValueError("eval_subset must be a mapping when provided.")

    output_dir = payload.get("output_dir")
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = PACKAGE_ROOT / output_dir

    return MergeConfig(
        adapters=[str(x) for x in adapters],
        method=str(method),
        merge_mode=merge_mode,
        method_params=method_params,
        transforms=transforms,
        lambda_policy=lambda_policy,
        optimizer=optimizer,
        search=search,
        constraint_nonnegative=bool(payload.get("constraint_nonnegative", True)),
        eval_tasks=[str(x) for x in eval_tasks_raw] if eval_tasks_raw is not None else None,
        split=split,
        save_merged=bool(payload.get("save_merged", False)),
        eval_subset=dict(eval_subset) if isinstance(eval_subset, Mapping) else None,
        output_dir=output_dir,
        compute_missing_interference_baselines=bool(payload.get("compute_missing_interference_baselines", True)),
    )


def load_merge_config(path: str | Path) -> MergeConfig:
    resolved = _resolve_path(path)
    with resolved.open("r") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("Merge config must be a mapping.")
    return normalize_merge_config(payload)


def merge_config_from_legacy_args(
    *,
    adapters: List[str],
    method: str,
    merge_mode: str,
    lambda_weight: Optional[float],
    params: Optional[Dict[str, Any]] = None,
) -> MergeConfig:
    method_params = dict(params or {})
    if lambda_weight is not None and "lambda" not in method_params:
        method_params["lambda"] = float(lambda_weight)

    lambda_policy = _parse_lambda_policy(
        method_params.get("lambda_policy"),
        fallback_lambda=method_params.get("lambda"),
    )
    transforms = _parse_transform_list(method_params.get("transforms"))
    optimizer = _parse_optimizer(method_params.get("optimizer"))

    return MergeConfig(
        adapters=list(adapters),
        method=method,
        merge_mode=merge_mode,
        method_params=method_params,
        transforms=transforms,
        lambda_policy=lambda_policy,
        optimizer=optimizer,
    )


def merge_spec_to_params(spec: MergeSpec | MergeConfig) -> Dict[str, Any]:
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
    "MergeConfig",
    "load_merge_config",
    "normalize_merge_config",
    "merge_config_from_legacy_args",
    "merge_spec_to_params",
]
