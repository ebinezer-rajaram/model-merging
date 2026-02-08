"""Lambda policy abstractions for weighted merging."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from merging.config.specs import LambdaPolicySpec

_LAYER_PATTERN = re.compile(r"\.layers\.(\d+)\.")


def extract_layer_index(param_key: str) -> Optional[int]:
    match = _LAYER_PATTERN.search(param_key)
    if match:
        return int(match.group(1))
    return None


@dataclass(frozen=True)
class BaseLambdaPolicy:
    def lambda_for_key(self, key: str) -> float:  # pragma: no cover - interface only
        raise NotImplementedError

    def describe(self) -> Dict[str, object]:
        raise NotImplementedError


@dataclass(frozen=True)
class ScalarLambdaPolicy(BaseLambdaPolicy):
    value: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Scalar lambda must be in [0,1], got {self.value}")

    def lambda_for_key(self, key: str) -> float:
        return self.value

    def describe(self) -> Dict[str, object]:
        return {"type": "scalar", "value": self.value}


@dataclass(frozen=True)
class PerLayerLambdaPolicy(BaseLambdaPolicy):
    default: float
    overrides: Mapping[int, float]

    def __post_init__(self) -> None:
        if not 0.0 <= self.default <= 1.0:
            raise ValueError(f"Per-layer default lambda must be in [0,1], got {self.default}")
        for layer, value in self.overrides.items():
            if layer < 0:
                raise ValueError(f"Layer indices must be >= 0, got {layer}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Per-layer lambda must be in [0,1], got layer={layer} value={value}")

    def lambda_for_key(self, key: str) -> float:
        layer_idx = extract_layer_index(key)
        if layer_idx is None:
            return self.default
        return float(self.overrides.get(layer_idx, self.default))

    def describe(self) -> Dict[str, object]:
        return {
            "type": "per_layer",
            "default": self.default,
            "overrides": {int(k): float(v) for k, v in self.overrides.items()},
        }


def build_lambda_policy(spec: Optional[LambdaPolicySpec], fallback_lambda: Optional[float] = None) -> BaseLambdaPolicy:
    if spec is None:
        if fallback_lambda is None:
            raise ValueError("Missing lambda policy and fallback lambda.")
        return ScalarLambdaPolicy(value=float(fallback_lambda))

    policy_type = spec.type.lower()
    if policy_type == "scalar":
        value = spec.value if spec.value is not None else fallback_lambda
        if value is None:
            raise ValueError("scalar lambda policy requires value.")
        return ScalarLambdaPolicy(value=float(value))
    if policy_type == "per_layer":
        default = spec.default if spec.default is not None else fallback_lambda
        if default is None:
            raise ValueError("per_layer lambda policy requires default.")
        return PerLayerLambdaPolicy(default=float(default), overrides=dict(spec.overrides))

    raise ValueError(f"Unsupported lambda policy type: {spec.type}")


__all__ = [
    "BaseLambdaPolicy",
    "ScalarLambdaPolicy",
    "PerLayerLambdaPolicy",
    "build_lambda_policy",
    "extract_layer_index",
]
