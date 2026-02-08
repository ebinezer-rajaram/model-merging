"""Pre-merge adapter transform plugin system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping

import torch

from merging.config.specs import TransformSpec

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


@dataclass(frozen=True)
class TransformScaffoldNote:
    name: str
    message: str


def _ties_transform_scaffold(weights: TensorDict, params: Mapping[str, object]) -> TensorDict:
    _ = params
    # Scaffold only: keep as no-op until TIES-as-transform is intentionally implemented.
    return dict(weights)


def register_builtin_transforms() -> None:
    if "identity" not in _TRANSFORMS:
        register_transform("identity", _identity_transform)
    if "ties_scaffold" not in _TRANSFORMS:
        register_transform("ties_scaffold", _ties_transform_scaffold)


register_builtin_transforms()


__all__ = [
    "register_transform",
    "get_transform",
    "list_transforms",
    "apply_transforms",
    "TransformScaffoldNote",
]
