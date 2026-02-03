"""Registry for merge methods and shared merge metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Optional
from pathlib import Path

import torch


@dataclass(frozen=True)
class MergeOutput:
    merged_delta: Dict[str, torch.Tensor]
    merged_weights: Optional[Dict[str, torch.Tensor]]
    metadata: Dict


@dataclass(frozen=True)
class MergeMethod:
    name: str
    requires_lambda: bool
    min_adapters: int
    max_adapters: Optional[int]
    saveable: bool
    merge_in_memory: Callable[..., MergeOutput]
    save_fn: Optional[Callable[..., Path]] = None

    def validate(self, num_adapters: int, lambda_weight: Optional[float]) -> None:
        if num_adapters < self.min_adapters:
            raise ValueError(
                f"{self.name} requires at least {self.min_adapters} adapters, got {num_adapters}"
            )
        if self.max_adapters is not None and num_adapters > self.max_adapters:
            raise ValueError(
                f"{self.name} supports at most {self.max_adapters} adapters, got {num_adapters}"
            )
        if self.requires_lambda and lambda_weight is None:
            raise ValueError(f"{self.name} requires lambda_weight.")


_REGISTRY: Dict[str, MergeMethod] = {}


def register_merge_method(method: MergeMethod) -> None:
    if method.name in _REGISTRY:
        raise ValueError(f"Merge method already registered: {method.name}")
    _REGISTRY[method.name] = method


def get_merge_method(name: str) -> MergeMethod:
    if not _REGISTRY:
        from merging.core import methods as _methods  # noqa: F401
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown merge method: {name}. Available: {available}")
    return _REGISTRY[name]


def list_merge_methods() -> list[str]:
    if not _REGISTRY:
        from merging.core import methods as _methods  # noqa: F401
    return sorted(_REGISTRY.keys())


def build_merge_metadata(
    *,
    method: str,
    merge_mode: str,
    num_adapters: int,
    source_metadata: list[Dict],
    num_parameters: int,
    lambda_weight: Optional[float] = None,
) -> Dict:
    metadata = {
        "merge_method": method,
        "merge_mode": merge_mode,
        "num_adapters": num_adapters,
        "timestamp": datetime.now().isoformat(),
        "source_adapters": source_metadata,
        "num_parameters": num_parameters,
    }
    if lambda_weight is not None:
        metadata["lambda"] = lambda_weight
    return metadata
