"""Backward-compatible exports for merge config/spec models.

Use `merging.config.unified` for new code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings

from merging.config.unified import (
    LambdaPolicySpec,
    MergeConfig,
    MergeSpec,
    OptimizerSpec,
    TransformSpec,
    load_merge_config,
    merge_config_from_legacy_args,
    merge_spec_to_params,
)


def load_merge_spec(path: str | Path) -> MergeSpec:
    """Deprecated alias that now loads via unified config."""
    warnings.warn(
        "load_merge_spec is deprecated; use load_merge_config(...).to_merge_spec().",
        DeprecationWarning,
        stacklevel=2,
    )
    return load_merge_config(path).to_merge_spec()


def merge_spec_from_legacy_args(
    *,
    adapters: List[str],
    method: str,
    merge_mode: str,
    lambda_weight: Optional[float],
    params: Optional[Dict[str, Any]] = None,
) -> MergeSpec:
    config = merge_config_from_legacy_args(
        adapters=adapters,
        method=method,
        merge_mode=merge_mode,
        lambda_weight=lambda_weight,
        params=params,
    )
    return config.to_merge_spec()


__all__ = [
    "TransformSpec",
    "LambdaPolicySpec",
    "OptimizerSpec",
    "MergeSpec",
    "MergeConfig",
    "load_merge_spec",
    "merge_spec_from_legacy_args",
    "merge_spec_to_params",
]
