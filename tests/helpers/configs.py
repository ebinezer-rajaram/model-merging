from __future__ import annotations

from pathlib import Path
from typing import Any

from merging.config.unified import MergeConfig, OptimizerSpec
from merging.optimizers.registry import OptimizerContext


def minimal_merge_config(tmp_path: Path, **overrides: Any) -> MergeConfig:
    payload: dict[str, Any] = {
        "adapters": ["emotion", "intent"],
        "method": "weighted_delta_n",
        "output_dir": tmp_path,
        "split": "validation",
        "eval_tasks": ["emotion"],
        "save_merged": False,
        "constraint_nonnegative": False,
    }
    payload.update(overrides)
    return MergeConfig(**payload)


def optimizer_context(**overrides: Any) -> OptimizerContext:
    payload: dict[str, Any] = {
        "method": "weighted_delta_n",
        "adapter_specs": [],
        "adapter_paths": [],
        "source_metadata": [],
        "merge_mode": "common",
        "output_dir": None,
        "method_params": {},
        "lambda_policy": None,
    }
    payload.update(overrides)
    return OptimizerContext(**payload)


def optimizer_spec(name: str, params: dict[str, Any] | None = None) -> OptimizerSpec:
    return OptimizerSpec(type=name, params=params or {})
