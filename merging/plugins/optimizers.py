"""Lambda optimizer plugin scaffolding for merge workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from merging.config.specs import LambdaPolicySpec, MergeSpec, OptimizerSpec


@dataclass(frozen=True)
class OptimizerContext:
    method: str
    adapter_specs: List[str]
    adapter_paths: List[Path]
    source_metadata: List[Dict[str, Any]]
    merge_mode: str
    output_dir: Optional[Path]
    method_params: Dict[str, Any]
    lambda_policy: Optional[LambdaPolicySpec]


@dataclass(frozen=True)
class OptimizerResult:
    lambda_policy: Optional[LambdaPolicySpec]
    provenance: Dict[str, Any]
    method_params_overrides: Dict[str, Any] = field(default_factory=dict)


class BaseOptimizer:
    name: str = "base"

    def optimize(self, spec: MergeSpec, context: OptimizerContext) -> OptimizerResult:  # pragma: no cover
        raise NotImplementedError


_OPTIMIZERS: Dict[str, BaseOptimizer] = {}


def register_optimizer(optimizer: BaseOptimizer) -> None:
    key = optimizer.name.strip().lower()
    if not key:
        raise ValueError("Optimizer name must be non-empty.")
    if key in _OPTIMIZERS:
        raise ValueError(f"Optimizer already registered: {key}")
    _OPTIMIZERS[key] = optimizer


def get_optimizer(name: str) -> BaseOptimizer:
    key = name.strip().lower()
    if key not in _OPTIMIZERS:
        available = ", ".join(sorted(_OPTIMIZERS.keys()))
        raise ValueError(f"Unknown optimizer '{name}'. Available: {available}")
    return _OPTIMIZERS[key]


def list_optimizers() -> List[str]:
    return sorted(_OPTIMIZERS.keys())


class NoneOptimizer(BaseOptimizer):
    name = "none"

    def optimize(self, spec: MergeSpec, context: OptimizerContext) -> OptimizerResult:
        _ = context
        return OptimizerResult(
            lambda_policy=spec.lambda_policy,
            provenance={"optimizer": "none", "status": "noop"},
            method_params_overrides={},
        )


class BayesOptimizerAdapter(BaseOptimizer):
    name = "bayes"

    def optimize(self, spec: MergeSpec, context: OptimizerContext) -> OptimizerResult:
        # Scaffold adapter:
        # - merge-sweep already owns bayes search as the executable flow.
        # - direct merge command can still carry optimizer metadata and
        #   use provided lambda policy/method params.
        _ = context
        return OptimizerResult(
            lambda_policy=spec.lambda_policy,
            provenance={
                "optimizer": "bayes",
                "status": "scaffold_passthrough",
                "note": "Use main.py merge-sweep for active bayes optimization.",
            },
            method_params_overrides={},
        )


class AdaMergingOptimizer(BaseOptimizer):
    name = "adamerging"

    def optimize(self, spec: MergeSpec, context: OptimizerContext) -> OptimizerResult:
        from merging.plugins.adamerging_engine import run_adamerging_optimizer

        return run_adamerging_optimizer(spec=spec, context=context)


def resolve_optimizer(spec: Optional[OptimizerSpec]) -> BaseOptimizer:
    if spec is None:
        return get_optimizer("none")
    return get_optimizer(spec.type or "none")


def optimize_lambda_policy(spec: MergeSpec, context: OptimizerContext) -> OptimizerResult:
    optimizer = resolve_optimizer(spec.optimizer)
    return optimizer.optimize(spec, context)


def apply_optimizer_overrides(
    method_params: Mapping[str, Any],
    result: OptimizerResult,
) -> Dict[str, Any]:
    merged = dict(method_params)
    merged.update(result.method_params_overrides or {})
    return merged


def register_builtin_optimizers() -> None:
    if "none" not in _OPTIMIZERS:
        register_optimizer(NoneOptimizer())
    if "bayes" not in _OPTIMIZERS:
        register_optimizer(BayesOptimizerAdapter())
    if "adamerging" not in _OPTIMIZERS:
        register_optimizer(AdaMergingOptimizer())


register_builtin_optimizers()


__all__ = [
    "OptimizerContext",
    "OptimizerResult",
    "BaseOptimizer",
    "get_optimizer",
    "list_optimizers",
    "resolve_optimizer",
    "optimize_lambda_policy",
    "apply_optimizer_overrides",
]
