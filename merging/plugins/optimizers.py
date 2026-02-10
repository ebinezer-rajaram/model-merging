"""Lambda optimizer plugin scaffolding for merge workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from merging.config.specs import LambdaPolicySpec, MergeSpec, OptimizerSpec
from merging.runtime.utils import PACKAGE_ROOT


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


def _resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = PACKAGE_ROOT / path
    return path


def _iter_sweep_json_paths(raw: str | Path) -> List[Path]:
    """Resolve a sweep file/dir/glob to concrete sweep_*.json paths."""
    import glob

    raw_str = str(raw)
    if any(ch in raw_str for ch in ("*", "?", "[")):
        return sorted(Path(p) for p in glob.glob(str(_resolve_path(raw_str))))

    path = _resolve_path(raw_str)
    if path.is_dir():
        return sorted(path.glob("sweep_*.json"))
    return [path]


def _load_json_object(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return None
    try:
        with path.open("r") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _task_names_from_source_metadata(source_metadata: List[Dict[str, Any]], *, fallback_count: int) -> List[str]:
    names: List[str] = []
    for i, meta in enumerate(source_metadata):
        task = meta.get("task")
        names.append(str(task) if task else f"adapter{i}")
    if not names:
        names = [f"adapter{i}" for i in range(fallback_count)]
    return names


def _sweep_looks_compatible(
    sweep: Mapping[str, Any],
    *,
    method: str,
    adapter_names: List[str],
    merge_mode: str,
) -> bool:
    sweep_method = str(sweep.get("method", ""))
    if sweep_method != method:
        return False

    sweep_adapters_raw = sweep.get("adapters")
    if not isinstance(sweep_adapters_raw, list):
        return False
    sweep_adapters = [str(x) for x in sweep_adapters_raw]
    if sorted(sweep_adapters) != sorted(adapter_names):
        return False

    sweep_mode = str(sweep.get("merge_mode", "common"))
    if sweep_mode != merge_mode:
        return False

    return True


def _extract_best_run_from_sweep(sweep: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    runs = sweep.get("runs")
    if not isinstance(runs, list) or not runs:
        return None

    best_index = sweep.get("best_index")
    if isinstance(best_index, int) and 0 <= best_index < len(runs):
        run = runs[best_index]
        return run if isinstance(run, dict) else None

    best_run: Optional[Dict[str, Any]] = None
    best_score = float("-inf")
    for run in runs:
        if not isinstance(run, dict):
            continue
        try:
            score = float(run.get("score", float("-inf")))
        except Exception:
            continue
        if score > best_score:
            best_run = run
            best_score = score
    return best_run


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
        params = dict(spec.optimizer.params if spec.optimizer is not None else {})

        explicit_best = params.get("best_params")
        if explicit_best is not None:
            if not isinstance(explicit_best, Mapping):
                raise ValueError("optimizer.params.best_params must be a mapping when provided.")
            overrides = dict(explicit_best)
            return OptimizerResult(
                lambda_policy=spec.lambda_policy,
                provenance={
                    "optimizer": "bayes",
                    "status": "applied_explicit_best_params",
                    "best_params": dict(overrides),
                },
                method_params_overrides=overrides,
            )

        # Support reusing summaries produced by merging/evaluation/bayes.py.
        sweep_paths = params.get("sweep_paths", params.get("warm_start_sweeps"))
        if sweep_paths is None:
            return OptimizerResult(
                lambda_policy=spec.lambda_policy,
                provenance={
                    "optimizer": "bayes",
                    "status": "scaffold_passthrough",
                    "note": "Provide optimizer.params.best_params or optimizer.params.sweep_paths to apply Bayes results.",
                },
                method_params_overrides={},
            )

        if isinstance(sweep_paths, (str, Path)):
            sweep_paths = [sweep_paths]
        if not isinstance(sweep_paths, list) or not all(isinstance(p, (str, Path)) for p in sweep_paths):
            raise ValueError("optimizer.params.sweep_paths must be a path string or list of path strings.")

        task_names = _task_names_from_source_metadata(
            context.source_metadata,
            fallback_count=len(context.adapter_specs),
        )
        compatible: List[Dict[str, Any]] = []
        total_read = 0

        for raw in sweep_paths:
            for path in _iter_sweep_json_paths(raw):
                payload = _load_json_object(path)
                if payload is None:
                    continue
                total_read += 1
                if not _sweep_looks_compatible(
                    payload,
                    method=context.method,
                    adapter_names=task_names,
                    merge_mode=context.merge_mode,
                ):
                    continue
                run = _extract_best_run_from_sweep(payload)
                if not isinstance(run, dict):
                    continue
                params_raw = run.get("params")
                if not isinstance(params_raw, Mapping):
                    continue
                try:
                    score = float(run.get("score", float("-inf")))
                except Exception:
                    score = float("-inf")
                compatible.append(
                    {
                        "path": str(path),
                        "score": score,
                        "params": dict(params_raw),
                    }
                )

        if not compatible:
            return OptimizerResult(
                lambda_policy=spec.lambda_policy,
                provenance={
                    "optimizer": "bayes",
                    "status": "no_compatible_sweep_found",
                    "sweep_paths": [str(p) for p in sweep_paths],
                    "files_read": total_read,
                    "expected": {
                        "method": context.method,
                        "adapters": task_names,
                        "merge_mode": context.merge_mode,
                    },
                },
                method_params_overrides={},
            )

        compatible.sort(key=lambda x: x["score"], reverse=True)
        best = compatible[0]
        overrides = dict(best["params"])
        return OptimizerResult(
            lambda_policy=spec.lambda_policy,
            provenance={
                "optimizer": "bayes",
                "status": "applied_from_sweep",
                "selected_sweep": best["path"],
                "selected_score": best["score"],
                "candidates": len(compatible),
                "files_read": total_read,
            },
            method_params_overrides=overrides,
        )


class AdaMergingOptimizer(BaseOptimizer):
    name = "adamerging"

    def optimize(self, spec: MergeSpec, context: OptimizerContext) -> OptimizerResult:
        from merging.plugins.adamerging_engine import run_adamerging_optimizer

        return run_adamerging_optimizer(spec=spec, context=context)


class GradientOptimizer(BaseOptimizer):
    name = "gradient"

    def optimize(self, spec: MergeSpec, context: OptimizerContext) -> OptimizerResult:
        from merging.plugins.gradient_engine import run_gradient_optimizer

        return run_gradient_optimizer(spec=spec, context=context)


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
    if "gradient" not in _OPTIMIZERS:
        register_optimizer(GradientOptimizer())


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
