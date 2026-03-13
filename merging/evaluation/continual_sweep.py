"""Shared helpers to run continual mode through the standard merge-sweep stack."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from merging.config.unified import MergeConfig
from merging.continual.engine import continual_merge_sources_to_artifact
from merging.continual.evaluate import (
    build_continual_tag,
    evaluate_continual_artifact,
    select_eval_tasks_for_sources,
)
from merging.continual.policy import ContinualMergePolicy
from merging.delta_sources.base import DeltaSource
from merging.delta_sources.resolver import resolve_delta_sources
from merging.runtime.utils import PACKAGE_ROOT


def is_continual_method(method: str) -> bool:
    return str(method).strip().lower() == "continual"


def _sanitize_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    token = token.strip("_")
    return token or "source"


def _source_label(spec: str) -> str:
    raw = str(spec).strip()
    if not raw:
        return "source"
    if "/" in raw or "\\" in raw:
        name = Path(raw).name
        return _sanitize_token(name or raw)
    return _sanitize_token(raw)


def default_continual_sweep_dir(config: MergeConfig) -> Path:
    combo = "_".join(sorted(_source_label(spec) for spec in config.adapters))
    if not combo:
        combo = "sources"
    return PACKAGE_ROOT / "artifacts" / "merged" / "continual" / combo / "sweeps"


@dataclass(frozen=True)
class ContinualSweepContext:
    x_source: DeltaSource
    y_source: DeltaSource
    eval_tasks: List[str]
    merge_mode: str
    energy_threshold: float
    store_dtype: str
    alpha_default: float
    lambda_default: float
    batch_size: Optional[int]
    use_cache: bool


def prepare_continual_context(config: MergeConfig) -> ContinualSweepContext:
    if len(config.adapters) != 2:
        raise ValueError(
            "Continual sweep requires exactly 2 adapter specs in config.adapters: [x_source, y_source]."
        )

    sources = resolve_delta_sources(config.adapters)
    x_source, y_source = sources[0], sources[1]

    tasks = select_eval_tasks_for_sources(
        x_tasks=x_source.constituent_tasks_flat(),
        y_tasks=y_source.constituent_tasks_flat(),
        explicit_eval_tasks=config.eval_tasks,
    )
    if not tasks:
        raise ValueError(
            "No evaluation tasks resolved for continual sweep. "
            "Set eval_tasks explicitly in the sweep config."
        )

    method_params = dict(config.method_params)
    energy_threshold = float(method_params.get("energy_threshold", 0.99))
    if not 0.0 < energy_threshold <= 1.0:
        raise ValueError(f"method_params.energy_threshold must be in (0,1], got {energy_threshold}")

    store_dtype = str(method_params.get("store_dtype", "float16")).strip().lower()
    if store_dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError(
            f"method_params.store_dtype must be one of float16|bfloat16|float32, got '{store_dtype}'"
        )

    alpha_default = float(method_params.get("alpha", 1.0))
    lambda_default = float(method_params.get("lambda", method_params.get("lambda_weight", 0.5)))

    batch_size_raw = method_params.get("batch_size")
    batch_size = int(batch_size_raw) if batch_size_raw is not None else None
    use_cache = bool(method_params.get("use_cache", False))

    return ContinualSweepContext(
        x_source=x_source,
        y_source=y_source,
        eval_tasks=tasks,
        merge_mode=config.merge_mode,
        energy_threshold=energy_threshold,
        store_dtype=store_dtype,
        alpha_default=alpha_default,
        lambda_default=lambda_default,
        batch_size=batch_size,
        use_cache=use_cache,
    )


def resolve_alpha_lambda(
    params: Mapping[str, Any],
    *,
    alpha_default: float,
    lambda_default: float,
) -> Tuple[float, float]:
    alpha = float(params.get("alpha", alpha_default))
    lambda_weight = float(params.get("lambda", params.get("lambda_weight", lambda_default)))
    return alpha, lambda_weight


def _format_float_token(value: float) -> str:
    token = f"{float(value):g}"
    return token.replace("-", "m").replace(".", "p")


def build_continual_run_dir(
    *,
    summary_dir: Path,
    run_index: int,
    alpha: float,
    lambda_weight: float,
) -> Path:
    run_tag = (
        f"run_{run_index:04d}_"
        f"alpha{_format_float_token(alpha)}_"
        f"lambda{_format_float_token(lambda_weight)}"
    )
    run_dir = summary_dir.parent / "runs" / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def evaluate_continual_point(
    *,
    config: MergeConfig,
    context: ContinualSweepContext,
    params: Mapping[str, Any],
    run_index: int,
    summary_dir: Path,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    alpha, lambda_weight = resolve_alpha_lambda(
        params,
        alpha_default=context.alpha_default,
        lambda_default=context.lambda_default,
    )

    policy = ContinualMergePolicy(alpha=alpha, lambda_weight=lambda_weight)
    run_dir = build_continual_run_dir(
        summary_dir=summary_dir,
        run_index=run_index,
        alpha=alpha,
        lambda_weight=lambda_weight,
    )

    merge_result = continual_merge_sources_to_artifact(
        x_source=context.x_source,
        y_source=context.y_source,
        policy=policy,
        output_dir=run_dir,
        energy_threshold=context.energy_threshold,
        merge_mode=context.merge_mode,
        store_dtype=context.store_dtype,
    )

    merge_tag = build_continual_tag(
        source_tasks=context.eval_tasks,
        alpha=alpha,
        lambda_weight=lambda_weight,
    )
    results = evaluate_continual_artifact(
        artifact_path=merge_result.artifact_dir,
        eval_tasks=context.eval_tasks,
        split=config.split,
        batch_size=context.batch_size,
        enable_cache=context.use_cache,
        show_summary=True,
        compute_missing_interference_baselines=config.compute_missing_interference_baselines,
        save_results=True,
        eval_subset=config.eval_subset,
        merge_tag=merge_tag,
        alpha=alpha,
        lambda_weight=lambda_weight,
    )

    metadata = {
        "alpha": alpha,
        "lambda": lambda_weight,
        "artifact_dir": str(merge_result.artifact_dir),
        "manifest_path": str(merge_result.manifest_path),
        "num_merged_params": int(merge_result.num_merged_params),
        "num_skipped_params": int(merge_result.num_skipped_params),
    }
    return results, metadata


def run_post_sweep_eval_for_best_continual(
    *,
    config: MergeConfig,
    context: ContinualSweepContext,
    best_params: Dict[str, Any],
    summary_dir: Path,
    score_fn: Any,
) -> Dict[str, Any]:
    post_cfg = config.post_sweep_eval
    if not isinstance(post_cfg, Mapping) or not bool(post_cfg.get("enabled", False)):
        return {"enabled": False}

    split = str(post_cfg.get("split", "test"))
    eval_tasks_raw = post_cfg.get("eval_tasks")
    eval_tasks: List[str]
    if isinstance(eval_tasks_raw, list):
        eval_tasks = [str(x) for x in eval_tasks_raw]
    else:
        eval_tasks = list(context.eval_tasks)

    alpha, lambda_weight = resolve_alpha_lambda(
        best_params,
        alpha_default=context.alpha_default,
        lambda_default=context.lambda_default,
    )
    policy = ContinualMergePolicy(alpha=alpha, lambda_weight=lambda_weight)
    run_dir = build_continual_run_dir(
        summary_dir=summary_dir,
        run_index=999999,
        alpha=alpha,
        lambda_weight=lambda_weight,
    )

    merge_result = continual_merge_sources_to_artifact(
        x_source=context.x_source,
        y_source=context.y_source,
        policy=policy,
        output_dir=run_dir,
        energy_threshold=context.energy_threshold,
        merge_mode=context.merge_mode,
        store_dtype=context.store_dtype,
    )

    merge_tag = build_continual_tag(
        source_tasks=eval_tasks,
        alpha=alpha,
        lambda_weight=lambda_weight,
    )
    results = evaluate_continual_artifact(
        artifact_path=merge_result.artifact_dir,
        eval_tasks=eval_tasks,
        split=split,
        batch_size=context.batch_size,
        enable_cache=context.use_cache,
        show_summary=True,
        compute_missing_interference_baselines=config.compute_missing_interference_baselines,
        save_results=True,
        merge_tag=merge_tag,
        alpha=alpha,
        lambda_weight=lambda_weight,
    )
    score, score_details = score_fn(results, config.constraint_nonnegative)
    return {
        "enabled": True,
        "split": split,
        "save_merged": True,
        "params": dict(best_params),
        "score": float(score),
        "score_details": score_details,
        "results": results,
        "artifact_dir": str(merge_result.artifact_dir),
        "manifest_path": str(merge_result.manifest_path),
    }


__all__ = [
    "ContinualSweepContext",
    "build_continual_run_dir",
    "default_continual_sweep_dir",
    "evaluate_continual_point",
    "is_continual_method",
    "prepare_continual_context",
    "resolve_alpha_lambda",
    "run_post_sweep_eval_for_best_continual",
]
