"""Sweep utilities for merge hyperparameters and ranking."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
import warnings

import json

from merging.config.unified import MergeConfig, LambdaPolicySpec, OptimizerSpec, load_merge_config, normalize_merge_config
from merging.evaluation.evaluate import evaluate_merged_adapter
from merging.engine.registry import get_merge_method, normalize_params
from merging.runtime.utils import PACKAGE_ROOT


@dataclass(frozen=True)
class SweepConfig:
    adapters: List[str]
    method: str
    grid: Dict[str, List[Any]] | None = None
    search: Optional[Dict[str, Any]] = None
    lambda_policy: Optional[Dict[str, Any]] = None
    optimizer: Optional[Dict[str, Any]] = None
    eval_subset: Optional[Dict[str, Any]] = None
    merge_mode: str = "common"
    eval_tasks: Optional[List[str]] = None
    split: str = "test"
    save_merged: bool = False
    compute_missing_interference_baselines: bool = True
    constraint_nonnegative: bool = True
    output_dir: Optional[Path] = None

    def to_merge_config(self) -> MergeConfig:
        payload: Dict[str, Any] = {
            "adapters": list(self.adapters),
            "method": self.method,
            "merge_mode": self.merge_mode,
            "search": dict(self.search) if isinstance(self.search, Mapping) else None,
            "eval_tasks": list(self.eval_tasks) if self.eval_tasks is not None else None,
            "split": self.split,
            "save_merged": self.save_merged,
            "constraint_nonnegative": self.constraint_nonnegative,
            "eval_subset": dict(self.eval_subset) if isinstance(self.eval_subset, Mapping) else None,
            "output_dir": str(self.output_dir) if self.output_dir is not None else None,
            "compute_missing_interference_baselines": self.compute_missing_interference_baselines,
        }
        if self.grid is not None:
            payload["grid"] = dict(self.grid)
        if self.lambda_policy is not None:
            payload["lambda_policy"] = dict(self.lambda_policy)
        if self.optimizer is not None:
            payload["optimizer"] = dict(self.optimizer)
        return normalize_merge_config(payload)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def load_sweep_config(path: Path) -> SweepConfig:
    warnings.warn(
        "load_sweep_config is deprecated; use merging.config.load_merge_config.",
        DeprecationWarning,
        stacklevel=2,
    )
    config = load_merge_config(path)
    return SweepConfig(
        adapters=list(config.adapters),
        method=config.method,
        grid=(dict(config.search.get("grid", {})) if isinstance(config.search, Mapping) else None),
        search=dict(config.search) if isinstance(config.search, Mapping) else None,
        lambda_policy=_lambda_policy_mapping(config.lambda_policy),
        optimizer=_optimizer_mapping(config.optimizer),
        eval_subset=(dict(config.eval_subset) if config.eval_subset is not None else None),
        merge_mode=config.merge_mode,
        eval_tasks=(list(config.eval_tasks) if config.eval_tasks is not None else None),
        split=config.split,
        save_merged=config.save_merged,
        compute_missing_interference_baselines=config.compute_missing_interference_baselines,
        constraint_nonnegative=config.constraint_nonnegative,
        output_dir=config.output_dir,
    )


def _expand_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    combos = []
    for combo in product(*values):
        combos.append({k: v for k, v in zip(keys, combo)})
    return combos


def _score_min_interference(
    results: Dict[str, Dict],
    constraint_nonnegative: bool,
) -> Tuple[float, Dict[str, Any]]:
    deltas = {}
    for task, metrics in results.items():
        value = metrics.get("interference_delta")
        if isinstance(value, (int, float)):
            deltas[task] = float(value)

    if not deltas:
        return float("-inf"), {"reason": "missing_interference_delta"}

    min_delta = min(deltas.values())
    mean_delta = sum(deltas.values()) / len(deltas)
    if constraint_nonnegative and min_delta < 0:
        return float("-inf"), {"min_interference_delta": min_delta, "mean_interference_delta": mean_delta}

    return min_delta, {"min_interference_delta": min_delta, "mean_interference_delta": mean_delta}


def _lambda_policy_mapping(spec: Optional[LambdaPolicySpec | Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if spec is None:
        return None
    if isinstance(spec, dict):
        return dict(spec)
    return {
        "type": spec.type,
        "value": spec.value,
        "default": spec.default,
        "overrides": dict(spec.overrides),
    }


def _optimizer_mapping(spec: Optional[OptimizerSpec | Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if spec is None:
        return None
    if isinstance(spec, dict):
        return dict(spec)
    return {"type": spec.type, "params": dict(spec.params)}


def run_sweep(config: MergeConfig | SweepConfig) -> Dict[str, Any]:
    if isinstance(config, SweepConfig):
        warnings.warn(
            "SweepConfig is deprecated; pass MergeConfig from merging.config.load_merge_config.",
            DeprecationWarning,
            stacklevel=2,
        )
        config = config.to_merge_config()

    search = config.search or {"type": "grid", "grid": {}}
    search_type = str(search.get("type", "grid")).lower()
    lambda_policy = _lambda_policy_mapping(config.lambda_policy)
    optimizer = _optimizer_mapping(config.optimizer)

    if search_type != "grid":
        from merging.evaluation.bayes import run_bayes_search

        return run_bayes_search(config, search)

    grid = search.get("grid", {})
    params_grid = _expand_grid(grid)
    if not params_grid:
        params_grid = [{}]

    started_at = datetime.now()
    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    summary_dir = config.output_dir
    if summary_dir is None:
        summary_dir = (
            PACKAGE_ROOT
            / "artifacts"
            / "merged"
            / config.method
            / "_".join(sorted(config.adapters))
            / "sweeps"
        )
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"sweep_{timestamp}.json"

    runs: List[Dict[str, Any]] = []
    best_idx = None
    best_score = float("-inf")
    best_tiebreak = float("-inf")

    summary: Dict[str, Any] = {
        "timestamp": started_at.isoformat(),
        "method": config.method,
        "adapters": config.adapters,
        "merge_mode": config.merge_mode,
        "lambda_policy": lambda_policy,
        "optimizer": optimizer,
        "eval_tasks": config.eval_tasks,
        "split": config.split,
        "eval_subset": config.eval_subset,
        "grid": search.get("grid", {}),
        "search": search,
        "constraint_nonnegative": config.constraint_nonnegative,
        "best_index": best_idx,
        "best_score": best_score,
        "runs": runs,
    }
    _atomic_write_json(summary_path, summary)

    try:
        for idx, params in enumerate(params_grid, 1):
            method_impl = get_merge_method(config.method)
            effective_params = normalize_params(method_impl, params=params)
            if lambda_policy is not None:
                effective_params["lambda_policy"] = lambda_policy
            if optimizer is not None:
                effective_params["optimizer"] = optimizer
            method_impl.validate(len(config.adapters), effective_params)

            print(f"\n[{idx}/{len(params_grid)}] Evaluating params: {params}")
            results = evaluate_merged_adapter(
                adapter_path=None,
                method=config.method,
                task_names=config.adapters,
                params=effective_params,
                eval_tasks=config.eval_tasks,
                split=config.split,
                save_merged=config.save_merged,
                save_results=True,
                show_summary=True,
                merge_mode=config.merge_mode,
                compute_missing_interference_baselines=config.compute_missing_interference_baselines,
                eval_subset=config.eval_subset,
            )

            score, stats = _score_min_interference(results, config.constraint_nonnegative)
            run_entry = {
                "params": params,
                "score": score,
                "score_details": stats,
                "results": results,
            }
            runs.append(run_entry)

            tiebreak = stats.get("mean_interference_delta", float("-inf"))
            if score > best_score or (score == best_score and tiebreak > best_tiebreak):
                best_score = score
                best_tiebreak = tiebreak
                best_idx = idx - 1
                print(f"🏆 Best so far: params={params} score={best_score:.4f}")

            summary["best_index"] = best_idx
            summary["best_score"] = best_score
            _atomic_write_json(summary_path, summary)
    except KeyboardInterrupt:
        summary["best_index"] = best_idx
        summary["best_score"] = best_score
        _atomic_write_json(summary_path, summary)
        print(f"\n⏹️  Sweep interrupted. Partial summary saved to {summary_path}")
        if best_idx is not None and runs:
            best_params = runs[best_idx]["params"]
            print(f"🏆 Best params so far: {best_params} (score={best_score:.4f})")
        return summary

    print(f"\n💾 Sweep summary saved to {summary_path}")
    if best_idx is not None:
        best_params = runs[best_idx]["params"]
        print(f"🏆 Best params: {best_params} (score={best_score:.4f})")

    return summary
