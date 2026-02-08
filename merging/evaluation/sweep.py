"""Sweep utilities for merge hyperparameters and ranking."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import json
import yaml

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


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def load_sweep_config(path: Path) -> SweepConfig:
    with path.open("r") as handle:
        data = yaml.safe_load(handle) or {}

    adapters = data.get("adapters")
    method = data.get("method")
    grid = data.get("grid")
    search = data.get("search")
    if not adapters or not method:
        raise ValueError("Sweep config must include 'adapters' and 'method'.")

    output_dir = data.get("output_dir")
    if output_dir:
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = PACKAGE_ROOT / output_dir

    return SweepConfig(
        adapters=adapters,
        method=method,
        grid=grid,
        search=search,
        lambda_policy=data.get("lambda_policy"),
        optimizer=data.get("optimizer"),
        eval_subset=data.get("eval_subset"),
        merge_mode=data.get("merge_mode", "common"),
        eval_tasks=data.get("eval_tasks"),
        split=data.get("split", "test"),
        save_merged=bool(data.get("save_merged", False)),
        compute_missing_interference_baselines=bool(data.get("compute_missing_interference_baselines", True)),
        constraint_nonnegative=bool(data.get("constraint_nonnegative", True)),
        output_dir=output_dir,
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


def run_sweep(config: SweepConfig) -> Dict[str, Any]:
    search = config.search or {"type": "grid", "grid": config.grid or {}}
    search_type = str(search.get("type", "grid")).lower()

    if search_type != "grid":
        from merging.evaluation.bayes import run_bayes_search

        return run_bayes_search(config, search)

    grid = search.get("grid", config.grid or {})
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
        "lambda_policy": config.lambda_policy,
        "optimizer": config.optimizer,
        "eval_tasks": config.eval_tasks,
        "split": config.split,
        "eval_subset": config.eval_subset,
        "grid": config.grid,
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
            if config.lambda_policy is not None:
                effective_params["lambda_policy"] = config.lambda_policy
            if config.optimizer is not None:
                effective_params["optimizer"] = config.optimizer
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
