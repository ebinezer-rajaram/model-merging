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
from merging.core.registry import get_merge_method
from merging.core.utils import PACKAGE_ROOT


@dataclass(frozen=True)
class SweepConfig:
    adapters: List[str]
    method: str
    grid: Dict[str, List[Any]] | None = None
    search: Optional[Dict[str, Any]] = None
    merge_mode: str = "common"
    eval_tasks: Optional[List[str]] = None
    split: str = "test"
    save_merged: bool = False
    constraint_nonnegative: bool = True
    output_dir: Optional[Path] = None


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
        merge_mode=data.get("merge_mode", "common"),
        eval_tasks=data.get("eval_tasks"),
        split=data.get("split", "test"),
        save_merged=bool(data.get("save_merged", False)),
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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

    for idx, params in enumerate(params_grid, 1):
        method_impl = get_merge_method(config.method)
        method_impl.validate(len(config.adapters), params)

        print(f"\n[{idx}/{len(params_grid)}] Evaluating params: {params}")
        results = evaluate_merged_adapter(
            adapter_path=None,
            method=config.method,
            task_names=config.adapters,
            params=params,
            eval_tasks=config.eval_tasks,
            split=config.split,
            save_merged=config.save_merged,
            save_results=True,
            show_summary=True,
            merge_mode=config.merge_mode,
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

    summary = {
        "timestamp": datetime.now().isoformat(),
        "method": config.method,
        "adapters": config.adapters,
        "merge_mode": config.merge_mode,
        "eval_tasks": config.eval_tasks,
        "split": config.split,
        "grid": config.grid,
        "search": config.search,
        "constraint_nonnegative": config.constraint_nonnegative,
        "best_index": best_idx,
        "best_score": best_score,
        "runs": runs,
    }

    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\n💾 Sweep summary saved to {summary_path}")
    if best_idx is not None:
        best_params = runs[best_idx]["params"]
        print(f"🏆 Best params: {best_params} (score={best_score:.4f})")

    return summary
