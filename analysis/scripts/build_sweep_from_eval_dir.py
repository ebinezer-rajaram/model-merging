#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


INVALID_SCORE = -1e9


@dataclass(frozen=True)
class _Score:
    score: float
    details: Dict[str, Any]
    tiebreak: float


def _score_min_interference(results: Mapping[str, Mapping[str, Any]], constraint_nonnegative: bool) -> _Score:
    deltas: List[float] = []
    for _task, metrics in results.items():
        value = metrics.get("interference_delta")
        if isinstance(value, (int, float)):
            deltas.append(float(value))

    if not deltas:
        return _Score(score=INVALID_SCORE, details={"reason": "missing_interference_delta"}, tiebreak=float("-inf"))

    min_delta = min(deltas)
    mean_delta = sum(deltas) / len(deltas)
    if constraint_nonnegative and min_delta < 0:
        return _Score(
            score=INVALID_SCORE,
            details={"min_interference_delta": min_delta, "mean_interference_delta": mean_delta},
            tiebreak=mean_delta,
        )

    return _Score(
        score=float(min_delta),
        details={"min_interference_delta": float(min_delta), "mean_interference_delta": float(mean_delta)},
        tiebreak=float(mean_delta),
    )


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a sweep_*.json summary from existing eval_results_*.json files.")
    p.add_argument("--eval-dir", required=True, help="Directory containing eval_results_*.json files.")
    p.add_argument("--method", required=True, help="Merge method name (e.g. weighted_delta).")
    p.add_argument("--adapters", nargs="+", required=True, help="Adapter task names (e.g. emotion speaker_ver).")
    p.add_argument("--split", default="test", help="Split name (default: test).")
    p.add_argument("--merge-mode", default="common", choices=("common", "strict"), help="Merge mode to record.")
    p.add_argument(
        "--constraint-nonnegative",
        action="store_true",
        help="Disallow negative interference delta in ranking (default: True).",
    )
    p.add_argument(
        "--allow-negative",
        action="store_true",
        help="Allow negative interference delta in ranking (overrides --constraint-nonnegative).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output sweep JSON path (default: <eval-dir>/../../sweeps/sweep_<timestamp>.json).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists() or not eval_dir.is_dir():
        raise SystemExit(f"--eval-dir must be an existing directory: {eval_dir}")

    constraint_nonnegative = True
    if args.allow_negative:
        constraint_nonnegative = False
    elif args.constraint_nonnegative:
        constraint_nonnegative = True

    eval_files = sorted(eval_dir.glob("eval_results_*.json"))
    if not eval_files:
        raise SystemExit(f"No eval_results_*.json files found under {eval_dir}")

    started_at = datetime.now()
    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    if args.output:
        out_path = Path(args.output)
    else:
        # Default: artifacts/merged/<method>/<task_combo>/sweeps/sweep_<timestamp>.json
        sweeps_dir = eval_dir.parent.parent / "sweeps"
        out_path = sweeps_dir / f"sweep_{timestamp}.json"

    runs: List[Dict[str, Any]] = []
    best_idx: Optional[int] = None
    best_score = float("-inf")
    best_tiebreak = float("-inf")

    for path in eval_files:
        with path.open("r") as handle:
            obj = json.load(handle)
        if not isinstance(obj, Mapping):
            continue

        if str(obj.get("split")) != str(args.split):
            continue
        if str(obj.get("merge_method")) != str(args.method):
            continue
        source_tasks = obj.get("source_tasks")
        if not isinstance(source_tasks, list) or sorted(map(str, source_tasks)) != sorted(map(str, args.adapters)):
            continue

        lam = obj.get("lambda")
        if not isinstance(lam, (int, float)):
            # Skip entries without the primary parameter.
            continue

        results = obj.get("results")
        if not isinstance(results, Mapping):
            continue

        score_obj = _score_min_interference(results, constraint_nonnegative=constraint_nonnegative)
        entry = {
            "params": {"lambda": float(lam)},
            "score": float(score_obj.score),
            "score_details": score_obj.details,
            "results": results,
        }
        runs.append(entry)

        if score_obj.score > best_score or (score_obj.score == best_score and score_obj.tiebreak > best_tiebreak):
            best_score = float(score_obj.score)
            best_tiebreak = float(score_obj.tiebreak)
            best_idx = len(runs) - 1

    if not runs:
        raise SystemExit("No compatible eval results found to build a sweep.")

    summary: Dict[str, Any] = {
        "timestamp": started_at.isoformat(),
        "method": args.method,
        "adapters": list(args.adapters),
        "merge_mode": args.merge_mode,
        "eval_tasks": None,
        "split": args.split,
        "grid": None,
        "search": {"type": "import_eval_dir", "eval_dir": str(eval_dir)},
        "constraint_nonnegative": constraint_nonnegative,
        "best_index": best_idx,
        "best_score": best_score,
        "runs": runs,
    }

    _atomic_write_json(out_path, summary)
    print(out_path)


if __name__ == "__main__":
    main()

