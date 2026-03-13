"""CLI entrypoints for continual compressed merge workflows."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from merging.continual.engine import (
    continual_merge_sources_to_artifact,
    materialize_existing_merge_to_artifact,
)
from merging.continual.evaluate import (
    build_continual_tag,
    evaluate_continual_artifact,
    select_eval_tasks_for_sources,
)
from merging.continual.policy import ContinualMergePolicy
from merging.delta_sources.resolver import resolve_delta_source


def _parse_float_list(values: Optional[Sequence[str]], *, default: Optional[List[float]] = None) -> List[float]:
    if values is None:
        return list(default or [])
    parsed: List[float] = []
    for raw in values:
        for token in str(raw).split(","):
            token = token.strip()
            if not token:
                continue
            parsed.append(float(token))
    return parsed


def _format_float_token(value: float) -> str:
    token = f"{float(value):g}"
    return token.replace("-", "m").replace(".", "p")


def materialize_merged_artifact_from_args(args) -> Path:
    output_dir = Path(args.output).expanduser() if args.output else None
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            Path("artifacts")
            / "continual"
            / "materialized"
            / f"run_{timestamp}"
        )

    result = materialize_existing_merge_to_artifact(
        merged_run_path=Path(args.merged_run_path).expanduser(),
        output_dir=output_dir,
        energy_threshold=float(args.energy_threshold),
        merge_mode=args.merge_mode,
        store_dtype=args.store_dtype,
    )
    print(f"✅ Materialized continual artifact at {result.artifact_dir}")
    print(f"   Manifest: {result.manifest_path}")
    print(f"   Merged params: {result.num_merged_params}")
    if result.num_skipped_params:
        print(f"   Skipped params: {result.num_skipped_params} ({result.skipped_reasons})")
    return result.artifact_dir


def continual_merge_from_args(args) -> Path:
    x_source = resolve_delta_source(args.x_source)
    y_source = resolve_delta_source(args.y_source)

    output_dir = Path(args.output).expanduser() if args.output else None
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("artifacts") / "continual" / "merged" / f"run_{timestamp}"

    policy = ContinualMergePolicy(alpha=float(args.alpha), lambda_weight=float(args.lambda_weight))
    result = continual_merge_sources_to_artifact(
        x_source=x_source,
        y_source=y_source,
        policy=policy,
        output_dir=output_dir,
        energy_threshold=float(args.energy_threshold),
        merge_mode=args.merge_mode,
        store_dtype=args.store_dtype,
    )
    print(f"✅ Continual merge artifact created at {result.artifact_dir}")
    print(f"   Manifest: {result.manifest_path}")
    print(f"   Merged params: {result.num_merged_params}")
    if result.num_skipped_params:
        print(f"   Skipped params: {result.num_skipped_params} ({result.skipped_reasons})")
    return result.artifact_dir


def evaluate_continual_from_args(args) -> Dict[str, Any]:
    if args.artifact_path:
        results = evaluate_continual_artifact(
            artifact_path=Path(args.artifact_path).expanduser(),
            eval_tasks=list(args.eval_tasks) if args.eval_tasks else None,
            split=args.split,
            batch_size=args.batch_size,
            enable_cache=bool(args.use_cache),
            show_summary=True,
            compute_missing_interference_baselines=bool(args.compute_missing_interference_baselines),
            save_results=bool(args.save_results),
            alpha=args.alpha,
            lambda_weight=args.lambda_weight,
        )
        return {
            "mode": "single",
            "results": results,
        }

    if not args.x_source or not args.y_source:
        raise ValueError("Sweep mode requires --x-source and --y-source when --artifact-path is not provided.")

    alphas = _parse_float_list(args.alpha_values, default=[args.alpha if args.alpha is not None else 1.0])
    lambdas = _parse_float_list(args.lambda_values, default=[args.lambda_weight if args.lambda_weight is not None else 0.5])
    if not alphas or not lambdas:
        raise ValueError("Sweep mode requires at least one alpha and one lambda value.")

    x_source = resolve_delta_source(args.x_source)
    y_source = resolve_delta_source(args.y_source)

    out_root = Path(args.output).expanduser() if args.output else None
    if out_root is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = Path("artifacts") / "continual" / "sweeps" / f"sweep_{timestamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    eval_tasks = select_eval_tasks_for_sources(
        x_tasks=x_source.constituent_tasks_flat(),
        y_tasks=y_source.constituent_tasks_flat(),
        explicit_eval_tasks=args.eval_tasks,
    )

    rows: List[Dict[str, Any]] = []
    best_idx: Optional[int] = None
    best_score = float("-inf")
    best_tiebreak = float("-inf")

    total = len(alphas) * len(lambdas)
    counter = 0
    for alpha in alphas:
        for lambda_weight in lambdas:
            counter += 1
            run_tag = f"run_alpha{_format_float_token(alpha)}_lambda{_format_float_token(lambda_weight)}"
            run_dir = out_root / "runs" / run_tag
            run_dir.mkdir(parents=True, exist_ok=True)

            policy = ContinualMergePolicy(alpha=float(alpha), lambda_weight=float(lambda_weight))
            merge_result = continual_merge_sources_to_artifact(
                x_source=x_source,
                y_source=y_source,
                policy=policy,
                output_dir=run_dir,
                energy_threshold=float(args.energy_threshold),
                merge_mode=args.merge_mode,
                store_dtype=args.store_dtype,
            )

            merge_tag = build_continual_tag(
                source_tasks=select_eval_tasks_for_sources(
                    x_tasks=x_source.constituent_tasks_flat(),
                    y_tasks=y_source.constituent_tasks_flat(),
                    explicit_eval_tasks=None,
                ),
                alpha=float(alpha),
                lambda_weight=float(lambda_weight),
            )
            results = evaluate_continual_artifact(
                artifact_path=merge_result.artifact_dir,
                eval_tasks=eval_tasks,
                split=args.split,
                batch_size=args.batch_size,
                enable_cache=bool(args.use_cache),
                show_summary=True,
                compute_missing_interference_baselines=bool(args.compute_missing_interference_baselines),
                save_results=bool(args.save_results),
                merge_tag=merge_tag,
                alpha=float(alpha),
                lambda_weight=float(lambda_weight),
            )

            deltas = [
                float(metrics["interference_delta"])
                for metrics in results.values()
                if isinstance(metrics.get("interference_delta"), (int, float))
            ]
            min_delta = min(deltas) if deltas else float("-inf")
            mean_delta = (sum(deltas) / len(deltas)) if deltas else float("-inf")

            row = {
                "alpha": float(alpha),
                "lambda": float(lambda_weight),
                "artifact_dir": str(merge_result.artifact_dir),
                "manifest_path": str(merge_result.manifest_path),
                "num_merged_params": int(merge_result.num_merged_params),
                "num_skipped_params": int(merge_result.num_skipped_params),
                "min_interference_delta": (None if min_delta == float("-inf") else float(min_delta)),
                "mean_interference_delta": (None if mean_delta == float("-inf") else float(mean_delta)),
                "results": results,
            }
            rows.append(row)

            if min_delta > best_score or (min_delta == best_score and mean_delta > best_tiebreak):
                best_score = min_delta
                best_tiebreak = mean_delta
                best_idx = len(rows) - 1

            print(
                f"[{counter}/{total}] alpha={alpha:g}, lambda={lambda_weight:g}, "
                f"min_interference={row['min_interference_delta']}, mean_interference={row['mean_interference_delta']}"
            )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "x_source": str(args.x_source),
        "y_source": str(args.y_source),
        "split": args.split,
        "eval_tasks": eval_tasks,
        "energy_threshold": float(args.energy_threshold),
        "merge_mode": args.merge_mode,
        "rows": rows,
        "best_index": best_idx,
        "best": (rows[best_idx] if best_idx is not None else None),
    }

    summary_path = out_root / "continual_sweep_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"💾 Continual sweep summary saved to {summary_path}")
    return summary


__all__ = [
    "continual_merge_from_args",
    "evaluate_continual_from_args",
    "materialize_merged_artifact_from_args",
]
