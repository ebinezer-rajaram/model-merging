"""Evaluation helpers for continual compressed artifacts."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from core.evaluation.evaluate_task import evaluate
from core.evaluation.split_utils import canonical_output_split
from merging.artifacts.continual_format import ContinualArtifactReader
from merging.evaluation.interference import (
    maybe_add_interference_delta,
    maybe_compute_interference_baselines,
)
from merging.runtime.utils import (
    resolve_merge_eval_dir,
    update_results_index,
)


def _ordered_union(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _format_float_token(value: float) -> str:
    token = f"{float(value):g}"
    return token.replace("-", "m").replace(".", "p")


def _is_test_other_output(output_split: str) -> bool:
    return canonical_output_split(output_split) == "test_other"


def build_continual_tag(
    *,
    source_tasks: Sequence[str],
    alpha: Optional[float] = None,
    lambda_weight: Optional[float] = None,
) -> str:
    parts = ["continual", "merged"]
    if source_tasks:
        parts.extend(source_tasks)
    if alpha is not None:
        parts.append(f"alpha{_format_float_token(alpha)}")
    if lambda_weight is not None:
        parts.append(f"lambda{_format_float_token(lambda_weight)}")
    return "_".join(parts)


def evaluate_continual_artifact(
    *,
    artifact_path: str | Path,
    eval_tasks: Optional[List[str]] = None,
    split: str = "test",
    batch_size: Optional[int] = None,
    enable_cache: bool = False,
    show_summary: bool = True,
    compute_missing_interference_baselines: bool = True,
    save_results: bool = True,
    eval_subset: Optional[Dict[str, Any]] = None,
    merge_tag: Optional[str] = None,
    alpha: Optional[float] = None,
    lambda_weight: Optional[float] = None,
    method_name: str = "continual",
) -> Dict[str, Dict[str, Any]]:
    """Evaluate a continual artifact on one or more tasks."""
    requested_split = str(split)
    output_split = canonical_output_split(requested_split)
    artifact_dir = Path(artifact_path).resolve()
    reader = ContinualArtifactReader(artifact_dir)
    manifest = reader.manifest

    source_tasks = [str(x) for x in manifest.get("constituent_tasks_flat", []) if str(x)]
    tasks_to_eval = list(eval_tasks) if eval_tasks else list(source_tasks)
    if not tasks_to_eval:
        raise ValueError("No evaluation tasks provided and no constituent tasks could be inferred.")

    if merge_tag is None:
        merge_tag = build_continual_tag(
            source_tasks=source_tasks,
            alpha=alpha,
            lambda_weight=lambda_weight,
        )

    if compute_missing_interference_baselines:
        maybe_compute_interference_baselines(
            tasks=tasks_to_eval,
            split=split,
            enable_cache=enable_cache,
            batch_size=batch_size,
            show_summary=show_summary,
            eval_subset=eval_subset,
        )

    results: Dict[str, Dict[str, Any]] = {}
    failed: List[str] = []

    for idx, task in enumerate(tasks_to_eval, start=1):
        if show_summary:
            print(f"\n[{idx}/{len(tasks_to_eval)}] Evaluating continual artifact on {task}...")
        try:
            payload = evaluate(
                task=task,
                adapter=None,
                split=split,
                batch_size=batch_size,
                trained_on_task=merge_tag,
                enable_cache=enable_cache,
                show_summary=show_summary,
                generate_confusion_matrix=False,
                continual_artifact_path=artifact_dir,
                adapter_label=merge_tag,
                merged_tasks=source_tasks,
                merged_method=method_name,
                eval_subset=eval_subset,
            )
            metrics = dict(payload.metrics)
            maybe_add_interference_delta(task, metrics, split, show_summary, eval_subset=eval_subset)
            results[task] = metrics
        except Exception as exc:
            failed.append(task)
            results[task] = {"error": str(exc)}

    deltas = [
        float(metrics["interference_delta"])
        for metrics in results.values()
        if isinstance(metrics, Mapping) and isinstance(metrics.get("interference_delta"), (int, float))
    ]
    agg = {
        "num_tasks_with_interference": len(deltas),
        "min_interference_delta": (min(deltas) if deltas else None),
        "mean_interference_delta": ((sum(deltas) / len(deltas)) if deltas else None),
    }

    summary = {
        "timestamp": datetime.now().isoformat(),
        "split": output_split,
        "requested_split": requested_split,
        "artifact_path": str(artifact_dir),
        "merge_tag": merge_tag,
        "source_tasks": source_tasks,
        "evaluated_tasks": tasks_to_eval,
        "eval_subset": eval_subset,
        "alpha": alpha,
        "lambda": lambda_weight,
        "results": results,
        "interference_aggregate": agg,
    }

    if save_results:
        run_results_path = artifact_dir / f"eval_results_{output_split}.json"
        with run_results_path.open("w") as handle:
            json.dump(summary, handle, indent=2)

        eval_dir = resolve_merge_eval_dir(method_name, source_tasks, output_split)
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_results_path = eval_dir / f"eval_results_{merge_tag}_{output_split}.json"
        with eval_results_path.open("w") as handle:
            json.dump(summary, handle, indent=2)

        metadata = {
            "merge_method": method_name,
            "source_adapters": manifest.get("source_metadata", []),
            "params": {
                "alpha": alpha,
                "lambda": lambda_weight,
                "dense_merge_semantics": manifest.get("dense_merge_semantics", {}),
            },
        }
        if not _is_test_other_output(output_split):
            update_results_index(
                eval_dir,
                merge_tag=merge_tag,
                split=output_split,
                results_path=eval_results_path,
                metadata=metadata,
                summary=summary,
                run_path=artifact_dir,
            )

    if failed:
        raise RuntimeError(
            f"Continual artifact evaluation failed for {len(failed)} task(s): {', '.join(failed)}"
        )

    return results


def select_eval_tasks_for_sources(
    *,
    x_tasks: Sequence[str],
    y_tasks: Sequence[str],
    explicit_eval_tasks: Optional[Sequence[str]],
) -> List[str]:
    if explicit_eval_tasks:
        return [str(x) for x in explicit_eval_tasks]
    return _ordered_union([*(str(x) for x in x_tasks), *(str(y) for y in y_tasks)])


__all__ = [
    "build_continual_tag",
    "evaluate_continual_artifact",
    "select_eval_tasks_for_sources",
]
