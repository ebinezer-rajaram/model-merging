"""Evaluation helpers for merged adapters."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from merging.utils import (
    build_merge_tag,
    infer_merged_source_tasks,
    load_merge_metadata,
    resolve_merged_adapter_path,
)


def evaluate_merged_adapter(
    *,
    adapter_path: Optional[str | Path] = None,
    method: Optional[str] = None,
    task_names: Optional[List[str]] = None,
    lambda_weight: Optional[float] = None,
    run_id: Optional[str] = None,
    eval_tasks: Optional[List[str]] = None,
    split: str = "test",
    batch_size: Optional[int] = None,
    enable_cache: bool = False,
    generate_confusion_matrix: bool = False,
    save_results: bool = True,
    show_summary: bool = True,
) -> Dict[str, Dict]:
    """Evaluate a merged adapter on one or more tasks."""
    from experiments.evaluate_task import evaluate

    merged_run_path = resolve_merged_adapter_path(
        adapter_path=adapter_path,
        method=method,
        task_names=task_names,
        lambda_weight=lambda_weight,
        run_id=run_id,
    )

    metadata = load_merge_metadata(merged_run_path)
    source_tasks = infer_merged_source_tasks(metadata, fallback=task_names)

    tasks_to_eval = eval_tasks or source_tasks
    if not tasks_to_eval:
        raise ValueError("No evaluation tasks provided or inferred for merged adapter.")

    merge_tag = build_merge_tag(metadata, source_tasks or task_names)

    results: Dict[str, Dict] = {}
    print(f"\n📊 Evaluating merged adapter on {len(tasks_to_eval)} task(s) ({split} split)")

    for i, task in enumerate(tasks_to_eval, 1):
        print(f"\n[{i}/{len(tasks_to_eval)}] Evaluating on {task}...")
        try:
            result = evaluate(
                task=task,
                adapter=str(merged_run_path),
                split=split,
                batch_size=batch_size,
                trained_on_task=merge_tag,
                enable_cache=enable_cache,
                show_summary=show_summary,
                generate_confusion_matrix=generate_confusion_matrix,
            )
            results[task] = result.metrics

            if show_summary:
                print(f"✅ {task} evaluation complete:")
                for key, value in sorted(result.metrics.items())[:5]:
                    if isinstance(value, (int, float)):
                        print(f"   {key}: {value:.4f}")
        except Exception as exc:
            print(f"❌ Failed to evaluate on {task}: {exc}")
            results[task] = {"error": str(exc)}

    if save_results:
        summary = {
            "split": split,
            "timestamp": datetime.now().isoformat(),
            "adapter_path": str(merged_run_path),
            "merge_tag": merge_tag,
            "source_tasks": source_tasks,
            "evaluated_tasks": tasks_to_eval,
            "results": results,
        }
        results_path = merged_run_path / f"eval_results_{split}.json"
        with results_path.open("w") as handle:
            json.dump(summary, handle, indent=2)
        if show_summary:
            print(f"\n💾 Evaluation results saved to {results_path}")

    return results


__all__ = ["evaluate_merged_adapter"]
