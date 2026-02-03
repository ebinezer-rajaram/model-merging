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
    PACKAGE_ROOT,
    resolve_best_adapter,
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

    results: Dict[str, Dict] = {}
    if method == "weighted_delta":
        if adapter_path is not None:
            raise ValueError("weighted_delta does not accept adapter_path (merge happens in-memory).")
        if not task_names or len(task_names) != 2:
            raise ValueError("weighted_delta requires exactly two tasks in --tasks.")
        if lambda_weight is None:
            raise ValueError("weighted_delta requires --lambda.")

        adapter1_path, meta1 = resolve_best_adapter(task_names[0])
        adapter2_path, meta2 = resolve_best_adapter(task_names[1])

        from experiments.extract_vector import extract_task_vector_from_lora
        from merging.weighted import merge_task_vectors_weighted

        print("\n📥 Extracting task vector 1 (delta W)")
        tv1 = extract_task_vector_from_lora(adapter1_path)
        print("\n📥 Extracting task vector 2 (delta W)")
        tv2 = extract_task_vector_from_lora(adapter2_path)

        merged_delta = merge_task_vectors_weighted(
            tv1,
            tv2,
            lambda_weight=lambda_weight,
            merge_mode="common",
        )

        metadata = {
            "merge_method": "weighted_delta",
            "lambda": lambda_weight,
            "merge_mode": "common",
            "num_adapters": 2,
            "timestamp": datetime.now().isoformat(),
            "source_adapters": [meta1, meta2],
            "num_parameters": len(merged_delta),
        }
        source_tasks = task_names
        tasks_to_eval = eval_tasks or source_tasks
        if not tasks_to_eval:
            raise ValueError("No evaluation tasks provided for weighted_delta.")

        merge_tag = build_merge_tag(metadata, source_tasks)
        print(f"\n📊 Evaluating weighted_delta on {len(tasks_to_eval)} task(s) ({split} split)")

        for i, task in enumerate(tasks_to_eval, 1):
            print(f"\n[{i}/{len(tasks_to_eval)}] Evaluating on {task}...")
            try:
                result = evaluate(
                    task=task,
                    adapter=None,
                    split=split,
                    batch_size=batch_size,
                    trained_on_task=merge_tag,
                    enable_cache=enable_cache,
                    show_summary=show_summary,
                    generate_confusion_matrix=generate_confusion_matrix,
                    delta_weights=merged_delta,
                    adapter_label=merge_tag,
                    merged_tasks=source_tasks,
                    merged_method=metadata.get("merge_method"),
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
                "adapter_path": None,
                "merge_tag": merge_tag,
                "source_tasks": source_tasks,
                "evaluated_tasks": tasks_to_eval,
                "results": results,
                "merge_method": "weighted_delta",
                "lambda": lambda_weight,
            }
            summary_dir = PACKAGE_ROOT / "artifacts" / "merged" / "weighted_delta"
            summary_dir.mkdir(parents=True, exist_ok=True)
            results_path = summary_dir / f"eval_results_{merge_tag}_{split}.json"
            with results_path.open("w") as handle:
                json.dump(summary, handle, indent=2)
            if show_summary:
                print(f"\n💾 Evaluation results saved to {results_path}")

        return results

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
                merged_tasks=source_tasks,
                merged_method=metadata.get("merge_method"),
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
