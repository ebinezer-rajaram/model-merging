"""Evaluation helpers for merged adapters."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from merging.core.registry import get_merge_method
from merging.core.utils import (
    _format_lambda,
    build_merge_tag,
    create_merge_output_path,
    infer_merged_source_tasks,
    load_merge_metadata,
    PACKAGE_ROOT,
    resolve_best_adapter,
    resolve_merged_adapter_path,
    save_merged_adapter,
)

EPS = 1e-8
TASK_METRICS = {
    "asr": ("wer", False),
    "emotion": ("macro_f1", True),
    "intent": ("accuracy", True),
    "kws": ("macro_f1", True),
    "langid": ("accuracy", True),
    "speaker_id": ("accuracy", True),
    "speaker_ver": ("accuracy", True),
}


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
    save_merged: bool = False,
    save_results: bool = True,
    show_summary: bool = True,
    merge_mode: str = "common",
) -> Dict[str, Dict]:
    """Evaluate a merged adapter on one or more tasks."""
    from experiments.evaluate_task import evaluate

    results: Dict[str, Dict] = {}
    if adapter_path is None:
        if not method:
            raise ValueError("method is required for in-memory merged evaluation.")
        if not task_names:
            raise ValueError("task_names are required for in-memory merged evaluation.")

        adapter_paths = []
        source_metadata = []
        for task_name in task_names:
            adapter_path_resolved, meta = resolve_best_adapter(task_name)
            adapter_paths.append(adapter_path_resolved)
            source_metadata.append(meta)

        method_impl = get_merge_method(method)
        if save_merged and not method_impl.saveable:
            raise ValueError(f"{method} cannot be saved as a LoRA adapter.")

        merged_delta, merged_weights, metadata = _merge_in_memory(
            method=method,
            adapter_paths=adapter_paths,
            source_metadata=source_metadata,
            lambda_weight=lambda_weight,
            merge_mode=merge_mode,
        )

        source_tasks = task_names
        tasks_to_eval = eval_tasks or source_tasks
        if not tasks_to_eval:
            raise ValueError("No evaluation tasks provided for merged adapter.")

        merge_tag = build_merge_tag(metadata, source_tasks)

        if save_merged:
            merged_run_path = _save_merged_adapter(
                method=method,
                task_names=task_names,
                adapter_paths=adapter_paths,
                merged_weights=merged_weights,
                metadata=metadata,
            )
            return _evaluate_saved_merged(
                merged_run_path=merged_run_path,
                metadata=metadata,
                tasks_to_eval=tasks_to_eval,
                split=split,
                batch_size=batch_size,
                enable_cache=enable_cache,
                generate_confusion_matrix=generate_confusion_matrix,
                save_results=save_results,
                show_summary=show_summary,
            )

        print(f"\n📊 Evaluating {metadata.get('merge_method')} in-memory on {len(tasks_to_eval)} task(s) ({split} split)")

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
                _maybe_add_interference_delta(task, results[task], split, show_summary)

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
                "merge_method": metadata.get("merge_method"),
            }
            if metadata.get("lambda") is not None:
                summary["lambda"] = metadata.get("lambda")
            method_name = metadata.get("merge_method", "merged")
            summary_dir = PACKAGE_ROOT / "artifacts" / "merged" / method_name
            if method_name == "weighted_delta" and source_tasks:
                task_combo = "_".join(sorted(source_tasks))
                summary_dir = summary_dir / task_combo
                if metadata.get("lambda") is not None:
                    summary_dir = summary_dir / _format_lambda(float(metadata["lambda"]))
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
            _maybe_add_interference_delta(task, results[task], split, show_summary)

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


def _merge_in_memory(
    *,
    method: str,
    adapter_paths: List[Path],
    source_metadata: List[Dict],
    lambda_weight: Optional[float],
    merge_mode: str,
) -> tuple[Dict[str, "torch.Tensor"], Optional[Dict[str, "torch.Tensor"]], Dict]:
    method_impl = get_merge_method(method)
    method_impl.validate(len(adapter_paths), lambda_weight)
    merge_output = method_impl.merge_in_memory(
        adapter_paths=adapter_paths,
        source_metadata=source_metadata,
        merge_mode=merge_mode,
        lambda_weight=lambda_weight,
    )
    return merge_output.merged_delta, merge_output.merged_weights, merge_output.metadata


def _load_eval_metric(task: str, split: str, filename: str, metric_key: str) -> Optional[float]:
    metrics_path = PACKAGE_ROOT / "artifacts" / task / "metrics" / "eval" / split / filename
    if not metrics_path.exists():
        return None
    try:
        with metrics_path.open("r") as handle:
            data = json.load(handle)
    except Exception:
        return None
    value = data.get(metric_key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _oriented_score(value: float, higher_is_better: bool) -> float:
    return value if higher_is_better else -value


def _maybe_add_interference_delta(task: str, metrics: Dict, split: str, show_summary: bool) -> None:
    task_key = task.lower()
    if task_key not in TASK_METRICS:
        return

    metric_key, higher_is_better = TASK_METRICS[task_key]
    merged_value = metrics.get(metric_key)
    if not isinstance(merged_value, (int, float)):
        return

    base_value = _load_eval_metric(task_key, split, "base_model.json", metric_key)
    task_value = _load_eval_metric(task_key, split, f"best_{task_key}_adapter.json", metric_key)
    if base_value is None or task_value is None:
        if show_summary:
            missing = []
            if base_value is None:
                missing.append("base_model.json")
            if task_value is None:
                missing.append(f"best_{task_key}_adapter.json")
            missing_str = ", ".join(missing)
            print(
                f"⚠️  Skipping interference_delta for {task_key}/{split}: "
                f"missing {missing_str} under artifacts/{task_key}/metrics/eval/{split}"
            )
        return

    merged_score = _oriented_score(float(merged_value), higher_is_better)
    base_score = _oriented_score(base_value, higher_is_better)
    task_score = _oriented_score(task_value, higher_is_better)
    denom = task_score - base_score
    if abs(denom) < EPS:
        return

    metrics["interference_delta"] = (merged_score - base_score) / denom
    metrics["interference_delta_meta"] = {
        "metric": metric_key,
        "base": base_value,
        "task_adapter": task_value,
        "merged": float(merged_value),
        "split": split,
    }
    if show_summary:
        print(f"   interference_delta: {metrics['interference_delta']:.4f}")


def _save_merged_adapter(
    *,
    method: str,
    task_names: List[str],
    adapter_paths: List[Path],
    merged_weights: Optional[Dict[str, "torch.Tensor"]],
    metadata: Dict,
) -> Path:
    method_impl = get_merge_method(method)
    if not method_impl.saveable:
        raise ValueError(f"{method} cannot be saved as a LoRA adapter.")

    output_path = create_merge_output_path(
        method,
        task_names,
        {"lambda": metadata.get("lambda")} if metadata.get("lambda") is not None else None,
    )

    if method_impl.save_fn is not None:
        method_impl.save_fn(
            adapter_paths=adapter_paths,
            output_path=output_path,
            merge_mode=metadata.get("merge_mode", "common"),
            show_progress=True,
        )
        return output_path

    if merged_weights is None:
        raise ValueError("Merged weights are required for saving this method.")

    save_merged_adapter(
        weights=merged_weights,
        output_path=output_path,
        reference_adapter_path=adapter_paths[0],
        metadata=metadata,
        register_run=True,
    )

    return output_path


def _evaluate_saved_merged(
    *,
    merged_run_path: Path,
    metadata: Dict,
    tasks_to_eval: List[str],
    split: str,
    batch_size: Optional[int],
    enable_cache: bool,
    generate_confusion_matrix: bool,
    save_results: bool,
    show_summary: bool,
) -> Dict[str, Dict]:
    from experiments.evaluate_task import evaluate

    results: Dict[str, Dict] = {}
    source_tasks = infer_merged_source_tasks(metadata)
    merge_tag = build_merge_tag(metadata, source_tasks)

    print(f"\n📊 Evaluating saved merged adapter on {len(tasks_to_eval)} task(s) ({split} split)")

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
            _maybe_add_interference_delta(task, results[task], split, show_summary)

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
