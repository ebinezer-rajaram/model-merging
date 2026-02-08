"""Evaluation helpers for merged adapters."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from merging.plugins.optimizers import apply_optimizer_overrides, OptimizerContext, optimize_lambda_policy
from merging.engine.registry import get_merge_method, normalize_params
from merging.engine.runner import resolve_adapter_specs
from merging.config.specs import merge_spec_from_legacy_args
from merging.runtime.utils import (
    _format_lambda,
    build_merge_tag,
    create_merge_output_path,
    infer_merged_source_tasks,
    load_merge_metadata,
    PACKAGE_ROOT,
    resolve_merge_eval_dir,
    update_results_index,
    resolve_merged_adapter_path,
    save_merged_adapter,
)
from core import compute_eval_subset_tag

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
    params: Optional[Dict[str, object]] = None,
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
    compute_missing_interference_baselines: bool = True,
    eval_subset: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict]:
    """Evaluate a merged adapter on one or more tasks."""
    from experiments.evaluate_task import evaluate

    results: Dict[str, Dict] = {}
    eval_tag: Optional[str] = None
    if eval_subset and bool(eval_subset.get("enabled", True)):
        eval_tag = compute_eval_subset_tag(eval_subset)
    if adapter_path is None:
        if not method:
            raise ValueError("method is required for in-memory merged evaluation.")
        if not task_names:
            raise ValueError("task_names are required for in-memory merged evaluation.")

        resolved = resolve_adapter_specs(task_names)
        adapter_paths = [path for path, _ in resolved]
        source_metadata = [meta for _, meta in resolved]

        method_impl = get_merge_method(method)
        effective_params = normalize_params(
            method_impl,
            params=params,
            legacy_lambda_weight=lambda_weight,
        )
        if save_merged and not method_impl.saveable:
            raise ValueError(f"{method} cannot be saved as a LoRA adapter.")

        merged_delta, merged_weights, metadata = _merge_in_memory(
            method=method,
            adapter_paths=adapter_paths,
            source_metadata=source_metadata,
            params=effective_params,
            merge_mode=merge_mode,
        )

        source_tasks = [meta.get("task") for meta in source_metadata if meta.get("task")]
        tasks_to_eval = eval_tasks or source_tasks or task_names
        if not tasks_to_eval:
            raise ValueError("No evaluation tasks provided for merged adapter.")

        merge_tag = build_merge_tag(metadata, source_tasks or None)

        if compute_missing_interference_baselines:
            _maybe_compute_interference_baselines(
                tasks=tasks_to_eval,
                split=split,
                enable_cache=enable_cache,
                batch_size=batch_size,
                show_summary=show_summary,
                eval_subset=eval_subset,
            )

        cached_endpoint_results: Optional[Dict[str, Dict]] = None
        lambda_value = metadata.get("lambda")
        if (
            isinstance(lambda_value, (int, float))
            and lambda_value in (0.0, 1.0)
            and source_tasks
            and len(source_tasks) == 2
            and all(source_tasks)
        ):
            endpoint_task = source_tasks[0] if lambda_value == 1.0 else source_tasks[1]
            cached: Dict[str, Dict] = {}
            cache_ok = True
            for task in tasks_to_eval:
                task_key = task.lower()
                metrics = _load_eval_metrics_json(
                    task_key,
                    split,
                    f"best_{endpoint_task}_adapter.json",
                    eval_tag=eval_tag,
                )
                if metrics is None:
                    cache_ok = False
                    break
                cached[task] = metrics
                _maybe_add_interference_delta(task, cached[task], split, show_summary, eval_tag=eval_tag)
            if cache_ok:
                cached_endpoint_results = cached
                if show_summary:
                    print(
                        f"♻️  Using cached cross-task metrics for endpoint λ={lambda_value:.1f} "
                        f"(adapter={endpoint_task})"
                    )

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
                eval_subset=eval_subset,
            )

        print(f"\n📊 Evaluating {metadata.get('merge_method')} in-memory on {len(tasks_to_eval)} task(s) ({split} split)")

        if cached_endpoint_results is not None:
            results.update(cached_endpoint_results)
        else:
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
                        eval_subset=eval_subset,
                    )
                    results[task] = result.metrics
                    _maybe_add_interference_delta(task, results[task], split, show_summary, eval_tag=eval_tag)

                    if show_summary:
                        print(f"✅ {task} evaluation complete:")
                        task_key = task.lower()
                        metric_key = None
                        if task_key in TASK_METRICS:
                            metric_key, _ = TASK_METRICS[task_key]
                            metric_value = result.metrics.get(metric_key)
                            if isinstance(metric_value, (int, float)):
                                print(f"   {metric_key}: {metric_value:.4f}")
                        if "interference_delta" in result.metrics:
                            print(f"   interference_delta: {result.metrics['interference_delta']:.4f}")
                        for key, value in sorted(result.metrics.items())[:3]:
                            if isinstance(value, (int, float)) and key not in {metric_key, "interference_delta"}:
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
            if eval_subset is not None:
                summary["eval_subset"] = eval_subset
            if eval_tag is not None:
                summary["eval_tag"] = eval_tag
            method_name = metadata.get("merge_method", "merged")
            eval_dir = resolve_merge_eval_dir(method_name, source_tasks, split)
            eval_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"__{eval_tag}" if eval_tag else ""
            results_path = eval_dir / f"eval_results_{merge_tag}{suffix}_{split}.json"
            with results_path.open("w") as handle:
                json.dump(summary, handle, indent=2)
            if show_summary:
                print(f"\n💾 Evaluation results saved to {results_path}")

            # Also write a run bundle for in-memory merges.
            output_path = create_merge_output_path(
                method_name,
                source_tasks or task_names or [],
                {"lambda": metadata.get("lambda")} if metadata.get("lambda") is not None else None,
            )
            merge_metadata_path = output_path / "merge_metadata.json"
            with merge_metadata_path.open("w") as handle:
                json.dump(metadata, handle, indent=2)
            run_suffix = f"__{eval_tag}" if eval_tag else ""
            run_results_path = output_path / f"eval_results_{split}{run_suffix}.json"
            with run_results_path.open("w") as handle:
                json.dump(summary, handle, indent=2)
            summary_path = output_path / "summary.json"
            with summary_path.open("w") as handle:
                json.dump(
                    {
                        "timestamp": summary["timestamp"],
                        "merge_tag": merge_tag,
                        "merge_method": metadata.get("merge_method"),
                        "params": metadata.get("params", {}),
                        "source_tasks": source_tasks,
                        "evaluated_tasks": tasks_to_eval,
                        "split": split,
                        "eval_tag": eval_tag,
                        "results_path": str(run_results_path),
                    },
                    handle,
                    indent=2,
                )
            if show_summary:
                print(f"📦 Run bundle saved to {output_path}")

            update_results_index(
                eval_dir,
                merge_tag=merge_tag,
                split=split,
                results_path=results_path,
                metadata=metadata,
                summary=summary,
                run_path=output_path,
            )

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

    if compute_missing_interference_baselines:
        _maybe_compute_interference_baselines(
            tasks=tasks_to_eval,
            split=split,
            enable_cache=enable_cache,
            batch_size=batch_size,
            show_summary=show_summary,
            eval_subset=eval_subset,
        )

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
                eval_subset=eval_subset,
            )
            results[task] = result.metrics
            _maybe_add_interference_delta(task, results[task], split, show_summary, eval_tag=eval_tag)

            if show_summary:
                print(f"✅ {task} evaluation complete:")
                task_key = task.lower()
                metric_key = None
                if task_key in TASK_METRICS:
                    metric_key, _ = TASK_METRICS[task_key]
                    metric_value = result.metrics.get(metric_key)
                    if isinstance(metric_value, (int, float)):
                        print(f"   {metric_key}: {metric_value:.4f}")
                if "interference_delta" in result.metrics:
                    print(f"   interference_delta: {result.metrics['interference_delta']:.4f}")
                for key, value in sorted(result.metrics.items())[:3]:
                    if isinstance(value, (int, float)) and key not in {metric_key, "interference_delta"}:
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
        if eval_subset is not None:
            summary["eval_subset"] = eval_subset
        if eval_tag is not None:
            summary["eval_tag"] = eval_tag
        suffix = f"__{eval_tag}" if eval_tag else ""
        results_path = merged_run_path / f"eval_results_{split}{suffix}.json"
        with results_path.open("w") as handle:
            json.dump(summary, handle, indent=2)
        if show_summary:
            print(f"\n💾 Evaluation results saved to {results_path}")

        method_name = metadata.get("merge_method", "merged")
        eval_dir = resolve_merge_eval_dir(method_name, source_tasks, split)
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_results_path = eval_dir / f"eval_results_{merge_tag}{suffix}_{split}.json"
        with eval_results_path.open("w") as handle:
            json.dump(summary, handle, indent=2)
        if show_summary:
            print(f"💾 Evaluation results saved to {eval_results_path}")

        update_results_index(
            eval_dir,
            merge_tag=merge_tag,
            split=split,
            results_path=eval_results_path,
            metadata=metadata,
            summary=summary,
            run_path=merged_run_path,
        )

    return results


def _merge_in_memory(
    *,
    method: str,
    adapter_paths: List[Path],
    source_metadata: List[Dict],
    params: Optional[Dict[str, object]],
    merge_mode: str,
) -> tuple[Dict[str, "torch.Tensor"], Optional[Dict[str, "torch.Tensor"]], Dict]:
    method_impl = get_merge_method(method)
    merge_spec = merge_spec_from_legacy_args(
        adapters=[str(p) for p in adapter_paths],
        method=method,
        merge_mode=merge_mode,
        lambda_weight=None if params is None else params.get("lambda"),
        params=params,
    )
    effective_params = normalize_params(method_impl, params=merge_spec.method_params)
    optimizer_result = optimize_lambda_policy(
        merge_spec,
        OptimizerContext(
            method=method,
            adapter_specs=[str(p) for p in adapter_paths],
            adapter_paths=adapter_paths,
            source_metadata=source_metadata,
            merge_mode=merge_mode,
            output_dir=None,
            method_params=dict(effective_params),
            lambda_policy=merge_spec.lambda_policy,
        ),
    )
    effective_params = apply_optimizer_overrides(effective_params, optimizer_result)
    if optimizer_result.lambda_policy is not None:
        effective_params["lambda_policy"] = {
            "type": optimizer_result.lambda_policy.type,
            "value": optimizer_result.lambda_policy.value,
            "default": optimizer_result.lambda_policy.default,
            "overrides": dict(optimizer_result.lambda_policy.overrides),
        }
    effective_params["optimizer"] = {
        "type": (merge_spec.optimizer.type if merge_spec.optimizer is not None else "none"),
        "params": (dict(merge_spec.optimizer.params) if merge_spec.optimizer is not None else {}),
        "provenance": optimizer_result.provenance,
    }
    method_impl.validate(len(adapter_paths), effective_params)
    merge_output = method_impl.merge_in_memory(
        adapter_paths=adapter_paths,
        source_metadata=source_metadata,
        merge_mode=merge_mode,
        params=effective_params,
    )
    return merge_output.merged_delta, merge_output.merged_weights, merge_output.metadata


def _with_eval_tag(filename: str, eval_tag: Optional[str]) -> str:
    if not eval_tag:
        return filename
    stem, dot, ext = filename.rpartition(".")
    if not dot:
        stem, ext = filename, "json"
    return f"{stem}__{eval_tag}.{ext}"


def _load_eval_metric(task: str, split: str, filename: str, metric_key: str, *, eval_tag: Optional[str] = None) -> Optional[float]:
    candidates = [_with_eval_tag(filename, eval_tag), filename] if eval_tag else [filename]
    metrics_path = None
    for name in candidates:
        path = PACKAGE_ROOT / "artifacts" / task / "metrics" / "eval" / split / name
        if path.exists():
            metrics_path = path
            break
    if metrics_path is None:
        return None
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


def _load_eval_metrics_json(
    task: str,
    split: str,
    filename: str,
    *,
    eval_tag: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    candidates = [_with_eval_tag(filename, eval_tag), filename] if eval_tag else [filename]
    metrics_path = None
    for name in candidates:
        path = PACKAGE_ROOT / "artifacts" / task / "metrics" / "eval" / split / name
        if path.exists():
            metrics_path = path
            break
    if metrics_path is None:
        return None
    if not metrics_path.exists():
        return None
    try:
        with metrics_path.open("r") as handle:
            data = json.load(handle)
    except Exception:
        return None
    if isinstance(data, dict):
        return data
    return None


def _oriented_score(value: float, higher_is_better: bool) -> float:
    return value if higher_is_better else -value


def _maybe_compute_interference_baselines(
    *,
    tasks: List[str],
    split: str,
    enable_cache: bool,
    batch_size: Optional[int],
    show_summary: bool,
    eval_subset: Optional[Dict[str, Any]] = None,
) -> None:
    """Ensure baseline metrics exist for computing interference_delta."""
    # Import lazily to avoid pulling in heavyweight deps for call sites
    # that only want to read or post-process metrics.
    from experiments.evaluate_task import evaluate

    eval_tag: Optional[str] = None
    if eval_subset and bool(eval_subset.get("enabled", True)):
        eval_tag = compute_eval_subset_tag(eval_subset)

    for task in tasks:
        task_key = task.lower()
        if task_key not in TASK_METRICS:
            continue

        metric_key, _ = TASK_METRICS[task_key]
        base_value = _load_eval_metric(task_key, split, "base_model.json", metric_key, eval_tag=eval_tag)
        task_value = _load_eval_metric(task_key, split, f"best_{task_key}_adapter.json", metric_key, eval_tag=eval_tag)
        if base_value is not None and task_value is not None:
            continue

        if show_summary:
            missing = []
            if base_value is None:
                missing.append("base_model.json")
            if task_value is None:
                missing.append(f"best_{task_key}_adapter.json")
            print(f"🧮 Computing missing interference baselines for {task_key}/{split}: {', '.join(missing)}")

        try:
            if base_value is None:
                evaluate(
                    task=task_key,
                    adapter=None,
                    split=split,
                    batch_size=batch_size,
                    enable_cache=enable_cache,
                    show_summary=show_summary,
                    generate_confusion_matrix=False,
                    eval_subset=eval_subset,
                )
        except Exception as exc:
            if show_summary:
                print(f"⚠️  Failed to compute base_model metrics for {task_key}/{split}: {exc}")

        try:
            if task_value is None:
                # Passing adapter=<task> resolves that task's "best" adapter run.
                evaluate(
                    task=task_key,
                    adapter=task_key,
                    split=split,
                    batch_size=batch_size,
                    enable_cache=enable_cache,
                    show_summary=show_summary,
                    generate_confusion_matrix=False,
                    eval_subset=eval_subset,
                )
        except Exception as exc:
            if show_summary:
                print(f"⚠️  Failed to compute best adapter metrics for {task_key}/{split}: {exc}")


def _maybe_add_interference_delta(task: str, metrics: Dict, split: str, show_summary: bool, *, eval_tag: Optional[str]) -> None:
    task_key = task.lower()
    if task_key not in TASK_METRICS:
        return

    metric_key, higher_is_better = TASK_METRICS[task_key]
    merged_value = metrics.get(metric_key)
    if not isinstance(merged_value, (int, float)):
        return

    base_value = _load_eval_metric(task_key, split, "base_model.json", metric_key, eval_tag=eval_tag)
    task_value = _load_eval_metric(task_key, split, f"best_{task_key}_adapter.json", metric_key, eval_tag=eval_tag)
    if base_value is None or task_value is None:
        if show_summary:
            missing = []
            if base_value is None:
                missing.append(_with_eval_tag("base_model.json", eval_tag))
            if task_value is None:
                missing.append(_with_eval_tag(f"best_{task_key}_adapter.json", eval_tag))
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
    eval_subset: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict]:
    from experiments.evaluate_task import evaluate

    results: Dict[str, Dict] = {}
    source_tasks = infer_merged_source_tasks(metadata)
    merge_tag = build_merge_tag(metadata, source_tasks)
    eval_tag: Optional[str] = None
    if eval_subset and bool(eval_subset.get("enabled", True)):
        eval_tag = compute_eval_subset_tag(eval_subset)

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
                eval_subset=eval_subset,
            )
            results[task] = result.metrics
            _maybe_add_interference_delta(task, results[task], split, show_summary, eval_tag=eval_tag)

            if show_summary:
                print(f"✅ {task} evaluation complete:")
                task_key = task.lower()
                metric_key = None
                if task_key in TASK_METRICS:
                    metric_key, _ = TASK_METRICS[task_key]
                    metric_value = result.metrics.get(metric_key)
                    if isinstance(metric_value, (int, float)):
                        print(f"   {metric_key}: {metric_value:.4f}")
                if "interference_delta" in result.metrics:
                    print(f"   interference_delta: {result.metrics['interference_delta']:.4f}")
                for key, value in sorted(result.metrics.items())[:3]:
                    if isinstance(value, (int, float)) and key not in {metric_key, "interference_delta"}:
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

        method_name = metadata.get("merge_method", "merged")
        eval_dir = resolve_merge_eval_dir(method_name, source_tasks, split)
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_results_path = eval_dir / f"eval_results_{merge_tag}_{split}.json"
        with eval_results_path.open("w") as handle:
            json.dump(summary, handle, indent=2)
        if show_summary:
            print(f"💾 Evaluation results saved to {eval_results_path}")

        update_results_index(
            eval_dir,
            merge_tag=merge_tag,
            split=split,
            results_path=eval_results_path,
            metadata=metadata,
            summary=summary,
            run_path=merged_run_path,
        )

    return results


__all__ = ["evaluate_merged_adapter"]
