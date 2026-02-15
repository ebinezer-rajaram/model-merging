"""Evaluation helpers for merged adapters."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from merging.optimizers.registry import apply_optimizer_overrides, OptimizerContext, optimize_lambda_policy
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
from merging.evaluation.interference import (
    TASK_METRICS,
    load_eval_metrics_json,
    maybe_add_interference_delta,
    maybe_compute_interference_baselines,
)
from merging.optimizers.core.heldout_reporting import export_heldout_tracking_artifacts


def _get_optimizer_bool_param(
    params: Optional[Dict[str, object]],
    key: str,
    default: bool = False,
) -> bool:
    if not isinstance(params, Mapping):
        return default
    optimizer = params.get("optimizer")
    if not isinstance(optimizer, Mapping):
        return default
    opt_params = optimizer.get("params")
    if not isinstance(opt_params, Mapping):
        return default
    if key not in opt_params:
        return default
    return bool(opt_params.get(key))


def _print_pre_eval_coefficients(metadata: Dict[str, Any], source_tasks: List[str]) -> None:
    params = metadata.get("params")
    if not isinstance(params, Mapping):
        return
    optimizer = params.get("optimizer")
    if not isinstance(optimizer, Mapping):
        return
    provenance = optimizer.get("provenance")
    if not isinstance(provenance, Mapping):
        return
    coeffs = provenance.get("final_task_coefficients")
    if not isinstance(coeffs, list) or not coeffs:
        return
    task_labels = source_tasks if len(source_tasks) == len(coeffs) else [f"task_{i}" for i in range(len(coeffs))]
    print("\n[AdaMerging] Final task coefficients (pre-eval):")
    for task, coeff in zip(task_labels, coeffs):
        try:
            print(f"  - {task}: {float(coeff):.6f}")
        except Exception:
            print(f"  - {task}: {coeff}")


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
        method_name = metadata.get("merge_method", "merged")
        run_output_path: Optional[Path] = None
        if (
            _get_optimizer_bool_param(effective_params, "save_metadata_pre_eval", default=False)
            or _get_optimizer_bool_param(effective_params, "print_coefficients_pre_eval", default=False)
        ):
            extra_params = {}
            if metadata.get("lambda") is not None:
                extra_params["lambda"] = metadata.get("lambda")
            if isinstance(metadata.get("optimizer"), Mapping):
                extra_params["optimizer"] = metadata.get("optimizer")
            run_output_path = create_merge_output_path(
                method_name,
                source_tasks or task_names or [],
                extra_params if extra_params else None,
            )
        if _get_optimizer_bool_param(effective_params, "save_metadata_pre_eval", default=False):
            if run_output_path is None:
                raise RuntimeError("Failed to resolve run output path for pre-eval metadata save.")
            merge_metadata_path = run_output_path / "merge_metadata.json"
            with merge_metadata_path.open("w") as handle:
                json.dump(metadata, handle, indent=2)
            if show_summary:
                print(f"\n💾 Pre-eval merge metadata saved to {merge_metadata_path}")
        if _get_optimizer_bool_param(effective_params, "print_coefficients_pre_eval", default=False):
            _print_pre_eval_coefficients(metadata, source_tasks)

        if compute_missing_interference_baselines:
            maybe_compute_interference_baselines(
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
                metrics = load_eval_metrics_json(
                    task_key,
                    split,
                    f"best_{endpoint_task}_adapter.json",
                    eval_tag=eval_tag,
                )
                if metrics is None:
                    cache_ok = False
                    break
                cached[task] = metrics
                maybe_add_interference_delta(task, cached[task], split, show_summary, eval_tag=eval_tag)
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
                    maybe_add_interference_delta(task, results[task], split, show_summary, eval_tag=eval_tag)

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
            eval_dir = resolve_merge_eval_dir(method_name, source_tasks, split)
            eval_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"__{eval_tag}" if eval_tag else ""
            results_path = eval_dir / f"eval_results_{merge_tag}{suffix}_{split}.json"
            with results_path.open("w") as handle:
                json.dump(summary, handle, indent=2)
            if show_summary:
                print(f"\n💾 Evaluation results saved to {results_path}")

            # Also write a run bundle for in-memory merges.
            extra_params = {}
            if metadata.get("lambda") is not None:
                extra_params["lambda"] = metadata.get("lambda")
            if isinstance(metadata.get("optimizer"), Mapping):
                extra_params["optimizer"] = metadata.get("optimizer")
            output_path = run_output_path if run_output_path is not None else create_merge_output_path(
                method_name,
                source_tasks or task_names or [],
                extra_params if extra_params else None,
            )
            merge_metadata_path = output_path / "merge_metadata.json"
            with merge_metadata_path.open("w") as handle:
                json.dump(metadata, handle, indent=2)
            export_heldout_tracking_artifacts(
                metadata=metadata,
                output_dir=output_path,
                show_summary=show_summary,
            )
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
        maybe_compute_interference_baselines(
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
            maybe_add_interference_delta(task, results[task], split, show_summary, eval_tag=eval_tag)

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
        export_heldout_tracking_artifacts(
            metadata=metadata,
            output_dir=merged_run_path,
            show_summary=show_summary,
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
        {
            k: v
            for k, v in {
                "lambda": metadata.get("lambda"),
                "optimizer": metadata.get("optimizer") if isinstance(metadata.get("optimizer"), Mapping) else None,
            }.items()
            if v is not None
        } or None,
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
            maybe_add_interference_delta(task, results[task], split, show_summary, eval_tag=eval_tag)

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
