"""Evaluation helpers for merged adapters."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from merging.utils import (
    build_merge_tag,
    create_merge_output_path,
    infer_merged_source_tasks,
    load_adapter_weights,
    load_merge_metadata,
    PACKAGE_ROOT,
    resolve_best_adapter,
    resolve_merged_adapter_path,
    save_merged_adapter,
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
    save_merged: bool = False,
    save_results: bool = True,
    show_summary: bool = True,
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

        if method == "weighted_delta" and save_merged:
            raise ValueError("weighted_delta cannot be saved as a LoRA adapter.")

        merged_delta, merged_weights, metadata = _merge_in_memory(
            method=method,
            adapter_paths=adapter_paths,
            source_metadata=source_metadata,
            lambda_weight=lambda_weight,
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
            summary_dir = PACKAGE_ROOT / "artifacts" / "merged" / metadata.get("merge_method", "merged")
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


def _merge_in_memory(
    *,
    method: str,
    adapter_paths: List[Path],
    source_metadata: List[Dict],
    lambda_weight: Optional[float],
) -> tuple[Dict[str, "torch.Tensor"], Optional[Dict[str, "torch.Tensor"]], Dict]:
    import torch
    from experiments.extract_vector import extract_task_vector_from_lora
    from merging.uniform import merge_adapters_uniform
    from merging.weighted import merge_adapters_weighted, merge_task_vectors_weighted

    if method == "weighted" and lambda_weight is None:
        raise ValueError("weighted requires lambda_weight.")
    if method == "weighted_delta" and lambda_weight is None:
        raise ValueError("weighted_delta requires lambda_weight.")

    if method == "weighted" and len(adapter_paths) != 2:
        raise ValueError("weighted requires exactly two adapters.")
    if method == "weighted_delta" and len(adapter_paths) != 2:
        raise ValueError("weighted_delta requires exactly two adapters.")

    merged_weights: Optional[Dict[str, torch.Tensor]] = None
    merged_delta: Dict[str, torch.Tensor]

    if method == "uniform":
        all_weights = [load_adapter_weights(p) for p in adapter_paths]
        merged_weights = merge_adapters_uniform(all_weights, merge_mode="common")
        merged_delta = _compute_delta_from_lora_weights(merged_weights, adapter_paths[0])
    elif method == "weighted":
        weights1 = load_adapter_weights(adapter_paths[0])
        weights2 = load_adapter_weights(adapter_paths[1])
        merged_weights = merge_adapters_weighted(
            weights1,
            weights2,
            lambda_weight=lambda_weight,
            merge_mode="common",
        )
        merged_delta = _compute_delta_from_lora_weights(merged_weights, adapter_paths[0])
    elif method == "task_vector":
        task_vectors = [extract_task_vector_from_lora(p) for p in adapter_paths]
        merged_delta = merge_adapters_uniform(task_vectors, merge_mode="common")
    elif method == "weighted_delta":
        tv1 = extract_task_vector_from_lora(adapter_paths[0])
        tv2 = extract_task_vector_from_lora(adapter_paths[1])
        merged_delta = merge_task_vectors_weighted(
            tv1,
            tv2,
            lambda_weight=lambda_weight,
            merge_mode="common",
        )
    else:
        raise ValueError(f"Unknown merge method: {method}")

    metadata = {
        "merge_method": method,
        "merge_mode": "common",
        "num_adapters": len(adapter_paths),
        "timestamp": datetime.now().isoformat(),
        "source_adapters": source_metadata,
        "num_parameters": len(merged_delta),
    }
    if lambda_weight is not None:
        metadata["lambda"] = lambda_weight
    return merged_delta, merged_weights, metadata


def _compute_delta_from_lora_weights(
    lora_weights: Dict[str, "torch.Tensor"],
    reference_adapter_path: Path,
) -> Dict[str, "torch.Tensor"]:
    import json

    config_path = reference_adapter_path / "adapter_config.json"
    with open(config_path, "r") as handle:
        config = json.load(handle)

    lora_alpha = config["lora_alpha"]
    lora_r = config["r"]
    scaling = lora_alpha / lora_r

    lora_pairs: Dict[str, Dict[str, "torch.Tensor"]] = {}
    for key, tensor in lora_weights.items():
        if ".lora_A." in key:
            base_key = key.replace(".lora_A.weight", "")
            lora_pairs.setdefault(base_key, {})["A"] = tensor
        elif ".lora_B." in key:
            base_key = key.replace(".lora_B.weight", "")
            lora_pairs.setdefault(base_key, {})["B"] = tensor

    task_vectors: Dict[str, "torch.Tensor"] = {}
    for base_key, pair in lora_pairs.items():
        if "A" in pair and "B" in pair:
            task_vectors[base_key] = (pair["B"] @ pair["A"]) * scaling

    return task_vectors


def _save_merged_adapter(
    *,
    method: str,
    task_names: List[str],
    adapter_paths: List[Path],
    merged_weights: Optional[Dict[str, "torch.Tensor"]],
    metadata: Dict,
) -> Path:
    if method == "weighted_delta":
        raise ValueError("weighted_delta cannot be saved as a LoRA adapter.")

    output_path = create_merge_output_path(
        method,
        task_names,
        {"lambda": metadata.get("lambda")} if metadata.get("lambda") is not None else None,
    )

    if method == "task_vector":
        from experiments.merge_vectors import merge_uniform_via_task_vectors

        merge_uniform_via_task_vectors(
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
