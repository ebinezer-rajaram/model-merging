"""High-level merge runner and adapter resolution."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from merging.optimizers.registry import apply_optimizer_overrides, OptimizerContext, optimize_lambda_policy
from merging.engine.registry import MergeResult, get_merge_method, normalize_params
from merging.config.specs import MergeSpec, merge_spec_from_legacy_args, merge_spec_to_params
from merging.runtime.utils import create_merge_output_path, resolve_best_adapter, save_merged_adapter
from merging.runtime.utils import build_merge_tag
from merging.runtime.utils import PACKAGE_ROOT


def resolve_adapter_specs(
    adapter_specs: List[str],
) -> List[Tuple[Path, Dict]]:
    """Resolve adapter specifications to paths and metadata."""
    resolved: List[Tuple[Path, Dict]] = []

    for spec in adapter_specs:
        path = Path(spec)
        if path.exists() and path.is_dir():
            adapter_path = path.resolve()
            metadata = {"path": str(adapter_path)}
            inferred_task = None
            try:
                from merging.runtime.utils import infer_task_from_path
                inferred_task = infer_task_from_path(str(adapter_path))
            except Exception:
                inferred_task = None
            if inferred_task:
                metadata["task"] = inferred_task
            resolved.append((adapter_path, metadata))
            print(f"✅ Using adapter at: {adapter_path}")
            continue

        try:
            adapter_path, metadata = resolve_best_adapter(spec)
            print(f"✅ Resolved '{spec}' to: {adapter_path}")
            metrics = metadata.get("metrics", {}) or {}
            # Print only a small scalar summary to avoid terminal spam.
            preferred = ("wer", "eval_wer", "accuracy", "eval_accuracy", "macro_f1", "eval_macro_f1", "eval_loss")
            shown = {k: metrics[k] for k in preferred if k in metrics}
            if not shown:
                shown = {k: v for k, v in list(metrics.items())[:6]}
            print(f"   Metrics: {shown}")
            resolved.append((adapter_path, metadata))
        except Exception as exc:
            raise ValueError(
                f"Could not resolve adapter spec '{spec}': {exc}\n"
                f"Provide either a task name (e.g., 'asr') or a valid adapter path."
            ) from exc

    return resolved


def run_merge(
    *,
    adapter_specs: List[str],
    method: str,
    lambda_weight: Optional[float],
    params: Optional[Dict[str, object]] = None,
    merge_mode: str,
    output: Optional[str],
    save_merged: bool,
    show_progress: bool = True,
    merge_spec: Optional[MergeSpec] = None,
) -> MergeResult:
    """Run a merge and optionally save a merged adapter."""
    if merge_spec is not None:
        adapter_specs = list(merge_spec.adapters)
        method = merge_spec.method
        merge_mode = merge_spec.merge_mode
        params = merge_spec_to_params(merge_spec)
        lambda_weight = merge_spec.method_params.get("lambda", lambda_weight)
    else:
        merge_spec = merge_spec_from_legacy_args(
            adapters=adapter_specs,
            method=method,
            merge_mode=merge_mode,
            lambda_weight=lambda_weight,
            params=params,
        )
        params = merge_spec_to_params(merge_spec)

    if show_progress:
        print("\n" + "=" * 60)
        print("🔀 Adapter Merging")
        print("=" * 60)
        print(f"\n📂 Resolving {len(adapter_specs)} adapter(s)...")

    resolved = resolve_adapter_specs(adapter_specs)
    adapter_paths = [path for path, _ in resolved]
    source_metadata = [meta for _, meta in resolved]
    task_names = [meta.get("task", f"adapter{i}") for i, meta in enumerate(source_metadata)]

    method_impl = get_merge_method(method)
    effective_params = normalize_params(method_impl, params=params, legacy_lambda_weight=lambda_weight)
    method_impl.validate(len(adapter_paths), effective_params)

    optimizer_result = optimize_lambda_policy(
        merge_spec,
        OptimizerContext(
            method=method,
            adapter_specs=adapter_specs,
            adapter_paths=adapter_paths,
            source_metadata=source_metadata,
            merge_mode=merge_mode,
            output_dir=Path(output) if output else None,
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

    output_path: Optional[Path] = None
    if save_merged:
        if not method_impl.saveable:
            raise ValueError(f"{method} cannot be saved as a LoRA adapter.")

        if output:
            base = Path(output)
            if not base.is_absolute():
                base = PACKAGE_ROOT / base
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = base / "runs" / f"run_{timestamp}"
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            extra_params = {}
            if effective_params.get("lambda") is not None:
                extra_params["lambda"] = effective_params["lambda"]
            if isinstance(effective_params.get("optimizer"), dict):
                extra_params["optimizer"] = effective_params["optimizer"]
            output_path = create_merge_output_path(
                method=method,
                task_names=task_names,
                extra_params=extra_params if extra_params else None,
            )

        if show_progress:
            print(f"\n💾 Output directory: {output_path}")

    merge_output = method_impl.merge_in_memory(
        adapter_paths=adapter_paths,
        source_metadata=source_metadata,
        merge_mode=merge_mode,
        params=effective_params,
    )

    if save_merged and output_path is not None:
        if method_impl.save_fn is not None:
            method_impl.save_fn(
                adapter_paths=adapter_paths,
                output_path=output_path,
                merge_mode=merge_mode,
                show_progress=show_progress,
                params=effective_params,
            )
        else:
            if merge_output.merged_weights is None:
                raise ValueError(f"{method} did not return merged weights for saving.")
            save_merged_adapter(
                weights=merge_output.merged_weights,
                output_path=output_path,
                reference_adapter_path=adapter_paths[0],
                metadata=merge_output.metadata,
                register_run=True,
            )

    merge_tag = build_merge_tag(merge_output.metadata, task_names)
    return MergeResult(
        method=method,
        params=effective_params,
        adapter_paths=adapter_paths,
        source_metadata=source_metadata,
        task_names=task_names,
        output_path=output_path,
        merge_output=merge_output,
        merge_tag=merge_tag,
    )
