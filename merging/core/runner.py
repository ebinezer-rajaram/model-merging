"""High-level merge runner and adapter resolution."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from merging.core.registry import MergeOutput, get_merge_method
from merging.core.utils import create_merge_output_path, resolve_best_adapter, save_merged_adapter


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
            resolved.append((adapter_path, metadata))
            print(f"✅ Using adapter at: {adapter_path}")
            continue

        try:
            adapter_path, metadata = resolve_best_adapter(spec)
            print(f"✅ Resolved '{spec}' to: {adapter_path}")
            print(f"   Metrics: {metadata.get('metrics', {})}")
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
) -> Tuple[Optional[Path], MergeOutput, List[str]]:
    """Run a merge and optionally save a merged adapter."""
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
    effective_params = dict(params or {})
    if lambda_weight is not None:
        effective_params.setdefault("lambda", lambda_weight)
    method_impl.validate(len(adapter_paths), effective_params)

    output_path: Optional[Path] = None
    if save_merged:
        if not method_impl.saveable:
            raise ValueError(f"{method} cannot be saved as a LoRA adapter.")

        if output:
            base = Path(output)
            if not base.is_absolute():
                base = Path(__file__).resolve().parents[1] / base
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = base / "runs" / f"run_{timestamp}"
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            extra_params = {}
            if lambda_weight is not None:
                extra_params["lambda"] = lambda_weight
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

    return output_path, merge_output, task_names
