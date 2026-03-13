"""Resolver for adapter/artifact specs into concrete delta sources."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from merging.artifacts.continual_format import is_continual_artifact_dir
from merging.delta_sources.base import DeltaSource
from merging.delta_sources.compressed_source import CompressedMergedDeltaSource
from merging.delta_sources.lora_source import LoRADeltaSource
from merging.runtime.utils import resolve_best_adapter


def resolve_delta_source(spec: str | Path | DeltaSource) -> DeltaSource:
    """Resolve a source spec into a delta source implementation."""
    if hasattr(spec, "materialize_dense_param_delta") and hasattr(spec, "list_target_params"):
        return spec  # type: ignore[return-value]

    path = Path(spec).expanduser() if not isinstance(spec, Path) else spec.expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    if path.exists() and path.is_dir():
        if is_continual_artifact_dir(path):
            return CompressedMergedDeltaSource(path)
        if (path / "adapter_model.safetensors").exists() and (path / "adapter_config.json").exists():
            return LoRADeltaSource(path)
        raise ValueError(
            f"Directory '{path}' is neither a PEFT adapter directory nor a continual artifact directory."
        )

    # Fallback: treat as task name for best-adapter resolution.
    spec_str = str(spec)
    adapter_path, metadata = resolve_best_adapter(spec_str)
    task_name = metadata.get("task") if isinstance(metadata, dict) else None
    return LoRADeltaSource(adapter_path, task=task_name if isinstance(task_name, str) else None)


def resolve_delta_sources(specs: Iterable[str | Path | DeltaSource]) -> List[DeltaSource]:
    return [resolve_delta_source(spec) for spec in specs]


__all__ = ["resolve_delta_source", "resolve_delta_sources"]
