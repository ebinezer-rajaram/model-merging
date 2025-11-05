"""Shared helpers for speech dataset preparation."""

from __future__ import annotations

import json
from multiprocessing import cpu_count
from pathlib import Path
from random import Random
from typing import Any, Dict, List, Optional, Sequence

from datasets import Dataset

__all__ = [
    "safe_cpu_count",
    "resolve_num_proc",
    "add_duration",
    "hours_to_seconds",
    "hours_key",
    "select_indices_by_duration",
    "select_random_indices",
    "compute_split_hours",
    "build_manifest",
    "load_cached_split",
    "save_cached_split",
    "num_proc_map_kwargs",
    "build_split_metadata",
    "subset_dataset_by_metadata",
    "normalize_split_metadata",
]


def safe_cpu_count() -> int:
    """Return a robust CPU count, falling back to 1 on failure."""
    try:
        count = cpu_count() or 1
    except NotImplementedError:
        count = 1
    return max(1, int(count))


def resolve_num_proc(num_proc: Optional[int | str]) -> Optional[int]:
    """Normalize num_proc values, handling the 'auto' sentinel."""
    if isinstance(num_proc, str):
        if num_proc.lower() != "auto":
            raise ValueError(f"Unsupported num_proc value: {num_proc}")
        return max(1, safe_cpu_count() // 2)
    if num_proc is None:
        return max(1, safe_cpu_count() // 2)
    return max(1, int(num_proc))


def num_proc_map_kwargs(num_proc: int) -> Dict[str, Any]:
    """Return map kwargs optimized for the provided num_proc."""
    return {"num_proc": num_proc} if num_proc > 1 else {}


def add_duration(
    batch: Dict[str, Any],
    *,
    audio_column: str = "audio",
    target_field: str = "duration",
) -> Dict[str, Any]:
    """Attach utterance duration (in seconds) using the provided audio column."""
    audio = batch.get(audio_column)
    if isinstance(audio, dict):
        array = audio.get("array")
        sampling_rate = audio.get("sampling_rate")
    else:
        array = getattr(audio, "array", None)
        sampling_rate = getattr(audio, "sampling_rate", None)

    if array is None or sampling_rate in (None, 0):
        batch[target_field] = 0.0
        return batch

    try:
        length = len(array)
    except TypeError:
        length = getattr(array, "shape", (0,))[0]

    batch[target_field] = float(length) / float(sampling_rate) if sampling_rate else 0.0
    return batch


def hours_to_seconds(hours: float | int | None) -> float:
    """Convert hours to seconds."""
    if hours is None:
        return 0.0
    return float(hours) * 3600.0


def hours_key(hours: float | int | None) -> str:
    """Create a filesystem-friendly key representing requested hours."""
    if hours is None:
        return "full"
    return str(int(float(hours) * 1000)).zfill(6)


def select_indices_by_duration(
    durations: Sequence[float],
    target_hours: float | int | None,
    seed: int,
) -> List[int]:
    """Randomly select indices until the cumulative duration matches the target."""
    if target_hours is None:
        return list(range(len(durations)))

    target_seconds = hours_to_seconds(target_hours)
    if target_seconds <= 0:
        return []

    rng = Random(seed)
    order = list(range(len(durations)))
    rng.shuffle(order)

    selected: List[int] = []
    cumulative = 0.0
    for idx in order:
        cumulative += durations[idx]
        selected.append(idx)
        if cumulative >= target_seconds:
            break

    return selected


def select_random_indices(
    total_items: int,
    target_count: Optional[int],
    seed: int,
) -> List[int]:
    """Randomly choose target_count indices out of total_items."""
    if target_count is None or target_count >= total_items:
        return list(range(total_items))
    if target_count <= 0:
        return []
    rng = Random(seed)
    indices = list(range(total_items))
    rng.shuffle(indices)
    return indices[:target_count]


def compute_split_hours(
    dataset: Dataset,
    indices: Sequence[int],
    *,
    duration_field: str = "duration",
) -> float:
    """Compute total duration (in hours) for a subset of dataset indices."""
    if not indices or duration_field not in dataset.column_names:
        return 0.0
    total_seconds = sum(float(dataset[duration_field][i]) for i in indices)
    return total_seconds / 3600.0


def build_manifest(
    ds: Dataset,
    indices: Sequence[int],
    *,
    audio_column: str = "audio",
    duration_field: str = "duration",
    fields: Optional[Sequence[str]] = None,
    include_index: bool = False,
) -> List[Dict[str, Any]]:
    """Create a lightweight manifest describing dataset samples."""
    manifest: List[Dict[str, Any]] = []
    if not indices:
        return manifest

    field_set = list(fields or [])
    for position, idx in enumerate(indices):
        example = ds[int(idx)]
        if not isinstance(example, dict):
            raise TypeError(
                f"Expected dataset row to be a mapping, received {type(example)!r}"
            )

        entry: Dict[str, Any] = {}
        if include_index:
            entry["index"] = int(idx)
            entry["order"] = position

        for field in field_set:
            entry[field] = example.get(field)

        if duration_field in example and example[duration_field] is not None:
            entry[duration_field] = float(example[duration_field])

        audio_info = example.get(audio_column) if audio_column else None
        if audio_info is not None:
            if isinstance(audio_info, dict):
                entry.setdefault("audio_path", audio_info.get("path"))
            else:
                entry.setdefault("audio_path", getattr(audio_info, "path", None))

        manifest.append(entry)
    return manifest


def build_split_metadata(
    dataset: Dataset,
    indices: Sequence[int],
    *,
    manifest_fields: Sequence[str],
    audio_column: str = "audio",
    duration_field: str = "duration",
) -> Dict[str, Any]:
    """Construct a metadata payload describing a dataset subset."""
    index_list = [int(idx) for idx in indices]
    return {
        "indices": index_list,
        "hours": compute_split_hours(dataset, index_list, duration_field=duration_field),
        "manifest": build_manifest(
            dataset,
            index_list,
            audio_column=audio_column,
            duration_field=duration_field,
            fields=manifest_fields,
        ),
    }


def subset_dataset_by_metadata(dataset: Dataset, metadata: Optional[Dict[str, Any]]) -> Dataset:
    """Select rows from a dataset using cached metadata indices."""
    if metadata is None:
        return dataset
    indices = metadata.get("indices")
    if indices is None:
        return dataset
    if not indices:
        return dataset.select([])
    return dataset.select(indices)


def normalize_split_metadata(
    raw_splits: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """Coerce cached split metadata into a canonical structure."""
    normalized: Dict[str, Dict[str, Any]] = {}
    for name, info in (raw_splits or {}).items():
        metadata = dict(info or {})
        indices = metadata.get("indices")
        metadata["indices"] = (
            [int(idx) for idx in indices]
            if indices is not None
            else None
        )
        if "hours" in metadata and metadata["hours"] is not None:
            metadata["hours"] = float(metadata["hours"])
        normalized[name] = metadata
    return normalized


def load_cached_split(cache_path: Path) -> Optional[Dict[str, Any]]:
    """Load a cached dataset split manifest if it exists."""
    if not cache_path.exists():
        return None
    with cache_path.open("r") as handle:
        return json.load(handle)


def save_cached_split(cache_path: Path, payload: Dict[str, Any]) -> None:
    """Persist a dataset split manifest for future reuse."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
