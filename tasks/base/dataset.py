"""Base dataset loading utilities shared across all tasks."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from datasets import Dataset, DatasetDict, load_dataset

from core import (
    add_duration,
    build_split_metadata,
    load_cached_split,
    normalize_split_metadata,
    num_proc_map_kwargs,
    resolve_num_proc,
    save_cached_split,
    select_random_indices,
    subset_dataset_by_metadata,
)

# Common constants
DEFAULT_AUDIO_COLUMN = "audio"
FALLBACK_AUDIO_COLUMNS: Tuple[str, ...] = ("speech", "audio_path")


def _samples_key(value: Optional[int]) -> str:
    """Format sample count for cache key."""
    return "full" if value is None else str(int(value)).zfill(6)


def _normalize_target_count(name: str, value: Optional[int]) -> Optional[int]:
    """Validate and normalize a target sample count."""
    if value is None:
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative, received {value!r}")
    return parsed


def _column_exists(dataset: DatasetDict, column: str) -> bool:
    """Check if a column exists across all splits in the dataset."""
    try:
        sample_split = next(iter(dataset.values()))
    except StopIteration:
        return False
    return column in sample_split.column_names


def _rename_column(dataset: DatasetDict, source: str, target: str) -> DatasetDict:
    """Rename a column across all splits."""
    if not source or source == target:
        return dataset
    if not _column_exists(dataset, source):
        raise ValueError(f"Column {source!r} not found in dataset splits.")
    if _column_exists(dataset, target):
        raise ValueError(
            f"Cannot rename column {source!r} to {target!r} because {target!r} already exists."
        )
    return dataset.rename_column(source, target)


def _select_candidate_column(dataset: DatasetDict, candidates: Iterable[str]) -> Optional[str]:
    """Select the first available column from a list of candidates."""
    for name in candidates:
        if _column_exists(dataset, name):
            return name
    return None


def _ensure_audio_column(
    dataset: DatasetDict,
    preferred: Optional[str] = None,
    fallback_columns: Tuple[str, ...] = FALLBACK_AUDIO_COLUMNS,
) -> Tuple[DatasetDict, str]:
    """Ensure the dataset has an audio column named DEFAULT_AUDIO_COLUMN."""
    try:
        sample_split = next(iter(dataset.values()))
    except StopIteration as exc:
        raise ValueError("Dataset has no splits to infer audio column from.") from exc

    # Check preferred column first
    if preferred:
        if preferred == DEFAULT_AUDIO_COLUMN and DEFAULT_AUDIO_COLUMN in sample_split.column_names:
            return dataset, DEFAULT_AUDIO_COLUMN
        if preferred in sample_split.column_names:
            if preferred != DEFAULT_AUDIO_COLUMN:
                dataset = dataset.rename_column(preferred, DEFAULT_AUDIO_COLUMN)
            return dataset, DEFAULT_AUDIO_COLUMN

    # Check if default column already exists
    if DEFAULT_AUDIO_COLUMN in sample_split.column_names:
        return dataset, DEFAULT_AUDIO_COLUMN

    # Try fallback columns
    for candidate in fallback_columns:
        if candidate in sample_split.column_names:
            dataset = dataset.rename_column(candidate, DEFAULT_AUDIO_COLUMN)
            return dataset, DEFAULT_AUDIO_COLUMN

    raise ValueError(
        f"Could not find an audio column in dataset splits. Checked {DEFAULT_AUDIO_COLUMN!r}"
        f" and {fallback_columns!r}."
    )


def _extract_label_names(dataset: Dataset) -> List[str]:
    """Extract label names from a dataset's label feature."""
    feature = dataset.features.get("label")
    if feature is not None:
        names = getattr(feature, "names", None)
        if names:
            return list(names)
        dtype = getattr(feature, "dtype", None)
        if dtype == "string":
            try:
                unique_values = dataset.unique("label")
            except (KeyError, TypeError):
                unique_values = None
            if unique_values:
                return sorted(str(value) for value in unique_values if value not in (None, ""))
    return []


def apply_split_percentages(
    dataset: DatasetDict,
    split_percentages: Mapping[str, float] | Sequence[float],
    train_split: str = "train",
    seed: int = 0,
    stratify_by_column: Optional[str] = None,
) -> DatasetDict:
    """Split a train set into train/validation/test according to specified ratios.

    Args:
        dataset: Dataset with at least a train split
        split_percentages: Either a dict {"train": 0.7, "validation": 0.15, "test": 0.15}
                          or a sequence [0.7, 0.15, 0.15]
        train_split: Name of the split to divide
        seed: Random seed for reproducibility
        stratify_by_column: Column to stratify by (typically "label")

    Returns:
        New DatasetDict with train/validation/test splits
    """
    if isinstance(split_percentages, Mapping):
        train_ratio = float(split_percentages.get("train", 0.0))
        val_ratio = float(split_percentages.get("validation", 0.0))
        test_ratio = float(split_percentages.get("test", 0.0))
        split_ratios = (train_ratio, val_ratio, test_ratio)
    else:
        ratios = list(split_percentages)
        if len(ratios) != 3:
            raise ValueError("split_percentages must contain exactly three values (train/val/test).")
        split_ratios = tuple(float(value) for value in ratios)

    total = sum(split_ratios)
    if total <= 0:
        raise ValueError("split_percentages must sum to a positive value.")
    split_ratios = tuple(value / total for value in split_ratios)

    if not train_split or train_split not in dataset:
        raise ValueError(f"Requested train split {train_split!r} not found in dataset.")
    base_train = dataset[train_split]

    # Determine stratification column
    stratify_column = stratify_by_column
    if stratify_column is None and "label" in base_train.column_names:
        stratify_column = "label"
    elif stratify_column and stratify_column not in base_train.column_names:
        raise ValueError(
            f"Requested stratify_by_column {stratify_column!r} not present in dataset."
        )

    train_ratio, val_ratio, test_ratio = split_ratios

    # Split off test set if needed
    if not (0.0 <= test_ratio < 1.0):
        raise ValueError("Test split percentage must be in the range [0, 1).")

    if test_ratio > 0.0:
        first_split = base_train.train_test_split(
            test_size=test_ratio,
            seed=seed,
            stratify_by_column=stratify_column,
        )
        train_portion = first_split["train"]
        test_portion = first_split["test"]
    else:
        train_portion = base_train
        test_portion = None

    # Split off validation set if needed
    if val_ratio > 0.0:
        if train_ratio + val_ratio <= 0:
            raise ValueError("Train and validation split percentages must be positive.")
        val_fraction = val_ratio / (train_ratio + val_ratio)
        second_split = train_portion.train_test_split(
            test_size=val_fraction,
            seed=seed + 1,
            stratify_by_column=stratify_column,
        )
        train_portion = second_split["train"]
        val_portion = second_split["test"]
    else:
        val_portion = None

    # Assemble new dataset
    new_splits: Dict[str, Dataset] = {"train": train_portion}
    if val_portion is not None:
        new_splits["validation"] = val_portion
    if test_portion is not None:
        new_splits["test"] = test_portion
    return DatasetDict(new_splits)


def assemble_splits(
    dataset: DatasetDict,
    train_split: str = "train",
    validation_split: Optional[str] = "validation",
    test_split: Optional[str] = "test",
) -> DatasetDict:
    """Assemble a standardized DatasetDict from arbitrary split names.

    Args:
        dataset: Source dataset with arbitrary split names
        train_split: Name of the training split in the source dataset
        validation_split: Name of the validation split (None to skip)
        test_split: Name of the test split (None to skip)

    Returns:
        DatasetDict with standardized split names ("train", "validation", "test")
    """
    assembled: Dict[str, Dataset] = {}
    if train_split and train_split in dataset:
        assembled["train"] = dataset[train_split]
    if validation_split and validation_split in dataset:
        assembled["validation"] = dataset[validation_split]
    if test_split and test_split in dataset:
        assembled["test"] = dataset[test_split]
    return DatasetDict(assembled)


def prepare_classification_dataset(
    dataset: DatasetDict,
    *,
    label_column: Optional[str] = None,
    fallback_label_columns: Tuple[str, ...] = ("intent", "emotion", "speaker", "label"),
    text_column: Optional[str] = None,
    fallback_text_columns: Tuple[str, ...] = ("text", "transcript", "sentence"),
) -> DatasetDict:
    """Prepare a dataset for classification by normalizing column names.

    This handles:
    - Renaming label column to "label"
    - Encoding string labels as integers
    - Optionally renaming text column to "text"

    Args:
        dataset: Input dataset
        label_column: Explicit label column name
        fallback_label_columns: Columns to try if label_column not specified
        text_column: Explicit text column name
        fallback_text_columns: Columns to try if text_column not specified

    Returns:
        Dataset with normalized columns
    """
    # Normalize label column
    if label_column and label_column != "label":
        dataset = _rename_column(dataset, label_column, "label")
    elif not _column_exists(dataset, "label"):
        fallback_label = _select_candidate_column(dataset, fallback_label_columns)
        if fallback_label:
            dataset = _rename_column(dataset, fallback_label, "label")
        else:
            raise ValueError(
                f"Dataset must contain a label column. Tried: {fallback_label_columns}. "
                "Provide 'label_column' explicitly."
            )

    # Normalize text column (optional)
    if text_column and text_column != "text":
        dataset = _rename_column(dataset, text_column, "text")
    elif not _column_exists(dataset, "text"):
        fallback_text = _select_candidate_column(dataset, fallback_text_columns)
        if fallback_text and fallback_text != "text":
            dataset = _rename_column(dataset, fallback_text, "text")

    # Encode string labels as integers if needed
    if _column_exists(dataset, "label"):
        probe_split = next(iter(dataset.keys())) if len(dataset) > 0 else None
        if probe_split is not None:
            label_feature = dataset[probe_split].features.get("label")
            dtype = getattr(label_feature, "dtype", None)
            if dtype == "string":
                dataset = dataset.class_encode_column("label")

    return dataset


def cache_and_sample_splits(
    dataset: DatasetDict,
    *,
    cache_path: Path,
    max_train_samples: Optional[int] = None,
    max_validation_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    seed: int = 0,
    manifest_fields: Tuple[str, ...] = (),
    audio_column: str = DEFAULT_AUDIO_COLUMN,
    cache_splits: bool = True,
    force_rebuild: bool = False,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset], Dict[str, Dict[str, Any]], int]:
    """Cache and sample splits with metadata.

    Returns:
        (train_subset, val_subset, test_subset, splits_metadata, payload_seed)
    """
    payload = load_cached_split(cache_path) if cache_splits and not force_rebuild else None
    payload_seed = seed

    splits_metadata: Dict[str, Dict[str, Any]]
    if payload is None:
        split_targets = {
            "train": (max_train_samples, seed),
            "validation": (max_validation_samples, seed + 1),
            "test": (max_test_samples, seed + 2),
        }

        splits_metadata = {}
        for split_name, (target_count, split_seed) in split_targets.items():
            if split_name not in dataset:
                continue
            split_ds = dataset[split_name]
            indices = select_random_indices(len(split_ds), target_count, split_seed)
            splits_metadata[split_name] = build_split_metadata(
                split_ds,
                indices,
                manifest_fields=manifest_fields,
                audio_column=audio_column,
            )

        if cache_splits:
            cache_payload = {
                "seed": seed,
                "splits": splits_metadata,
            }
            if additional_metadata:
                cache_payload.update(additional_metadata)
            save_cached_split(cache_path, cache_payload)
    else:
        splits_metadata = normalize_split_metadata(payload.get("splits"))
        payload_seed = int(payload.get("seed", seed))

    def _select(split_name: str) -> Optional[Dataset]:
        if split_name not in dataset:
            return None
        return subset_dataset_by_metadata(dataset[split_name], splits_metadata.get(split_name))

    train_subset = _select("train")
    validation_subset = _select("validation")
    test_subset = _select("test")

    return train_subset, validation_subset, test_subset, splits_metadata, payload_seed


def add_duration_to_dataset(
    dataset: DatasetDict,
    audio_column: str = DEFAULT_AUDIO_COLUMN,
    num_proc: Optional[int | str] = None,
) -> DatasetDict:
    """Add duration field to all splits in a dataset."""
    effective_num_proc = resolve_num_proc(num_proc)
    map_kwargs = num_proc_map_kwargs(effective_num_proc)
    duration_fn = partial(add_duration, audio_column=audio_column)
    return dataset.map(duration_fn, **map_kwargs)


def load_and_prepare_dataset(
    dataset_name: str,
    dataset_config: Optional[str] = None,
    *,
    label_column: Optional[str] = None,
    fallback_label_columns: Tuple[str, ...] = ("label", "intent", "emotion", "speaker"),
    text_column: Optional[str] = None,
    fallback_text_columns: Tuple[str, ...] = ("text", "transcript", "sentence"),
    audio_column: Optional[str] = None,
    split_percentages: Optional[Mapping[str, float] | Sequence[float]] = None,
    train_split: str = "train",
    validation_split: Optional[str] = "validation",
    test_split: Optional[str] = "test",
    stratify_by_column: Optional[str] = None,
    seed: int = 0,
    revision: Optional[str] = None,
) -> Tuple[DatasetDict, str]:
    """Load and prepare a dataset with standardized preprocessing.

    This is a high-level helper that:
    1. Loads the dataset from HuggingFace
    2. Normalizes column names (label, text, audio)
    3. Applies split percentages if specified
    4. Assembles standard split names

    Args:
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration name
        label_column: Explicit label column name
        fallback_label_columns: Columns to try for labels
        text_column: Explicit text column name
        fallback_text_columns: Columns to try for text
        audio_column: Explicit audio column name
        split_percentages: Optional split ratios
        train_split: Source train split name
        validation_split: Source validation split name
        test_split: Source test split name
        stratify_by_column: Column to stratify splits by
        seed: Random seed
        revision: Dataset revision

    Returns:
        (dataset, audio_column_name)
    """
    # Load dataset
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, revision=revision)
    else:
        dataset = load_dataset(dataset_name, revision=revision)

    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected DatasetDict, received {type(dataset)!r}")

    # Prepare for classification (normalize columns)
    dataset = prepare_classification_dataset(
        dataset,
        label_column=label_column,
        fallback_label_columns=fallback_label_columns,
        text_column=text_column,
        fallback_text_columns=fallback_text_columns,
    )

    # Apply split percentages if specified
    if split_percentages is not None:
        dataset = apply_split_percentages(
            dataset,
            split_percentages=split_percentages,
            train_split=train_split,
            seed=seed,
            stratify_by_column=stratify_by_column,
        )
    else:
        # Assemble standard split names
        dataset = assemble_splits(
            dataset,
            train_split=train_split,
            validation_split=validation_split,
            test_split=test_split,
        )

    # Ensure audio column exists and is named correctly
    dataset, audio_column_name = _ensure_audio_column(dataset, preferred=audio_column)

    return dataset, audio_column_name


def print_dataset_summary(
    task_emoji: str,
    task_name: str,
    train_subset: Optional[Dataset],
    validation_subset: Optional[Dataset],
    test_subset: Optional[Dataset],
    splits_metadata: Dict[str, Dict[str, Any]],
    label_names: Optional[List[str]] = None,
    seed: int = 0,
    num_proc: Optional[int] = None,
    extra_info: str = "",
) -> None:
    """Print a standardized dataset summary.

    Args:
        task_emoji: Emoji for the task (e.g., "📝", "😊", "🗣️")
        task_name: Human-readable task name
        train_subset: Training dataset
        validation_subset: Validation dataset
        test_subset: Test dataset
        splits_metadata: Metadata about splits
        label_names: Optional list of label names
        seed: Random seed used
        num_proc: Number of processes used
        extra_info: Additional info to append
    """
    def _summarize(split_name: str, subset: Optional[Dataset], label: Optional[str] = None) -> str:
        if subset is None:
            return f"{label or split_name}=∅"
        hours = float(splits_metadata.get(split_name, {}).get("hours", 0.0) or 0.0)
        return f"{label or split_name}={len(subset)} (~{hours:.2f} h)"

    label_part = (
        f"labels={len(label_names)} classes" if label_names else ""
    )
    summary = ", ".join(
        _summarize(split_name, subset, display)
        for split_name, subset, display in (
            ("train", train_subset, "train"),
            ("validation", validation_subset, "val"),
            ("test", test_subset, "test"),
        )
    )

    parts = [summary]
    if label_part:
        parts.append(label_part)
    parts.append(f"seed={seed}")
    if num_proc is not None:
        parts.append(f"num_proc={num_proc}")
    if extra_info:
        parts.append(extra_info)

    print(f"{task_emoji} {task_name}: {', '.join(parts)}.")


__all__ = [
    "DEFAULT_AUDIO_COLUMN",
    "FALLBACK_AUDIO_COLUMNS",
    "_samples_key",
    "_normalize_target_count",
    "_column_exists",
    "_rename_column",
    "_select_candidate_column",
    "_ensure_audio_column",
    "_extract_label_names",
    "apply_split_percentages",
    "assemble_splits",
    "prepare_classification_dataset",
    "cache_and_sample_splits",
    "add_duration_to_dataset",
    "load_and_prepare_dataset",
    "print_dataset_summary",
]
