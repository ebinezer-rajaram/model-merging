"""Dataset helpers and data collator for emotion recognition tasks."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from datasets import Dataset, DatasetDict, load_dataset

from core import (
    add_duration,
    build_split_metadata,
    compute_speaker_stats,
    load_cached_split,
    normalize_audio,
    normalize_split_metadata,
    num_proc_map_kwargs,
    resolve_num_proc,
    save_cached_split,
    select_random_indices,
    subset_dataset_by_metadata,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATASET_CACHE_ROOT = PACKAGE_ROOT / "artifacts" / "emotion" / "datasets"

DEFAULT_DATASET_NAME = "superb"
DEFAULT_DATASET_CONFIG = "er"
DEFAULT_AUDIO_COLUMN = "audio"
FALLBACK_AUDIO_COLUMNS = ("speech",)
MANIFEST_FIELDS: Tuple[str, ...] = (
    "label",
    "text",
    "speaker",
    "gender",
    "emotion",
    "fileid",
)


def _samples_key(value: Optional[int]) -> str:
    return "full" if value is None else str(int(value)).zfill(6)


def _normalize_target_count(name: str, value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative, received {value!r}")
    return parsed


def _column_exists(dataset: DatasetDict, column: str) -> bool:
    try:
        sample_split = next(iter(dataset.values()))
    except StopIteration:
        return False
    return column in sample_split.column_names


def _rename_column(dataset: DatasetDict, source: str, target: str) -> DatasetDict:
    if not source or source == target:
        return dataset
    if not _column_exists(dataset, source):
        raise ValueError(f"Column {source!r} not found in dataset splits.")
    if _column_exists(dataset, target):
        raise ValueError(
            f"Cannot rename column {source!r} to {target!r} because {target!r} already exists."
        )
    return dataset.rename_column(source, target)


def _ensure_audio_column(
    dataset: DatasetDict, preferred: Optional[str] = None
) -> Tuple[DatasetDict, str]:
    """Rename audio column to the preferred name if needed."""
    try:
        sample_split = next(iter(dataset.values()))
    except StopIteration as exc:
        raise ValueError("Dataset has no splits to infer audio column from.") from exc

    if preferred:
        if preferred == DEFAULT_AUDIO_COLUMN and DEFAULT_AUDIO_COLUMN in sample_split.column_names:
            return dataset, DEFAULT_AUDIO_COLUMN
        if preferred in sample_split.column_names:
            if preferred != DEFAULT_AUDIO_COLUMN:
                dataset = dataset.rename_column(preferred, DEFAULT_AUDIO_COLUMN)
            return dataset, DEFAULT_AUDIO_COLUMN

    if DEFAULT_AUDIO_COLUMN in sample_split.column_names:
        return dataset, DEFAULT_AUDIO_COLUMN

    for candidate in FALLBACK_AUDIO_COLUMNS:
        if candidate in sample_split.column_names:
            dataset = dataset.rename_column(candidate, DEFAULT_AUDIO_COLUMN)
            return dataset, DEFAULT_AUDIO_COLUMN

    raise ValueError(
        f"Could not find an audio column in dataset splits. Checked {DEFAULT_AUDIO_COLUMN!r}"
        f" and {FALLBACK_AUDIO_COLUMNS!r}."
    )


def _extract_label_names(dataset: Dataset) -> List[str]:
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


def load_superb_emotion_dataset(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_config: str = DEFAULT_DATASET_CONFIG,
    max_train_samples: Optional[int] = None,
    max_validation_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    max_duration: Optional[float] = None,
    min_duration: Optional[float] = None,
    seed: int = 0,
    num_proc: Optional[int | str] = None,
    cache_dir: Optional[Path | str] = None,
    cache_splits: bool = True,
    force_rebuild: bool = False,
    label_column: Optional[str] = None,
    text_column: Optional[str] = None,
    audio_column: Optional[str] = None,
    split_percentages: Optional[Mapping[str, float] | Sequence[float]] = None,
    train_split: str = "train",
    validation_split: Optional[str] = "validation",
    test_split: Optional[str] = "test",
    stratify_by_column: Optional[str] = None,
    revision: Optional[str] = None,
    data_dir: Optional[str | Path] = None,
    normalize_audio_flag: bool = False,
    normalize_per_speaker: bool = False,
    target_rms_db: float = -25.0,
) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset], List[str]]:
    """
    Load a speech emotion recognition dataset with optional sub-sampling.

    Returns train, validation, test splits (None when missing) and label names.
    """

    max_train_samples = _normalize_target_count("max_train_samples", max_train_samples)
    max_validation_samples = _normalize_target_count("max_validation_samples", max_validation_samples)
    max_test_samples = _normalize_target_count("max_test_samples", max_test_samples)

    # Handle local MELD dataset
    if dataset_name == "local_meld":
        from .meld_loader import load_meld_from_local
        if data_dir is None:
            data_dir = "data/meld"
        dataset: DatasetDict = load_meld_from_local(data_dir=data_dir)
    else:
        dataset: DatasetDict = load_dataset(dataset_name, dataset_config, revision=revision)

    if label_column and label_column != "label":
        dataset = _rename_column(dataset, label_column, "label")
    elif not label_column and not _column_exists(dataset, "label"):
        raise ValueError(
            "Dataset must contain a 'label' column or specify 'label_column' to rename the target."
        )

    if text_column and text_column != "text":
        dataset = _rename_column(dataset, text_column, "text")

    preferred_audio_column = audio_column

    if _column_exists(dataset, "label"):
        # Ensure stratification can operate on a categorical feature.
        probe_split = (
            train_split
            if train_split and train_split in dataset
            else next(iter(dataset.keys())) if len(dataset) > 0 else None
        )
        if probe_split is not None:
            label_feature = dataset[probe_split].features.get("label")
        else:
            label_feature = None

        dtype = getattr(label_feature, "dtype", None)
        if dtype == "string":
            dataset = dataset.class_encode_column("label")

    split_ratios: Optional[Tuple[float, float, float]] = None
    if split_percentages is not None:
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

    if split_ratios is not None:
        if not train_split or train_split not in dataset:
            raise ValueError(f"Requested train split {train_split!r} not found in dataset.")
        base_train = dataset[train_split]

        stratify_column = stratify_by_column
        if stratify_column is None and "label" in base_train.column_names:
            stratify_column = "label"
        elif stratify_column and stratify_column not in base_train.column_names:
            raise ValueError(
                f"Requested stratify_by_column {stratify_column!r} not present in dataset."
            )

        train_ratio, val_ratio, test_ratio = split_ratios

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

        new_splits: Dict[str, Dataset] = {"train": train_portion}
        if val_portion is not None:
            new_splits["validation"] = val_portion
        if test_portion is not None:
            new_splits["test"] = test_portion
        dataset = DatasetDict(new_splits)
    else:
        assembled: Dict[str, Dataset] = {}
        if train_split and train_split in dataset:
            assembled["train"] = dataset[train_split]
        if validation_split and validation_split in dataset:
            assembled["validation"] = dataset[validation_split]
        if test_split and test_split in dataset:
            assembled["test"] = dataset[test_split]
        dataset = DatasetDict(assembled)

    dataset, audio_column = _ensure_audio_column(dataset, preferred=preferred_audio_column)

    effective_num_proc = resolve_num_proc(num_proc)
    cache_root = Path(cache_dir) if cache_dir is not None else DATASET_CACHE_ROOT

    # Build cache key that includes all preprocessing parameters
    norm_key = "nonorm"
    if normalize_audio_flag:
        norm_key = f"norm_spk{normalize_per_speaker}_rms{int(target_rms_db)}"
    min_dur_key = "nomin" if min_duration is None else f"min{int(min_duration*1000)}ms"
    max_dur_key = "nomax" if max_duration is None else f"max{int(max_duration)}s"

    processed_cache_dir = (
        cache_root / "processed" /
        f"{dataset_name}_{dataset_config}_{min_dur_key}_{max_dur_key}_{norm_key}"
    )

    if processed_cache_dir.exists() and not force_rebuild:
        # Load preprocessed dataset from cache
        from datasets import load_from_disk
        print(f"Loading preprocessed dataset from cache: {processed_cache_dir.name}")
        dataset = load_from_disk(str(processed_cache_dir))
    else:
        print(f"Preprocessing dataset (will be cached for future runs)...")
        map_kwargs = num_proc_map_kwargs(effective_num_proc)

        # Add duration
        duration_fn = partial(add_duration, audio_column=audio_column)
        dataset = dataset.map(duration_fn, **map_kwargs)

        # Filter by duration (min and max in a single pass)
        if min_duration is not None or max_duration is not None:
            original_sizes = {name: len(split) for name, split in dataset.items()}

            def duration_filter(example):
                duration = example.get("duration", 0)
                if min_duration is not None and duration < min_duration:
                    return False
                if max_duration is not None and duration > max_duration:
                    return False
                return True

            dataset = dataset.filter(duration_filter)
            filtered_sizes = {name: len(split) for name, split in dataset.items()}
            removed = {name: original_sizes[name] - filtered_sizes[name] for name in original_sizes}

            if any(count > 0 for count in removed.values()):
                removed_info = ", ".join(f"{name}={count}" for name, count in removed.items() if count > 0)
                filter_desc = []
                if min_duration is not None:
                    filter_desc.append(f"min={min_duration}s")
                if max_duration is not None:
                    filter_desc.append(f"max={max_duration}s")
                print(f"⚠️  Filtered out {removed_info} samples outside duration range ({', '.join(filter_desc)})")

        # Apply audio normalization if requested
        if normalize_audio_flag:
            speaker_stats = None

            # Compute speaker statistics if speaker-level normalization is enabled
            if normalize_per_speaker:
                # Use train split for computing speaker statistics
                train_data = dataset.get("train")
                if train_data is not None and "Speaker" in train_data.column_names:
                    speaker_stats = compute_speaker_stats(
                        train_data,
                        speaker_column="Speaker",
                        audio_column=audio_column,
                    )
                else:
                    print("⚠️  Speaker-level normalization requested but Speaker column not found, falling back to global normalization")

            # Apply normalization to all splits
            print(f"Normalizing audio (speaker-level={normalize_per_speaker}, target_rms={target_rms_db} dB)...")
            normalize_fn = partial(
                normalize_audio,
                audio_column=audio_column,
                speaker_column="Speaker" if normalize_per_speaker else None,
                speaker_stats=speaker_stats,
                target_rms_db=target_rms_db,
                normalize_per_speaker=normalize_per_speaker,
            )
            dataset = dataset.map(normalize_fn, **map_kwargs)

        # Save to disk for future runs
        if cache_splits:
            processed_cache_dir.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving preprocessed dataset to cache: {processed_cache_dir.name}")
            dataset.save_to_disk(str(processed_cache_dir))
    # Include max_duration in cache key to invalidate cache when filtering changes
    duration_key = "none" if max_duration is None else f"{int(max_duration)}s"
    cache_name = (
        f"{dataset_name}_{dataset_config}"
        f"_train_{_samples_key(max_train_samples)}"
        f"_val_{_samples_key(max_validation_samples)}"
        f"_test_{_samples_key(max_test_samples)}"
        f"_maxdur_{duration_key}"
        f"_seed_{int(seed)}.json"
    )
    cache_path = cache_root / cache_name

    payload = (
        load_cached_split(cache_path) if cache_splits and not force_rebuild else None
    )
    payload_seed = seed

    label_names: List[str]
    splits_metadata: Dict[str, Dict[str, Any]]
    if payload is None:
        label_names = _extract_label_names(dataset.get("train") or next(iter(dataset.values())))
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
                manifest_fields=MANIFEST_FIELDS,
                audio_column=audio_column,
            )

        if cache_splits:
            save_cached_split(
                cache_path,
                {
                    "seed": seed,
                    "dataset": dataset_name,
                    "config": dataset_config,
                    "audio_column": audio_column,
                    "label_names": label_names,
                    "splits": splits_metadata,
                },
            )
    else:
        label_names = list(payload.get("label_names", []))
        splits_metadata = normalize_split_metadata(payload.get("splits"))
        payload_seed = int(payload.get("seed", seed))

    def _select(split_name: str) -> Optional[Dataset]:
        if split_name not in dataset:
            return None
        return subset_dataset_by_metadata(dataset[split_name], splits_metadata.get(split_name))

    train_subset = _select("train")
    validation_subset = _select("validation")
    test_subset = _select("test")

    def _summarize(split_name: str, subset: Optional[Dataset], label: Optional[str] = None) -> str:
        if subset is None:
            return f"{label or split_name}=∅"
        hours = float(splits_metadata.get(split_name, {}).get("hours", 0.0) or 0.0)
        return f"{label or split_name}={len(subset)} (~{hours:.2f} h)"

    label_part = f"labels={len(label_names)} classes" if label_names else "label names unavailable"
    summary = ", ".join(
        _summarize(split_name, subset, display)
        for split_name, subset, display in (
            ("train", train_subset, "train"),
            ("validation", validation_subset, "val"),
            ("test", test_subset, "test"),
        )
    )

    print(
        "😊 Emotion dataset:"
        f" {summary} ({label_part}, seed={payload_seed}, num_proc={effective_num_proc})."
    )

    return train_subset, validation_subset, test_subset, label_names


@dataclass
class EmotionDataCollator:
    """Prepare batches for emotion recognition fine-tuning.

    Always uses chat template with both user message (audio + instruction) and assistant response (ground truth).
    During evaluation, the CustomTrainer's prediction_step will strip out the ground truth before generation.
    """

    processor: Any
    sampling_rate: int
    label_names: Sequence[str]
    include_transcript: bool = True

    def _label_to_text(self, value: Any) -> str:
        if value is None:
            return ""
        try:
            index = int(value)
        except (TypeError, ValueError):
            return str(value)
        if 0 <= index < len(self.label_names):
            return str(self.label_names[index])
        return str(index)

    def _build_instruction(self, transcript: str) -> str:
        """Build the instruction text for the user message."""
        # Format class options for the prompt
        class_options = ", ".join(self.label_names)

        # Simplified prompt with explicit output cue (matches original successful format)
        # The "Emotion:" prompt at the end provides a clear cue for the model to output the label
        instruction = (
            f"What is the speaker's emotion in the provided audio clip? "
            f"Choose from: {class_options}."
        )
        transcript = (transcript or "").strip()
        if transcript and self.include_transcript:
            instruction += f"\nTranscript: {transcript}"

        # Add explicit output cue to guide the model
        instruction += "\nEmotion:"

        return instruction

    # OLD PROMPT METHOD (kept as backup):
    # def _build_prompt(self, transcript: str) -> str:
    #     """Assemble the textual prompt for the model."""
    #     # Format class options for the prompt
    #     class_options = ", ".join(self.label_names)
    #
    #     # Optimized prompt (based on prompt comparison results)
    #     # This format performed best in zero-shot evaluation (51.5% accuracy, 0.267 F1)
    #     base = (
    #         f"{self.processor.audio_token}"
    #         f"Listen carefully to the audio and identify the speaker's emotional state. "
    #         f"Choose the most appropriate emotion from: {class_options}."
    #     )
    #     transcript = (transcript or "").strip()
    #     if transcript and self.include_transcript:
    #         base += f"\n\nTranscript: \"{transcript}\"\n\nEmotion:"
    #     else:
    #         base += "\n\nEmotion:"
    #
    #     return base

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Convert dataset rows into model-ready tensors."""
        audio_arrays = [feature["audio"]["array"] for feature in features]
        transcripts = [feature.get("text", "") for feature in features]
        label_strings = [self._label_to_text(feature.get("label")) for feature in features]

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "padding_side", None) != "left":
            tokenizer.padding_side = "left"

        # Build prompts using chat template format (matching ASR approach)
        # Always include both user message and assistant response with ground truth
        # During evaluation, CustomTrainer's prediction_step will truncate before generation
        prompts = []
        for text, label in zip(transcripts, label_strings):
            instruction = self._build_instruction(text)

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": None},
                        {"type": "text", "text": instruction}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": label}
                    ]
                }
            ]
            prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=False,
                tokenize=False
            )
            prompts.append(prompt)

        inputs = self.processor(
            audio=audio_arrays,
            sampling_rate=self.sampling_rate,
            text=prompts,
            return_tensors="pt",
            padding=True,
        )

        labels = inputs["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        audio_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.audio_token
        )

        # Mask padding and audio tokens
        labels = labels.masked_fill(labels == pad_id, -100)
        labels = labels.masked_fill(labels == audio_token_id, -100)

        # Mask everything except the assistant's emotion response
        for i, label in enumerate(label_strings):
            # Tokenize the ground truth emotion label to identify it in the full sequence
            label_tokens = tokenizer.encode(label, add_special_tokens=False)

            # Find where the label appears in the input_ids
            input_ids = inputs["input_ids"][i]
            label_length = len(label_tokens)

            # Search for the label tokens in the sequence
            found = False
            for j in range(len(input_ids) - label_length + 1):
                if torch.all(input_ids[j:j + label_length] == torch.tensor(label_tokens, device=input_ids.device)):
                    # Mask everything before the label
                    labels[i, :j] = -100
                    found = True
                    break

            # Fallback: mask based on sequence structure
            if not found:
                non_masked = (labels[i] != -100).nonzero(as_tuple=False)
                if len(non_masked) > label_length:
                    mask_until = non_masked[-label_length].item()
                    labels[i, :mask_until] = -100

        inputs["labels"] = labels
        return inputs


__all__ = ["load_superb_emotion_dataset", "EmotionDataCollator"]
