"""Dataset helpers and data collator for speaker identification tasks."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from datasets import Dataset, DatasetDict

from core import resolve_num_proc
from core.tasks.dataset import (
    _column_exists,
    _extract_label_names,
    _normalize_target_count,
    _samples_key,
    add_duration_to_dataset,
    cache_and_sample_splits,
    filter_by_duration,
    load_and_prepare_dataset,
    print_dataset_summary,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATASET_CACHE_ROOT = PACKAGE_ROOT / "artifacts" / "speaker_id" / "datasets"

DEFAULT_DATASET_NAME = "speechcolab/voxceleb1"
FALLBACK_LABEL_COLUMNS: Tuple[str, ...] = ("speaker", "speaker_id", "spk_id")
FALLBACK_TEXT_COLUMNS: Tuple[str, ...] = ("text", "transcription", "transcript")

MANIFEST_FIELDS: Tuple[str, ...] = (
    "label",
    "speaker",
    "speaker_id",
    "path",
    "file",
    "text",
)


def _duration_key(min_duration: Optional[float], max_duration: Optional[float]) -> str:
    if min_duration is None and max_duration is None:
        return "all"
    parts: List[str] = []
    if min_duration is not None:
        parts.append(f"min{min_duration:g}")
    if max_duration is not None:
        parts.append(f"max{max_duration:g}")
    return "_".join(parts).replace(".", "p")


def _select_speakers(
    dataset: DatasetDict,
    *,
    max_speakers: Optional[int],
    seed: int,
    train_split: Optional[str],
) -> Tuple[DatasetDict, Optional[List[str]]]:
    """Select speakers with the most samples from the dataset."""
    if max_speakers is None:
        return dataset, None

    if train_split and train_split in dataset:
        base_split = dataset[train_split]
    else:
        try:
            base_split = next(iter(dataset.values()))
        except StopIteration:
            return dataset, None

    try:
        speaker_pool = list(base_split.unique("label"))
    except (KeyError, TypeError):
        return dataset, None

    if not speaker_pool:
        return dataset, None

    if len(speaker_pool) <= max_speakers:
        selected = speaker_pool
    else:
        # Count samples per speaker and select top N by sample count
        from collections import Counter
        speaker_counts = Counter(base_split["label"])
        # Sort speakers by count (descending) and take top max_speakers
        top_speakers = [spk for spk, _ in speaker_counts.most_common(max_speakers)]
        selected = top_speakers

    normalized = {str(item) for item in selected}

    # Use fast Arrow-based filtering instead of Python function
    # This is MUCH faster than .filter() for large datasets
    filtered: Dict[str, Dataset] = {}
    for split_name, split_ds in dataset.items():
        # Get indices where label is in the allowed set
        labels = split_ds["label"]
        indices = [i for i, label in enumerate(labels) if str(label) in normalized]
        # Select by indices (very fast with Arrow)
        filtered_split = split_ds.select(indices)
        filtered[split_name] = filtered_split
    return DatasetDict(filtered), [str(item) for item in selected]


def _limit_samples_per_speaker(
    dataset: DatasetDict,
    *,
    max_samples: Optional[int],
    seed: int,
) -> DatasetDict:
    """Limit the number of samples per speaker."""
    if max_samples is None or max_samples <= 0:
        return dataset

    rng = random.Random(seed)
    limited: Dict[str, Dataset] = {}

    for split_name, split_ds in dataset.items():
        labels = split_ds["label"] if "label" in split_ds.column_names else []
        buckets: Dict[str, List[int]] = {}
        for idx, label in enumerate(labels):
            key = str(label)
            buckets.setdefault(key, []).append(idx)

        selected_indices: List[int] = []
        for indices in buckets.values():
            if len(indices) <= max_samples:
                selected_indices.extend(indices)
            else:
                selected_indices.extend(sorted(rng.sample(indices, max_samples)))
        selected_indices.sort()
        limited[split_name] = split_ds.select(selected_indices)

    return DatasetDict(limited)


def load_voxceleb_speaker_dataset(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_config: Optional[str] = None,
    max_train_samples: Optional[int] = None,
    max_validation_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    max_speakers: Optional[int] = None,
    max_samples_per_speaker: Optional[int] = None,
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
    data_dir: Optional[str] = None,
) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset], List[str]]:
    """
    Load a speaker identification dataset derived from VoxCeleb (or compatible format).

    Returns train, validation, test splits (None when missing) and label names.
    """
    # Normalize sample counts
    max_train_samples = _normalize_target_count("max_train_samples", max_train_samples)
    max_validation_samples = _normalize_target_count("max_validation_samples", max_validation_samples)
    max_test_samples = _normalize_target_count("max_test_samples", max_test_samples)
    max_speakers = _normalize_target_count("max_speakers", max_speakers)
    max_samples_per_speaker = _normalize_target_count("max_samples_per_speaker", max_samples_per_speaker)

    # Load and prepare dataset WITHOUT splitting first (speaker filtering needs to happen before splits)
    dataset, audio_column_name = load_and_prepare_dataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        label_column=label_column,
        fallback_label_columns=FALLBACK_LABEL_COLUMNS,
        text_column=text_column,
        fallback_text_columns=FALLBACK_TEXT_COLUMNS,
        audio_column=audio_column,
        split_percentages=None,  # Don't split yet - do it after speaker filtering
        train_split=train_split,
        validation_split=validation_split,
        test_split=test_split,
        stratify_by_column=stratify_by_column,
        seed=seed,
        data_dir=data_dir,
    )

    # Apply speaker-specific filtering BEFORE creating splits
    dataset, selected_speakers = _select_speakers(
        dataset,
        max_speakers=max_speakers,
        seed=seed,
        train_split="train" if "train" in dataset else None,
    )
    dataset = _limit_samples_per_speaker(
        dataset,
        max_samples=max_samples_per_speaker,
        seed=seed + 17,
    )

    # Now apply split percentages after filtering
    if split_percentages:
        from core.tasks.dataset import apply_split_percentages
        dataset = apply_split_percentages(
            dataset,
            split_percentages=split_percentages,
            train_split=train_split,
            seed=seed,
            stratify_by_column=stratify_by_column or "label",
        )

    # Build cache directory path
    cache_root = Path(cache_dir) if cache_dir is not None else DATASET_CACHE_ROOT
    effective_num_proc = resolve_num_proc(num_proc)

    # Add duration information (with caching)
    dataset = add_duration_to_dataset(
        dataset,
        audio_column=audio_column_name,
        num_proc=effective_num_proc,
        cache_dir=cache_root if not force_rebuild else None,
    )

    # Filter by duration if specified (with caching)
    dataset = filter_by_duration(
        dataset,
        max_duration=max_duration,
        min_duration=min_duration,
        cache_dir=cache_root if not force_rebuild else None,
    )

    # Build cache path for sampled splits
    dataset_key = dataset_name.replace("/", "_")
    config_key = dataset_config or "default"
    speakers_key = "all" if max_speakers is None else str(max_speakers).zfill(4)
    per_speaker_key = "all" if max_samples_per_speaker is None else str(max_samples_per_speaker).zfill(4)
    duration_key = _duration_key(min_duration, max_duration)
    cache_name = (
        f"{dataset_key}_{config_key}"
        f"_speakers_{speakers_key}_per_{per_speaker_key}"
        f"_dur_{duration_key}"
        f"_train_{_samples_key(max_train_samples)}"
        f"_val_{_samples_key(max_validation_samples)}"
        f"_test_{_samples_key(max_test_samples)}"
        f"_seed_{int(seed)}.json"
    )
    cache_path = cache_root / cache_name

    # Extract label names - use selected_speakers if available (actual speaker IDs from data)
    # Otherwise fall back to ClassLabel feature names
    label_probe = dataset.get("train") or next(iter(dataset.values()))
    if selected_speakers:
        label_names = selected_speakers  # Use actual speaker IDs like ['2613', '1638', ...]
    else:
        label_names = _extract_label_names(label_probe)  # Fall back to ClassLabel feature

    # Cache and sample splits
    train_subset, validation_subset, test_subset, splits_metadata, payload_seed = cache_and_sample_splits(
        dataset,
        cache_path=cache_path,
        max_train_samples=max_train_samples,
        max_validation_samples=max_validation_samples,
        max_test_samples=max_test_samples,
        seed=seed,
        manifest_fields=MANIFEST_FIELDS,
        audio_column=audio_column_name,
        cache_splits=cache_splits,
        force_rebuild=force_rebuild,
        additional_metadata={
            "dataset": dataset_name,
            "config": dataset_config,
            "audio_column": audio_column_name,
            "label_names": label_names,
            "selected_speakers": selected_speakers,
        },
    )

    # Print summary
    speaker_info = f", {len(selected_speakers)} speakers" if selected_speakers else ""
    print_dataset_summary(
        task_emoji="🗣️",
        task_name="Speaker-ID dataset",
        train_subset=train_subset,
        validation_subset=validation_subset,
        test_subset=test_subset,
        splits_metadata=splits_metadata,
        label_names=label_names,
        seed=payload_seed,
        num_proc=effective_num_proc,
        extra_info=speaker_info.lstrip(", "),
    )

    return train_subset, validation_subset, test_subset, label_names


@dataclass
class SpeakerIdentificationCollator:
    """Prepare batches for speaker identification fine-tuning.

    Always uses chat template with both user message (audio + instruction) and assistant response (ground truth).
    During evaluation, the CustomTrainer's prediction_step will strip out the ground truth before generation.
    """

    processor: Any
    sampling_rate: int
    label_names: Sequence[str]
    include_transcript: bool = False
    max_audio_length: Optional[float] = None  # Maximum audio duration in seconds (trim if longer)

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
        transcript = (transcript or "").strip()
        # Simplified prompt without listing all speakers (prevents OOM with large speaker counts)
        # Guide model to output numeric ID format
        instruction = "Who is the speaker in the provided audio segment? Output only the numeric speaker ID (e.g., 1234)."

        if transcript and self.include_transcript:
            instruction += f"\nTranscript: {transcript}"

        return instruction

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_arrays = [feature["audio"]["array"] for feature in features]

        # Trim audio if max_audio_length is specified
        if self.max_audio_length is not None:
            max_samples = int(self.max_audio_length * self.sampling_rate)
            audio_arrays = [arr[:max_samples] if len(arr) > max_samples else arr for arr in audio_arrays]

        transcripts = [feature.get("text", "") for feature in features]
        label_strings = [self._label_to_text(feature.get("label")) for feature in features]

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "padding_side", None) != "left":
            tokenizer.padding_side = "left"

        # Build prompts using chat template format
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

        # Mask everything except the assistant's speaker response
        for i, label in enumerate(label_strings):
            label_tokens = tokenizer.encode(label, add_special_tokens=False)
            input_ids = inputs["input_ids"][i]
            label_length = len(label_tokens)

            # Search for the label tokens in the sequence
            found = False
            for j in range(len(input_ids) - label_length + 1):
                if torch.all(input_ids[j:j + label_length] == torch.tensor(label_tokens, device=input_ids.device)):
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


__all__ = ["load_voxceleb_speaker_dataset", "SpeakerIdentificationCollator"]
