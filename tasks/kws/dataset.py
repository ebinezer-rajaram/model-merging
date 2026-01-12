"""Dataset helpers and data collator for keyword spotting tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from datasets import Dataset, DatasetDict, load_dataset

from core import resolve_num_proc
from core.tasks.dataset import (
    _extract_label_names,
    _normalize_target_count,
    _samples_key,
    add_duration_to_dataset,
    cache_and_sample_splits,
    filter_by_duration,
    print_dataset_summary,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATASET_CACHE_ROOT = PACKAGE_ROOT / "artifacts" / "kws" / "datasets"

DEFAULT_DATASET_NAME = "google/speech_commands"
DEFAULT_DATASET_CONFIG = "v0.02"
FALLBACK_LABEL_COLUMNS: Tuple[str, ...] = ("label", "word")

# Known commands in v0.02 (35 total - excludes silence and unknown)
KNOWN_COMMANDS = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five",
    "follow", "forward", "four", "go", "happy", "house", "learn", "left",
    "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila",
    "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes", "zero"
]

MANIFEST_FIELDS: Tuple[str, ...] = (
    "label",
    "speaker_id",
    "utterance_id",
    "file",
)


def load_speech_commands_kws_dataset(
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
    data_dir: Optional[str] = None,
    revision: Optional[str] = None,
) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset], List[str]]:
    """
    Load the Google Speech Commands dataset for keyword spotting.

    Filters to include only known commands (excludes silence and unknown classes).

    Args:
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration (v0.01 or v0.02)
        max_train_samples: Maximum training samples
        max_validation_samples: Maximum validation samples
        max_test_samples: Maximum test samples
        max_duration: Maximum audio duration in seconds
        min_duration: Minimum audio duration in seconds
        seed: Random seed
        num_proc: Number of processes for parallel processing
        cache_dir: Cache directory
        cache_splits: Whether to cache splits
        force_rebuild: Force rebuild cache
        label_column: Label column name (not used, kept for API compatibility)
        text_column: Text column name (not used, kept for API compatibility)
        audio_column: Audio column name
        split_percentages: Train/val/test split percentages (not used)
        train_split: Training split name
        validation_split: Validation split name
        test_split: Test split name
        stratify_by_column: Column to stratify by (not used)
        data_dir: Data directory (not used)
        revision: Dataset revision (default: refs/convert/parquet)

    Returns:
        Tuple of (train_dataset, validation_dataset, test_dataset, label_names)
    """
    # Normalize sample counts
    max_train_samples = _normalize_target_count("max_train_samples", max_train_samples)
    max_validation_samples = _normalize_target_count("max_validation_samples", max_validation_samples)
    max_test_samples = _normalize_target_count("max_test_samples", max_test_samples)

    # Use parquet branch if no revision specified
    if revision is None:
        revision = "refs/convert/parquet"

    print(f"📥 Loading Google Speech Commands dataset ({dataset_config})")
    print(f"   Strategy: Load from parquet branch, filter to known commands only")
    print(f"   Revision: {revision}")

    # Load dataset from parquet branch using data_files to specify folder structure
    # The parquet branch has structure: v0.02/train/*.parquet, v0.02/validation/*.parquet, etc.
    dataset = load_dataset(
        dataset_name,
        data_files={
            "train": f"{dataset_config}/train/*.parquet",
            "validation": f"{dataset_config}/validation/*.parquet",
            "test": f"{dataset_config}/test/*.parquet",
        },
        revision=revision,
    )

    print(f"   Loaded splits: {list(dataset.keys())}")
    for split_name, split_ds in dataset.items():
        print(f"      {split_name}: {len(split_ds)} samples")

    # Audio column is standard "audio"
    audio_column_name = audio_column or "audio"

    # Filter to exclude only silence (keep all 35 command words)
    def _filter_silence(example):
        """Exclude only the silence class, keep all 35 command words."""
        label = example.get("label")

        # The silence class is label 35 in v0.02
        # We want to keep labels 0-34 (all 35 command words, ignoring is_unknown field)
        if label == 35:
            return False

        return True

    print(f"Filtering to exclude silence class only (keeping all 35 command words)...")
    initial_sizes = {split_name: len(split_ds) for split_name, split_ds in dataset.items()}
    dataset = dataset.filter(_filter_silence)
    filtered_sizes = {split_name: len(split_ds) for split_name, split_ds in dataset.items()}

    for split_name in initial_sizes:
        removed = initial_sizes[split_name] - filtered_sizes.get(split_name, 0)
        kept = filtered_sizes.get(split_name, 0)
        print(f"  {split_name}: kept {kept} samples, removed {removed} (silence only)")

    # Apply sample limits early to avoid processing too much data
    if max_train_samples and train_split in dataset:
        original_size = len(dataset[train_split])
        dataset[train_split] = dataset[train_split].select(range(min(max_train_samples, original_size)))
        print(f"   Limited train split: {original_size} -> {len(dataset[train_split])} samples")

    if max_validation_samples and validation_split in dataset:
        original_size = len(dataset[validation_split])
        dataset[validation_split] = dataset[validation_split].select(range(min(max_validation_samples, original_size)))
        print(f"   Limited validation split: {original_size} -> {len(dataset[validation_split])} samples")

    if max_test_samples and test_split in dataset:
        original_size = len(dataset[test_split])
        dataset[test_split] = dataset[test_split].select(range(min(max_test_samples, original_size)))
        print(f"   Limited test split: {original_size} -> {len(dataset[test_split])} samples")

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

    # Filter corrupted audio files
    def _validate_audio(example):
        """Check if audio can be loaded without errors."""
        try:
            _ = example[audio_column_name]["array"]
            return True
        except (RuntimeError, Exception):
            return False

    print(f"Filtering corrupted audio files from dataset...")
    initial_sizes = {split_name: len(split_ds) for split_name, split_ds in dataset.items()}
    dataset = dataset.filter(_validate_audio)
    filtered_sizes = {split_name: len(split_ds) for split_name, split_ds in dataset.items()}

    for split_name in initial_sizes:
        removed = initial_sizes[split_name] - filtered_sizes.get(split_name, 0)
        if removed > 0:
            print(f"  {split_name}: removed {removed} corrupted audio file(s)")

    # Build cache path
    cache_root = Path(cache_dir) if cache_dir is not None else DATASET_CACHE_ROOT
    dataset_key = dataset_name.replace("/", "_")
    config_key = dataset_config or "default"
    cache_name = (
        f"{dataset_key}_{config_key}"
        f"_train_{_samples_key(max_train_samples)}"
        f"_val_{_samples_key(max_validation_samples)}"
        f"_test_{_samples_key(max_test_samples)}"
        f"_seed_{int(seed)}"
        f"_filtered_{filtered_sizes.get('train', 0)}_{filtered_sizes.get('validation', 0)}_{filtered_sizes.get('test', 0)}.json"
    )
    cache_path = cache_root / cache_name

    # Extract label names from the dataset
    # Speech Commands has ClassLabel feature with names
    label_probe = dataset.get(train_split) or next(iter(dataset.values()))
    label_names = _extract_label_names(label_probe)

    # If extraction failed, use known commands list
    if not label_names:
        print("   Warning: Could not extract label names from dataset features")
        print("   Using predefined list of 35 known commands")
        label_names = KNOWN_COMMANDS

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
            "revision": revision,
        },
    )

    # Print summary
    print_dataset_summary(
        task_emoji="🎤",
        task_name="Keyword Spotting dataset",
        train_subset=train_subset,
        validation_subset=validation_subset,
        test_subset=test_subset,
        splits_metadata=splits_metadata,
        label_names=label_names,
        seed=payload_seed,
        num_proc=effective_num_proc,
    )

    return train_subset, validation_subset, test_subset, label_names


@dataclass
class KeywordSpottingCollator:
    """Prepare batches for keyword spotting fine-tuning.

    Always uses chat template with both user message (audio + instruction) and assistant response (ground truth).
    During evaluation, the CustomTrainer's prediction_step will strip out the ground truth before generation.
    """

    processor: Any
    sampling_rate: int
    label_names: Sequence[str]

    def _label_to_text(self, value: Any) -> str:
        """Convert label value to text."""
        if value is None:
            return ""
        try:
            index = int(value)
        except (TypeError, ValueError):
            return str(value)
        if 0 <= index < len(self.label_names):
            return str(self.label_names[index])
        return str(index)

    def _build_instruction(self) -> str:
        """Build the instruction text for the user message."""
        # Format class options for the prompt
        # For 35 classes, we'll keep them in the prompt but make it clear
        class_options = ", ".join(self.label_names)
        instruction = (
            f"What word is spoken in the audio? "
            f"Choose from: {class_options}. "
            f"Output only the word."
        )
        return instruction

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Prepare batch tensors for the trainer."""
        # Filter out corrupted audio samples
        valid_features = []
        for feature in features:
            try:
                _ = feature["audio"]["array"]
                valid_features.append(feature)
            except (RuntimeError, Exception) as e:
                print(f"Warning: Skipping corrupted audio sample: {e}")
                continue

        if not valid_features:
            raise RuntimeError("All audio samples in batch are corrupted")

        audio_arrays = [feature["audio"]["array"] for feature in valid_features]
        label_strings = [self._label_to_text(feature.get("label")) for feature in valid_features]

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "padding_side", None) != "left":
            tokenizer.padding_side = "left"

        # Build prompts using chat template format
        instruction = self._build_instruction()
        prompts = []
        for label in label_strings:
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

        # Mask everything except the assistant's keyword response
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


__all__ = ["load_speech_commands_kws_dataset", "KeywordSpottingCollator"]
