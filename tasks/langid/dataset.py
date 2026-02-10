"""Dataset helpers and data collator for language identification tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from datasets import Dataset, DatasetDict

from core import resolve_num_proc
from core.tasks.collator import build_strict_label_mask
from core.tasks.dataset import (
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
DATASET_CACHE_ROOT = PACKAGE_ROOT / "artifacts" / "langid" / "datasets"

DEFAULT_DATASET_NAME = "google/fleurs"
FALLBACK_LABEL_COLUMNS: Tuple[str, ...] = ("lang_id", "language", "label")

# FLEURS language config codes (country_code format)
# Map language codes to config names
FLEURS_LANGUAGE_CONFIGS = {
    "en": "en_us",  # English (US)
    "de": "de_de",  # German
    "fr": "fr_fr",  # French
    "es": "es_419",  # Spanish (Latin America)
    "zh": "cmn_hans_cn",  # Mandarin Chinese (Simplified)
    "ar": "ar_eg",  # Arabic (Egyptian)
    "hi": "hi_in",  # Hindi (India)
    "ja": "ja_jp",  # Japanese
    "ru": "ru_ru",  # Russian
    "pt": "pt_br",  # Portuguese (Brazil)
}

# Language code to full name mapping
LANGUAGE_CODE_TO_NAME = {
    "en": "english",
    "de": "german",
    "fr": "french",
    "es": "spanish",
    "zh": "chinese",
    "ar": "arabic",
    "hi": "hindi",
    "ja": "japanese",
    "ru": "russian",
    "pt": "portuguese",
}

# Sorted language names for consistent ClassLabel encoding
LANGUAGE_NAMES = ["arabic", "chinese", "english", "french", "german", "hindi", "japanese", "portuguese", "russian", "spanish"]

MANIFEST_FIELDS: Tuple[str, ...] = (
    "label",
    "lang",
)


def _load_fleurs_languages(
    languages: Sequence[str],
    train_split: str = "train",
    validation_split: str = "validation",
    test_split: Optional[str] = "test",
) -> DatasetDict:
    """Load FLEURS dataset for selected languages only.

    Args:
        languages: List of language codes (e.g., ["en", "de", "fr", "es", "zh"])
        train_split: Training split name
        validation_split: Validation split name
        test_split: Test split name (optional)

    Returns:
        DatasetDict with train/validation/test splits
    """
    from datasets import load_dataset, DatasetDict, concatenate_datasets

    print(f"   Loading FLEURS dataset for {len(languages)} languages...")
    print(f"   Strategy: Load individual language folders from parquet branch")

    # Map our language codes to FLEURS config folder names
    fleurs_lang_map = FLEURS_LANGUAGE_CONFIGS

    selected_fleurs_langs = [fleurs_lang_map.get(lang, lang) for lang in languages]
    print(f"   Loading FLEURS language folders: {', '.join(selected_fleurs_langs)}")

    # Collect datasets for each split across all languages
    split_datasets = {train_split: [], validation_split: [], test_split: []}

    # Load each language folder separately
    for simple_code in languages:
        fleurs_config = fleurs_lang_map.get(simple_code)
        if fleurs_config is None:
            print(f"   Warning: Unknown language code '{simple_code}', skipping")
            continue

        print(f"   Loading language: {fleurs_config} ({simple_code})...")

        # Load directly from the language-specific parquet folder
        lang_dataset = load_dataset(
            "google/fleurs",
            data_files={
                "train": f"{fleurs_config}/train/*.parquet",
                "validation": f"{fleurs_config}/validation/*.parquet",
                "test": f"{fleurs_config}/test/*.parquet",
            },
            revision="refs/convert/parquet",
        )

        # Process each split
        for split_name in [train_split, validation_split, test_split]:
            if split_name is None or split_name not in lang_dataset:
                continue

            split_ds = lang_dataset[split_name]
            print(f"      {split_name}: {len(split_ds)} samples")

            # Add our simplified language labels
            def add_labels(example):
                return {
                    **example,
                    "lang": simple_code,
                    "label": LANGUAGE_CODE_TO_NAME.get(simple_code, simple_code),
                }

            labeled_ds = split_ds.map(add_labels, desc=f"Adding labels to {fleurs_config}/{split_name}")
            split_datasets[split_name].append(labeled_ds)

    # Concatenate all language datasets for each split
    all_splits = {}
    for split_name in [train_split, validation_split, test_split]:
        if split_name is None or not split_datasets[split_name]:
            continue

        print(f"   Merging {split_name} split from {len(split_datasets[split_name])} languages...")
        merged_split = concatenate_datasets(split_datasets[split_name])
        all_splits[split_name] = merged_split
        print(f"      ✓ {split_name}: {len(merged_split)} total samples")

    return DatasetDict(all_splits)


def load_fleurs_langid_dataset(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_config: Optional[str] = None,
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
    test_split: Optional[str] = None,
    stratify_by_column: Optional[str] = None,
    data_dir: Optional[str] = None,
    languages: Optional[Sequence[str]] = None,
) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset], List[str]]:
    """
    Load the VoxLingua107 dataset for language identification.

    Args:
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration name
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
        label_column: Label column name
        text_column: Text column name
        audio_column: Audio column name
        split_percentages: Train/val/test split percentages
        train_split: Training split name
        validation_split: Validation split name
        test_split: Test split name
        stratify_by_column: Column to stratify by
        data_dir: Data directory
        languages: List of language codes to include (e.g., ["en", "de", "fr", "es", "zh"])

    Returns:
        Tuple of (train_dataset, validation_dataset, test_dataset, label_names)
    """
    # Default to 5 languages if not specified
    if languages is None:
        languages = ["en", "de", "fr", "es", "zh"]

    # Normalize sample counts
    max_train_samples = _normalize_target_count("max_train_samples", max_train_samples)
    max_validation_samples = _normalize_target_count("max_validation_samples", max_validation_samples)
    max_test_samples = _normalize_target_count("max_test_samples", max_test_samples)

    print(f"📥 Loading FLEURS dataset for {len(languages)} languages: {', '.join(languages)}")
    print(f"   Strategy: Load only selected language configs (efficient & fast)")

    # Load FLEURS dataset for only the selected languages
    dataset = _load_fleurs_languages(
        languages=languages,
        train_split=train_split,
        validation_split=validation_split,
        test_split=test_split,
    )

    # FLEURS already has audio in the right format
    audio_column_name = "audio"

    # Apply sample limits early to avoid processing too much data
    if max_train_samples and train_split in dataset:
        original_size = len(dataset[train_split])
        dataset[train_split] = dataset[train_split].select(range(min(max_train_samples, original_size)))
        print(f"   Limited train split: {original_size} -> {len(dataset[train_split])} samples")

    if max_validation_samples and validation_split in dataset:
        original_size = len(dataset[validation_split])
        dataset[validation_split] = dataset[validation_split].select(range(min(max_validation_samples, original_size)))
        print(f"   Limited validation split: {original_size} -> {len(dataset[validation_split])} samples")

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
    languages_key = "_".join(sorted(languages))
    cache_name = (
        f"{dataset_key}_{config_key}_{languages_key}"
        f"_train_{_samples_key(max_train_samples)}"
        f"_val_{_samples_key(max_validation_samples)}"
        f"_test_{_samples_key(max_test_samples)}"
        f"_seed_{int(seed)}"
        f"_filtered_{filtered_sizes.get('train', 0)}_{filtered_sizes.get('validation', 0)}_{filtered_sizes.get('test', 0)}.json"
    )
    cache_path = cache_root / cache_name

    # Extract label names - use LANGUAGE_NAMES for consistent ordering
    label_names = LANGUAGE_NAMES

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
            "languages": list(languages),
        },
    )

    # Print summary
    print_dataset_summary(
        task_emoji="🌐",
        task_name="Language ID dataset",
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
class LanguageIdentificationCollator:
    """Prepare batches for language identification fine-tuning.

    Always uses chat template with both user message (audio + instruction) and assistant response (ground truth).
    During evaluation, the CustomTrainer's prediction_step will strip out the ground truth before generation.
    """

    processor: Any
    sampling_rate: int
    label_names: Sequence[str]
    warn_on_label_mask_fallback: bool = True

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
        class_options = ", ".join(self.label_names)
        instruction = f"What language is being spoken in the audio? Choose from: {class_options}. Output only the label."
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

        # Mask everything except the assistant's language response
        for i, label in enumerate(label_strings):
            label_tokens = tokenizer.encode(label, add_special_tokens=False)
            input_ids = inputs["input_ids"][i]
            mask_stats = build_strict_label_mask(
                input_ids=input_ids,
                labels=labels[i],
                label_tokens=label_tokens,
                ignore_index=-100,
            )
            if self.warn_on_label_mask_fallback and bool(mask_stats.get("used_fallback", False)):
                reason = str(mask_stats.get("fallback_reason"))
                print(
                    "[langid-collator] label mask fallback used "
                    f"(reason={reason}, label='{label}', kept={mask_stats.get('kept_token_count', 0)})"
                )

        inputs["labels"] = labels
        return inputs


__all__ = ["load_fleurs_langid_dataset", "LanguageIdentificationCollator"]
