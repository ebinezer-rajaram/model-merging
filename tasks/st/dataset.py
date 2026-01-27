"""Dataset loading and collation for Speech Translation tasks."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, DatasetDict

# Suppress Qwen audio output warning
logging.getLogger().addFilter(lambda record: "System prompt modified" not in record.getMessage())
warnings.filterwarnings("ignore", message=".*System prompt modified.*")
warnings.filterwarnings("ignore", message=".*audio output may not work.*")

from core import resolve_num_proc
from core.tasks.dataset import (
    _column_exists,
    _ensure_audio_column,
    _normalize_target_count,
    _rename_column,
    _samples_key,
    _select_candidate_column,
    add_duration_to_dataset,
    assemble_splits,
    cache_and_sample_splits,
    filter_by_duration,
    load_dataset,
    print_dataset_summary,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATASET_CACHE_ROOT = PACKAGE_ROOT / "artifacts" / "st" / "datasets"

DEFAULT_DATASET_NAME = "fixie-ai/covost2"
DEFAULT_DATASET_CONFIG = "en_de"
FALLBACK_SOURCE_COLUMNS: Tuple[str, ...] = ("sentence", "text", "source", "source_text")
FALLBACK_TRANSLATION_COLUMNS: Tuple[str, ...] = ("translation", "target", "target_text")

MANIFEST_FIELDS: Tuple[str, ...] = ("id", "text", "translation", "path")

# Language code to full language name mapping for prompts
LANGUAGE_NAMES = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ar": "Arabic",
    "tr": "Turkish",
    "ca": "Catalan",
    "cy": "Welsh",
    "et": "Estonian",
    "fa": "Persian",
    "id": "Indonesian",
    "lv": "Latvian",
    "mn": "Mongolian",
    "sl": "Slovenian",
    "sv": "Swedish",
    "ta": "Tamil",
}


def load_covost2_dataset(
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
    audio_column: Optional[str] = None,
    source_column: Optional[str] = None,
    translation_column: Optional[str] = None,
    train_split: str = "train",
    validation_split: Optional[str] = "validation",
    test_split: Optional[str] = "test",
) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
    """
    Load CoVoST2 speech translation dataset with optional sub-sampling.

    Filters out examples containing "REMOVE" in source or translation text.

    Returns train, validation, test splits (None when missing).
    """
    # Normalize sample counts
    max_train_samples = _normalize_target_count("max_train_samples", max_train_samples)
    max_validation_samples = _normalize_target_count("max_validation_samples", max_validation_samples)
    max_test_samples = _normalize_target_count("max_test_samples", max_test_samples)

    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config)

    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected DatasetDict, received {type(dataset)!r}")

    # Normalize column names (source text and translation)
    if source_column and source_column != "text":
        dataset = _rename_column(dataset, source_column, "text")
    elif not _column_exists(dataset, "text"):
        fallback_source = _select_candidate_column(dataset, FALLBACK_SOURCE_COLUMNS)
        if fallback_source and fallback_source != "text":
            dataset = _rename_column(dataset, fallback_source, "text")
        else:
            raise ValueError("Dataset must contain a source text column. Provide 'source_column'.")

    if translation_column and translation_column != "translation":
        dataset = _rename_column(dataset, translation_column, "translation")
    elif not _column_exists(dataset, "translation"):
        fallback_translation = _select_candidate_column(dataset, FALLBACK_TRANSLATION_COLUMNS)
        if fallback_translation and fallback_translation != "translation":
            dataset = _rename_column(dataset, fallback_translation, "translation")
        else:
            raise ValueError("Dataset must contain a translation column. Provide 'translation_column'.")

    # Assemble splits
    dataset = assemble_splits(
        dataset,
        train_split=train_split,
        validation_split=validation_split,
        test_split=test_split,
    )

    # Ensure audio column
    dataset, audio_column_name = _ensure_audio_column(dataset, preferred=audio_column)

    # Filter out examples with empty audio bytes by checking the raw Arrow data
    # This must be done FIRST, before any operation that might try to decode audio
    print("🔍 Checking for empty audio bytes...")
    audio_bytes_filtered = {}
    for split_name in list(dataset.keys()):
        before = len(dataset[split_name])
        ds = dataset[split_name]

        # Manually build list of valid indices by checking raw bytes or paths
        valid_indices = []
        for idx in range(len(ds)):
            try:
                # Access the raw row data from Arrow table without decoding
                row = ds.data.slice(idx, 1).to_pydict()
                audio_data = row[audio_column_name][0]

                # Check if bytes field exists and is non-empty
                if isinstance(audio_data, dict):
                    audio_bytes = audio_data.get("bytes")
                    audio_path = audio_data.get("path")
                    audio_array = audio_data.get("array")
                    if (audio_bytes and len(audio_bytes) > 0) or audio_path or audio_array is not None:
                        valid_indices.append(idx)
                elif audio_data is not None:
                    # If not a dict, assume it's valid
                    valid_indices.append(idx)
            except Exception:
                # Skip problematic rows
                continue

        # Select only valid indices
        if len(valid_indices) < before:
            dataset[split_name] = ds.select(valid_indices)
            audio_bytes_filtered[split_name] = before - len(valid_indices)

    if audio_bytes_filtered:
        print(f"🚫 Filtered examples with empty audio bytes: {audio_bytes_filtered}")

    # Filter out examples containing "REMOVE" (text-only filter)
    # Now safe to filter since empty audio bytes have been removed
    def _filter_remove(source_text: Any, translation_text: Any) -> bool:
        """Filter out examples containing 'REMOVE' in source or translation."""
        source_str = str(source_text or "")
        translation_str = str(translation_text or "")
        return "REMOVE" not in source_str and "REMOVE" not in translation_str

    filtered_counts = {}
    for split_name in list(dataset.keys()):
        before = len(dataset[split_name])
        # Filter on text columns only - doesn't access audio
        dataset[split_name] = dataset[split_name].filter(
            _filter_remove,
            input_columns=["text", "translation"],
            desc=f"Filtering {split_name} for REMOVE keyword",
        )
        after = len(dataset[split_name])
        if after != before:
            filtered_counts[split_name] = before - after

    if filtered_counts:
        print(f"🚫 Filtered examples containing 'REMOVE': {filtered_counts}")

    # Build cache root early so duration caching matches other tasks
    cache_root = Path(cache_dir) if cache_dir is not None else DATASET_CACHE_ROOT

    # Add duration information (with caching)
    effective_num_proc = resolve_num_proc(num_proc)
    dataset = add_duration_to_dataset(
        dataset,
        audio_column=audio_column_name,
        num_proc=effective_num_proc,
        cache_dir=cache_root if not force_rebuild else None,
    )

    # Filter out corrupted audio files without triggering dataset audio decoding
    # Some rows have empty bytes; decoding them inside datasets.filter crashes before our handler runs.
    print("🔍 Checking for corrupted audio files...")
    corrupted_counts = {}
    for split_name in list(dataset.keys()):
        before = len(dataset[split_name])
        ds = dataset[split_name]
        valid_indices = []

        for idx in range(len(ds)):
            try:
                row = ds.data.slice(idx, 1).to_pydict()
                audio_data = row[audio_column_name][0]

                if isinstance(audio_data, dict):
                    audio_bytes = audio_data.get("bytes")
                    audio_path = audio_data.get("path")
                    audio_array = audio_data.get("array")

                    if (
                        audio_bytes is not None
                        and len(audio_bytes) == 0
                        and not audio_path
                        and audio_array is None
                    ):
                        raise ValueError("empty audio bytes")

                    if audio_path:
                        path = Path(audio_path)
                        # Only validate existence for absolute paths; some datasets store relative paths.
                        if path.is_absolute():
                            if not path.is_file() or path.stat().st_size == 0:
                                raise ValueError("missing/empty audio file")

                    if (
                        (audio_bytes is not None and len(audio_bytes) > 0)
                        or audio_path
                        or audio_array is not None
                    ):
                        valid_indices.append(idx)
                    else:
                        raise ValueError("missing audio content")
                elif audio_data is not None:
                    valid_indices.append(idx)
                else:
                    raise ValueError("missing audio content")
            except Exception as e:
                sample_id = None
                try:
                    sample_id = row.get("id", [None])[0]
                except Exception:
                    sample_id = None
                sample_label = sample_id if sample_id is not None else f"index_{idx}"
                print(
                    f"⚠️  Found corrupted audio at {split_name}[{idx}] (id={sample_label}): {str(e)[:100]}"
                )

        if len(valid_indices) < before:
            dataset[split_name] = ds.select(valid_indices)
            corrupted_counts[split_name] = before - len(valid_indices)

    if corrupted_counts:
        print(f"🚫 Filtered corrupted audio files: {corrupted_counts}")

    # Filter by duration if specified (with caching)
    dataset = filter_by_duration(
        dataset,
        max_duration=max_duration,
        min_duration=min_duration,
        cache_dir=cache_root if not force_rebuild else None,
    )

    # Build cache path
    dataset_key = dataset_name.replace("/", "_")
    config_key = dataset_config or "default"
    cache_name = (
        f"{dataset_key}_{config_key}"
        f"_train_{_samples_key(max_train_samples)}"
        f"_val_{_samples_key(max_validation_samples)}"
        f"_test_{_samples_key(max_test_samples)}"
        f"_seed_{int(seed)}.json"
    )
    cache_path = cache_root / cache_name

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
        },
    )

    # Print summary
    print_dataset_summary(
        task_emoji="🌍",
        task_name="Speech Translation dataset",
        train_subset=train_subset,
        validation_subset=validation_subset,
        test_subset=test_subset,
        splits_metadata=splits_metadata,
        seed=payload_seed,
        num_proc=effective_num_proc,
    )

    return train_subset, validation_subset, test_subset


@dataclass
class STCollator:
    """Data collator for Speech Translation with language-specific prompts.

    Always uses chat template with both user message (audio + instruction) and assistant response (ground truth).
    During evaluation, the CustomTrainer's prediction_step will strip out the ground truth before generation.
    """

    processor: Any
    sampling_rate: int
    instruction: Optional[str] = None
    language_pair: str = "en_de"  # Format: {source_lang}_{target_lang}

    def __post_init__(self):
        """Generate instruction from language pair if not provided."""
        if self.instruction is None:
            source_lang, target_lang = self.language_pair.split("_")
            source_name = LANGUAGE_NAMES.get(source_lang, source_lang.upper())
            target_name = LANGUAGE_NAMES.get(target_lang, target_lang.upper())
            # Use simpler, more direct instruction that focuses on the audio
            # Avoid mentioning source/target languages explicitly as it confuses the model
            self.instruction = f"Translate this audio to {target_name}."

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Prepare batch tensors for the trainer."""
        # Filter out corrupted audio files
        valid_features = []
        for i, feature in enumerate(features):
            try:
                # Try to access the audio array - this will fail for corrupted files
                _ = feature["audio"]["array"]
                valid_features.append(feature)
            except (RuntimeError, ValueError, KeyError) as e:
                # Log the error and skip this sample
                sample_id = feature.get("id", f"index_{i}")
                print(f"⚠️  Skipping corrupted audio sample {sample_id}: {str(e)}")
                continue

        # If all samples in the batch are corrupted, raise an error
        if not valid_features:
            raise RuntimeError("All samples in batch have corrupted audio files")

        # Extract valid audio arrays and translations
        audio_arrays = [feature["audio"]["array"] for feature in valid_features]
        translations = [feature["translation"] for feature in valid_features]

        # Build prompts using chat template format (matching ASR pattern)
        # Always include both user message and assistant response with ground truth
        # During evaluation, CustomTrainer's prediction_step will truncate before generation
        prompts = []
        for translation in translations:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": None},
                        {"type": "text", "text": self.instruction}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": translation}
                    ]
                }
            ]
            prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=False,
                tokenize=False
            )
            prompts.append(prompt)

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "padding_side", None) != "left":
            tokenizer.padding_side = "left"

        inputs = self.processor(
            audio=audio_arrays,
            sampling_rate=self.sampling_rate,
            text=prompts,
            return_tensors="pt",
            padding=True,
        )

        # Create labels for training
        labels = inputs["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        audio_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.audio_token
        )

        # Mask padding and audio tokens
        labels = labels.masked_fill(labels == pad_id, -100)
        labels = labels.masked_fill(labels == audio_token_id, -100)

        # Mask everything except the assistant's translation response
        # With chat template, the format is: <user_message><audio><instruction><assistant_header>TRANSLATION
        # We only want to train on predicting TRANSLATION, not the prompts/headers
        # For each sample, find where the assistant's actual translation starts
        for i, translation in enumerate(translations):
            # Tokenize the ground truth translation to identify it in the full sequence
            translation_tokens = tokenizer.encode(translation, add_special_tokens=False)

            # Find where the translation appears in the input_ids
            input_ids = inputs["input_ids"][i]
            translation_length = len(translation_tokens)

            # Search for the translation tokens in the sequence (search from the end backwards)
            found = False
            # Start from the end and work backwards to find the last occurrence
            for j in range(len(input_ids) - translation_length, -1, -1):
                # Skip padding tokens
                if input_ids[j] == pad_id:
                    continue
                # Check if translation matches at this position
                translation_tensor = torch.tensor(translation_tokens, device=input_ids.device, dtype=input_ids.dtype)
                if torch.equal(input_ids[j:j + translation_length], translation_tensor):
                    # Mask everything before the translation
                    labels[i, :j] = -100
                    found = True
                    break

            # If we didn't find an exact match, try a more flexible approach
            if not found:
                # Try to find the assistant marker and mask before it
                # The chat template typically uses specific tokens to mark roles
                assistant_markers = ["assistant", "<|im_start|>assistant"]

                # Decode the full sequence to find markers
                full_text = tokenizer.decode(input_ids, skip_special_tokens=False)

                # Find the position of "assistant" marker
                assistant_pos = -1
                for marker in assistant_markers:
                    if marker in full_text:
                        # Find where this marker appears in the tokenized sequence
                        marker_tokens = tokenizer.encode(marker, add_special_tokens=False)
                        for j in range(len(input_ids) - len(marker_tokens) + 1):
                            if torch.equal(input_ids[j:j + len(marker_tokens)],
                                         torch.tensor(marker_tokens, device=input_ids.device, dtype=input_ids.dtype)):
                                assistant_pos = j + len(marker_tokens)
                                break
                        if assistant_pos >= 0:
                            break

                if assistant_pos >= 0:
                    # Mask everything before the assistant's response
                    labels[i, :assistant_pos] = -100
                    found = True

            # Last resort fallback: keep only the last N tokens where N is translation length
            if not found:
                # Find non-masked, non-padding tokens
                non_masked = (labels[i] != -100).nonzero(as_tuple=False).squeeze(-1)
                if len(non_masked) > 0:
                    # Keep approximately translation_length tokens at the end
                    # Add some buffer (1.5x) in case tokenization differs slightly
                    keep_length = int(translation_length * 1.5)
                    if len(non_masked) > keep_length:
                        mask_until = non_masked[-keep_length].item()
                        labels[i, :mask_until] = -100

        inputs["labels"] = labels
        return inputs


__all__ = ["load_covost2_dataset", "STCollator", "LANGUAGE_NAMES"]
