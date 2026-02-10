"""Dataset helpers and data collator for emotion recognition tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from datasets import Dataset

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
DATASET_CACHE_ROOT = PACKAGE_ROOT / "artifacts" / "emotion" / "datasets"

DEFAULT_DATASET_NAME = "superb"
DEFAULT_DATASET_CONFIG = "er"
FALLBACK_LABEL_COLUMNS: Tuple[str, ...] = ("emotion", "label")

MANIFEST_FIELDS: Tuple[str, ...] = (
    "label",
    "text",
    "speaker",
    "gender",
    "emotion",
    "fileid",
)


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
    data_dir: Optional[str] = None,
) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset], List[str]]:
    """
    Load the SUPERB emotion recognition dataset (or a compatible dataset).

    Returns train, validation, test splits (None when missing) and label names.
    """
    # Normalize sample counts
    max_train_samples = _normalize_target_count("max_train_samples", max_train_samples)
    max_validation_samples = _normalize_target_count("max_validation_samples", max_validation_samples)
    max_test_samples = _normalize_target_count("max_test_samples", max_test_samples)

    # Load and prepare dataset (handles column normalization, splitting, etc.)
    dataset, audio_column_name = load_and_prepare_dataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        label_column=label_column,
        fallback_label_columns=FALLBACK_LABEL_COLUMNS,
        text_column=text_column,
        audio_column=audio_column,
        split_percentages=split_percentages,
        train_split=train_split,
        validation_split=validation_split,
        test_split=test_split,
        stratify_by_column=stratify_by_column,
        seed=seed,
        data_dir=data_dir,
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
    cache_name = (
        f"{dataset_key}_{config_key}"
        f"_train_{_samples_key(max_train_samples)}"
        f"_val_{_samples_key(max_validation_samples)}"
        f"_test_{_samples_key(max_test_samples)}"
        f"_seed_{int(seed)}.json"
    )
    cache_path = cache_root / cache_name

    # Extract label names
    label_probe = dataset.get("train") or next(iter(dataset.values()))
    label_names = _extract_label_names(label_probe)

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
        },
    )

    # Print summary
    print_dataset_summary(
        task_emoji="😊",
        task_name="Emotion dataset",
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
class EmotionRecognitionCollator:
    """Prepare batches for emotion recognition fine-tuning.

    Always uses chat template with both user message (audio + instruction) and assistant response (ground truth).
    During evaluation, the CustomTrainer's prediction_step will strip out the ground truth before generation.
    """

    processor: Any
    sampling_rate: int
    label_names: Sequence[str]
    include_transcript: bool = True
    warn_on_label_mask_fallback: bool = True

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
        # Format class options for the prompt
        class_options = ", ".join(self.label_names)
        instruction = f"What emotion is expressed in the spoken utterance? Choose from: {class_options}. Output only the label."

        if transcript and self.include_transcript:
            instruction += f"\nTranscript: {transcript}"

        return instruction

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_arrays = [feature["audio"]["array"] for feature in features]
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

        # Mask everything except the assistant's emotion response
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
                    "[emotion-collator] label mask fallback used "
                    f"(reason={reason}, label='{label}', kept={mask_stats.get('kept_token_count', 0)})"
                )

        inputs["labels"] = labels
        return inputs


__all__ = ["load_superb_emotion_dataset", "EmotionRecognitionCollator"]
