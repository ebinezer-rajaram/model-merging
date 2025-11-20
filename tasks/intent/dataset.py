"""Dataset helpers and data collator for intent classification tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from datasets import Dataset

from core import resolve_num_proc
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
DATASET_CACHE_ROOT = PACKAGE_ROOT / "artifacts" / "intent" / "datasets"

DEFAULT_DATASET_NAME = "slurp"
FALLBACK_LABEL_COLUMNS: Tuple[str, ...] = ("intent", "scenario", "action")

MANIFEST_FIELDS: Tuple[str, ...] = (
    "label",
    "intent",
    "scenario",
    "action",
    "sentence",
    "utt_id",
    "path",
)


def load_slurp_intent_dataset(
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
    validation_split: Optional[str] = "devel",
    test_split: Optional[str] = "test",
    stratify_by_column: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset], List[str]]:
    """
    Load the SLURP dataset (or a compatible intent classification dataset).

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
        task_emoji="📝",
        task_name="Intent dataset",
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
class IntentClassificationCollator:
    """Prepare batches for intent classification fine-tuning.

    Always uses chat template with both user message (audio + instruction) and assistant response (ground truth).
    During evaluation, the CustomTrainer's prediction_step will strip out the ground truth before generation.
    """

    processor: Any
    sampling_rate: int
    label_names: Sequence[str]
    include_transcript: bool = True
    prepend_scenario: bool = False

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

    def _build_instruction(self, transcript: str, metadata: Dict[str, Any]) -> str:
        """Build the instruction text for the user message."""
        transcript = (transcript or "").strip()
        # Format class options for the prompt
        class_options = ", ".join(self.label_names)
        instruction = f"What is the user's intent from the spoken utterance? Choose from: {class_options}. Output only the label."

        if self.prepend_scenario:
            scenario = metadata.get("scenario")
            action = metadata.get("action")
            scenario_parts = []
            if scenario:
                scenario_parts.append(f"Scenario: {scenario}")
            if action:
                scenario_parts.append(f"Action: {action}")
            if scenario_parts:
                instruction += "\n" + "\n".join(scenario_parts)
        if transcript and self.include_transcript:
            instruction += f"\nTranscript: {transcript}"

        return instruction

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
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
        transcripts = [feature.get("text", "") for feature in valid_features]
        label_strings = [self._label_to_text(feature.get("label")) for feature in valid_features]

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "padding_side", None) != "left":
            tokenizer.padding_side = "left"

        # Build prompts using chat template format
        prompts = []
        for text, feature, label in zip(transcripts, valid_features, label_strings):
            instruction = self._build_instruction(text, feature)

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

        # Mask everything except the assistant's intent response
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


__all__ = ["load_slurp_intent_dataset", "IntentClassificationCollator"]
