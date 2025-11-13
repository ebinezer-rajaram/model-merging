"""Dataset loading and collation for ASR tasks."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset

# Suppress Qwen audio output warning (we're only doing ASR, not audio generation)
# This is a root logger warning, so we need to filter it via logging
logging.getLogger().addFilter(lambda record: "System prompt modified" not in record.getMessage())
warnings.filterwarnings("ignore", message=".*System prompt modified.*")
warnings.filterwarnings("ignore", message=".*audio output may not work.*")

from core import (
    add_duration,
    build_split_metadata,
    hours_key,
    load_cached_split,
    normalize_split_metadata,
    num_proc_map_kwargs,
    resolve_num_proc,
    save_cached_split,
    select_indices_by_duration,
    subset_dataset_by_metadata,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATASET_CACHE_ROOT = PACKAGE_ROOT / "artifacts" / "asr" / "datasets"

MANIFEST_FIELDS: Tuple[str, ...] = ("id", "speaker_id", "chapter_id", "text")


def _normalize_cached_payload(
    payload: Dict[str, Any],
    fallback_seed: int,
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    if "splits" in payload:
        return normalize_split_metadata(payload.get("splits")), int(payload.get("seed", fallback_seed))

    # Legacy payload support (prior format without nested splits).
    splits = {
        "train": {
            "indices": list(payload.get("train_indices", [])),
            "hours": float(payload.get("train_duration_hours", 0.0)),
            "manifest": payload.get("train_manifest", []),
        },
        "validation": {
            "indices": list(payload.get("val_indices", [])),
            "hours": float(payload.get("val_duration_hours", 0.0)),
            "manifest": payload.get("val_manifest", []),
        },
    }
    return splits, int(payload.get("seed", fallback_seed))


def load_librispeech_subset(
    train_hours: float = 10.0,
    val_hours: Optional[float] = 1.0,
    seed: int = 0,
    *,
    num_proc: Optional[int | str] = None,
    cache_dir: Optional[Path | str] = None,
    cache_splits: bool = True,
    force_rebuild: bool = False,
    return_full_validation: bool = False,
    return_test_split: bool = False,
    test_split: str = "test.clean",
    test_hours: Optional[float] = None,
) -> Tuple[Dataset, Dataset] | Tuple[Dataset, Dataset, Dataset] | Tuple[Dataset, Dataset, Dataset, Dataset]:
    """Load configurable LibriSpeech subsets with cached manifests."""
    base = load_dataset("librispeech_asr", "clean")
    train_full, val_full = base["train.100"], base["validation"]

    effective_num_proc = resolve_num_proc(num_proc)

    train_full = train_full.map(add_duration, **num_proc_map_kwargs(effective_num_proc))
    val_full = val_full.map(add_duration, **num_proc_map_kwargs(effective_num_proc))
    test_full = None
    if return_test_split:
        available_splits = [name for name in base.keys() if name.startswith("test")]
        split_aliases = {
            "test.clean": "test",
            "test-clean": "test",
            "test_other": "test.other",
            "test-other": "test.other",
        }
        lookup_split = split_aliases.get(test_split, test_split)
        try:
            test_full = base[lookup_split]
        except KeyError as exc:
            raise ValueError(
                f"Requested test split '{test_split}' is not available. Options: {available_splits}."
            ) from exc
        test_full = test_full.map(add_duration, **num_proc_map_kwargs(effective_num_proc))

    cache_root = Path(cache_dir) if cache_dir is not None else DATASET_CACHE_ROOT
    cache_name = (
        f"train_{hours_key(train_hours)}_val_{hours_key(val_hours)}"
        f"_seed_{int(seed)}.json"
    )
    cache_path = cache_root / cache_name

    cached_payload = load_cached_split(cache_path) if cache_splits and not force_rebuild else None

    splits_metadata: Dict[str, Dict[str, Any]]
    payload_seed = seed

    if cached_payload is None:
        train_indices = select_indices_by_duration(train_full["duration"], train_hours, seed)
        val_indices = select_indices_by_duration(val_full["duration"], val_hours, seed + 1)

        splits_metadata = {
            "train": build_split_metadata(
                train_full,
                train_indices,
                manifest_fields=MANIFEST_FIELDS,
            ),
            "validation": build_split_metadata(
                val_full,
                val_indices,
                manifest_fields=MANIFEST_FIELDS,
            ),
        }
        payload_seed = seed

        if cache_splits:
            save_cached_split(
                cache_path,
                {
                    "seed": payload_seed,
                    "dataset": "librispeech_asr",
                    "config": "clean",
                    "splits": splits_metadata,
                },
            )
    else:
        splits_metadata, payload_seed = _normalize_cached_payload(cached_payload, seed)

    train_subset = subset_dataset_by_metadata(train_full, splits_metadata.get("train"))
    val_subset = subset_dataset_by_metadata(val_full, splits_metadata.get("validation"))

    train_hours_logged = float(splits_metadata.get("train", {}).get("hours", 0.0) or 0.0)
    val_hours_logged = float(splits_metadata.get("validation", {}).get("hours", 0.0) or 0.0)
    test_subset: Optional[Dataset] = None
    test_hours_logged: Optional[float] = None

    if return_test_split and test_full is not None:
        if test_hours is not None:
            test_indices = select_indices_by_duration(test_full["duration"], test_hours, seed + 2)
            test_subset = test_full.select(test_indices)
            total_seconds = sum(float(test_full["duration"][idx]) for idx in test_indices)
            test_hours_logged = total_seconds / 3600.0
        else:
            test_subset = test_full
            durations = test_subset["duration"] if "duration" in test_subset.column_names else []
            total_seconds = float(sum(durations)) if durations else 0.0
            test_hours_logged = total_seconds / 3600.0 if total_seconds else 0.0

    print(
        "🎧 LibriSpeech subset:"
        f" train={len(train_subset)} samples ≈ {train_hours_logged:.2f} h,"
        f" val={len(val_subset)} samples ≈ {val_hours_logged:.2f} h"
        f" (seed={payload_seed}, num_proc={effective_num_proc})."
    )

    if return_test_split and test_subset is not None:
        print(
            f"🧪 Test split '{test_split}': {len(test_subset)} samples"
            + (f" ≈ {test_hours_logged:.2f} h." if test_hours_logged is not None else ".")
        )

    if return_full_validation and return_test_split and test_subset is not None:
        return train_subset, val_subset, val_full, test_subset
    if return_full_validation:
        return train_subset, val_subset, val_full
    if return_test_split and test_subset is not None:
        return train_subset, val_subset, test_subset
    return train_subset, val_subset


def load_librispeech_10h() -> Tuple[Dataset, Dataset]:
    """Backwards-compatible loader returning a 10h/1h split."""
    return load_librispeech_subset(train_hours=10.0, val_hours=1.0)


@dataclass
class OmniASRCollator:
    """Data collator for Omni ASR inputs.

    Always uses chat template with both user message (audio + instruction) and assistant response (ground truth).
    During evaluation, the CustomTrainer's prediction_step will strip out the ground truth before generation.
    """

    processor: Any
    sampling_rate: int
    instruction: str = "Only output the transcription of this audio. Do not include any explanations, summaries, or questions."

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Prepare batch tensors for the trainer."""
        audio_arrays = [feature["audio"]["array"] for feature in features]
        texts = [feature["text"] for feature in features]

        # Build prompts using chat template format (matching reference test script)
        # Always include both user message and assistant response with ground truth
        # During evaluation, CustomTrainer's prediction_step will truncate before generation
        prompts = []
        for text in texts:
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
                        {"type": "text", "text": text}
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

        # Mask everything except the assistant's transcription response
        # With chat template, the format is: <user_message><audio><instruction><assistant_header>TRANSCRIPTION
        # We only want to train on predicting TRANSCRIPTION, not the prompts/headers
        # For each sample, find where the assistant's actual transcription starts
        # The chat template adds role markers, so we need to mask everything before the transcription
        for i, text in enumerate(texts):
            # Tokenize the ground truth transcription to identify it in the full sequence
            transcription_tokens = tokenizer.encode(text, add_special_tokens=False)

            # Find where the transcription appears in the input_ids
            input_ids = inputs["input_ids"][i]
            transcription_length = len(transcription_tokens)

            # Search for the transcription tokens in the sequence
            found = False
            for j in range(len(input_ids) - transcription_length + 1):
                # Check if transcription matches at this position
                if torch.all(input_ids[j:j + transcription_length] == torch.tensor(transcription_tokens, device=input_ids.device)):
                    # Mask everything before the transcription
                    labels[i, :j] = -100
                    found = True
                    break

            # If we didn't find an exact match, fall back to masking based on sequence structure
            if not found:
                # Mask everything except the last tokens (likely the transcription)
                # This is a fallback and may not be perfect
                non_masked = (labels[i] != -100).nonzero(as_tuple=False)
                if len(non_masked) > transcription_length:
                    # Mask all but the last transcription_length tokens
                    mask_until = non_masked[-transcription_length].item()
                    labels[i, :mask_until] = -100

        inputs["labels"] = labels
        return inputs
