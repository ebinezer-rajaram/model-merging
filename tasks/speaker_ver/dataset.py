"""Dataset helpers and data collator for speaker verification task."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import Audio, Dataset, DatasetDict

from core import resolve_num_proc
from core.tasks.dataset import (
    _normalize_target_count,
    _samples_key,
    apply_split_percentages,
    cache_and_sample_splits,
    load_and_prepare_dataset,
    print_dataset_summary,
)

# Reuse speaker selection from speaker_id
from tasks.speaker_id.dataset import _select_speakers

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATASET_CACHE_ROOT = PACKAGE_ROOT / "artifacts" / "speaker_ver" / "datasets"

DEFAULT_DATASET_NAME = "acul3/voxceleb2"
FALLBACK_LABEL_COLUMNS: Tuple[str, ...] = ("speaker", "speaker_id", "spk_id")
FALLBACK_TEXT_COLUMNS: Tuple[str, ...] = ("text", "transcription", "transcript")

MANIFEST_FIELDS: Tuple[str, ...] = (
    "label",
    "speaker_id",
    # Note: audio_a and audio_b contain numpy arrays which aren't JSON serializable
    # so we don't include them in the manifest
)


def _extract_audio_array(audio_data: Any) -> np.ndarray:
    """Extract audio array from various audio data formats.

    Args:
        audio_data: Audio data in various formats (dict with array, numpy array, etc.)

    Returns:
        Numpy array with audio samples
    """
    # Handle dict format (standard HuggingFace datasets format)
    if isinstance(audio_data, dict):
        array = audio_data.get("array")
        if array is None:
            return np.array([], dtype=np.float32)
        # Already a numpy array - just convert dtype
        if isinstance(array, np.ndarray):
            return array.astype(np.float32)
        # Otherwise try to convert (handles lists, etc.)
        return np.asarray(array, dtype=np.float32)

    # Handle raw numpy arrays
    elif isinstance(audio_data, np.ndarray):
        return audio_data.astype(np.float32)

    # For any other object, try direct conversion
    else:
        try:
            return np.asarray(audio_data, dtype=np.float32)
        except (TypeError, ValueError) as e:
            print(f"Warning: Could not extract audio array from {type(audio_data)}: {e}")
            return np.array([], dtype=np.float32)



def _generate_ver_pairs(
    dataset: Dataset,
    pairs_per_speaker: int,
    seed: int,
) -> Dataset:
    """Generate balanced positive/negative speaker verification pairs.

    Args:
        dataset: Dataset with 'label' and 'audio' columns
        pairs_per_speaker: Total number of pairs per speaker (split 50/50 positive/negative)
        seed: Random seed for reproducibility

    Returns:
        Dataset with columns: audio_a, audio_b, label (0=different, 1=same)
    """
    rng = random.Random(seed)

    # Group sample indices by speaker using fast array access
    labels = dataset["label"]
    speaker_to_indices = defaultdict(list)
    for idx, speaker_id in enumerate(labels):
        speaker_to_indices[speaker_id].append(idx)

    speakers = list(speaker_to_indices.keys())
    num_pos = pairs_per_speaker // 2
    num_neg = pairs_per_speaker // 2

    # Generate pair indices first (don't access audio data yet)
    pair_indices = []

    # Generate positive pairs (same speaker, different utterances)
    for speaker, indices in speaker_to_indices.items():
        if len(indices) < 2:
            continue  # Need at least 2 samples for positive pair

        for _ in range(num_pos):
            idx_a, idx_b = rng.sample(indices, 2)
            pair_indices.append((idx_a, idx_b, 1, str(speaker)))

    # Generate negative pairs (different speakers)
    for speaker in speakers:
        indices_a = speaker_to_indices[speaker]

        for _ in range(num_neg):
            # Sample a different speaker
            other_speakers = [s for s in speakers if s != speaker]
            if not other_speakers:
                continue

            other_speaker = rng.choice(other_speakers)
            idx_a = rng.choice(indices_a)
            idx_b = rng.choice(speaker_to_indices[other_speaker])
            pair_indices.append((idx_a, idx_b, 0, f"{speaker}_{other_speaker}"))

    # Shuffle the pair indices
    rng.shuffle(pair_indices)

    # Now create pairs with decoded audio arrays in one pass (decode-on-access)
    all_pairs = []
    for idx_a, idx_b, label, speaker_id in pair_indices:
        all_pairs.append({
            "audio_a": {"array": dataset[idx_a]["audio"]["array"]},
            "audio_b": {"array": dataset[idx_b]["audio"]["array"]},
            "label": label,
            "speaker_id": speaker_id,
        })

    num_positive = sum(1 for _, _, label, _ in pair_indices if label == 1)
    num_negative = len(pair_indices) - num_positive

    print(f"Generated {num_positive} positive pairs and {num_negative} negative pairs")
    print(f"Total: {len(all_pairs)} verification pairs")

    # Create dataset from lists - audio_a and audio_b still contain dict/AudioDecoder references
    return Dataset.from_list(all_pairs)


def _build_pairs_cache_dir(cache_root: Path, cache_meta: Dict[str, Any]) -> Path:
    """Build a deterministic cache directory for speaker-verification pairs."""
    payload = json.dumps(cache_meta, sort_keys=True, default=str).encode("utf-8")
    cache_key = hashlib.md5(payload).hexdigest()[:16]
    return cache_root / "pairs" / cache_key


def _load_cached_pairs(cache_dir: Path, split_name: str) -> Optional[Dataset]:
    """Load cached pairs split if available."""
    split_dir = cache_dir / split_name
    if not split_dir.exists():
        return None
    try:
        return Dataset.load_from_disk(str(split_dir))
    except Exception:
        return None


def _save_cached_pairs(cache_dir: Path, split_name: str, dataset: Dataset) -> None:
    """Save pairs split to cache."""
    split_dir = cache_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(split_dir))


def _add_duration_to_pairs(
    dataset: Dataset,
    audio_column_a: str = "audio_a",
    audio_column_b: str = "audio_b",
    sampling_rate: int = 16000,
    num_proc: Optional[int] = None,
) -> Dataset:
    """Add duration columns for both audio_a and audio_b.

    Args:
        dataset: Dataset with audio_a and audio_b columns (numpy arrays)
        audio_column_a: Name of first audio column
        audio_column_b: Name of second audio column
        sampling_rate: Audio sampling rate
        num_proc: Number of processes for parallel processing

    Returns:
        Dataset with duration_a and duration_b columns added
    """
    def add_durations(example):
        # Audio columns are now plain numpy arrays
        array_a = example[audio_column_a]
        array_b = example[audio_column_b]

        duration_a = len(array_a) / sampling_rate if len(array_a) > 0 else 0.0
        duration_b = len(array_b) / sampling_rate if len(array_b) > 0 else 0.0

        return {
            **example,
            "duration_a": duration_a,
            "duration_b": duration_b,
        }

    return dataset.map(add_durations, num_proc=num_proc, desc="Adding durations")


def _filter_pairs_by_duration(
    dataset: Dataset,
    max_duration: Optional[float] = None,
    min_duration: Optional[float] = None,
) -> Dataset:
    """Filter pairs where either audio violates duration constraints.

    Args:
        dataset: Dataset with duration_a and duration_b columns
        max_duration: Maximum duration in seconds
        min_duration: Minimum duration in seconds

    Returns:
        Filtered dataset
    """
    if max_duration is None and min_duration is None:
        return dataset

    def filter_fn(example):
        dur_a = example.get("duration_a", 0.0)
        dur_b = example.get("duration_b", 0.0)

        if min_duration is not None:
            if dur_a < min_duration or dur_b < min_duration:
                return False

        if max_duration is not None:
            if dur_a > max_duration or dur_b > max_duration:
                return False

        return True

    original_size = len(dataset)
    filtered = dataset.filter(filter_fn, desc="Filtering by duration")
    print(f"Filtered pairs: {original_size} → {len(filtered)}")

    return filtered


def load_speaker_ver_dataset(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_config: Optional[str] = None,
    max_train_samples: Optional[int] = None,
    max_validation_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    max_speakers: Optional[int] = None,
    pairs_per_speaker: int = 200,
    max_duration: Optional[float] = None,
    min_duration: Optional[float] = None,
    max_audio_length: Optional[float] = None,  # Not used in loader, used by collator
    audio_gap_seconds: float = 0.5,  # Not used in loader, used by collator
    split_by_speakers: bool = True,  # Split speakers first (zero-shot) or pairs (standard)
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
    Load speaker verification dataset from VoxCeleb2 with pair generation.

    Generates balanced positive (same speaker) and negative (different speaker) pairs.

    Returns train, validation, test splits (None when missing) and label names (["no", "yes"]).
    """
    # Normalize sample counts
    max_train_samples = _normalize_target_count("max_train_samples", max_train_samples)
    max_validation_samples = _normalize_target_count("max_validation_samples", max_validation_samples)
    max_test_samples = _normalize_target_count("max_test_samples", max_test_samples)
    max_speakers = _normalize_target_count("max_speakers", max_speakers)

    # Load and prepare base dataset
    dataset, audio_column_name = load_and_prepare_dataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        label_column=label_column,
        fallback_label_columns=FALLBACK_LABEL_COLUMNS,
        text_column=text_column,
        fallback_text_columns=FALLBACK_TEXT_COLUMNS,
        audio_column=audio_column,
        split_percentages=None,  # Don't split yet - do after pair generation
        train_split=train_split,
        validation_split=validation_split,
        test_split=test_split,
        stratify_by_column=stratify_by_column,
        seed=seed,
        data_dir=data_dir,
    )

    # Check dataset before speaker selection
    print(f"\nDataset splits before speaker selection: {list(dataset.keys())}")
    for split_name, split_ds in dataset.items():
        num_samples = len(split_ds)
        num_speakers = len(set(split_ds["label"])) if "label" in split_ds.column_names else 0
        print(f"  {split_name}: {num_samples} samples, {num_speakers} speakers")

    # Apply speaker selection
    dataset, selected_speakers = _select_speakers(
        dataset,
        max_speakers=max_speakers,
        seed=seed,
        train_split="train" if "train" in dataset else None,
    )

    # Check after speaker selection
    if selected_speakers:
        print(f"\nAfter speaker selection: {len(selected_speakers)} speakers selected")

    # Use only the train split for generating pairs
    # VoxCeleb2 typically only has a train split
    base_dataset = dataset.get("train") or next(iter(dataset.values()))

    # Add duration information to base dataset before filtering
    from core.tasks.dataset import add_duration_to_dataset, filter_by_duration

    effective_num_proc = resolve_num_proc(num_proc)
    cache_root = Path(cache_dir) if cache_dir is not None else DATASET_CACHE_ROOT

    base_dataset_dict = DatasetDict({"train": base_dataset})
    base_dataset_dict = add_duration_to_dataset(
        base_dataset_dict,
        audio_column=audio_column_name,
        num_proc=effective_num_proc,
        cache_dir=cache_root if not force_rebuild else None,
    )

    # Filter by duration
    base_dataset_dict = filter_by_duration(
        base_dataset_dict,
        max_duration=max_duration,
        min_duration=min_duration,
        cache_dir=cache_root if not force_rebuild else None,
    )

    base_dataset = base_dataset_dict["train"]

    # Ensure decode-on-access matches other tasks (audio["array"])
    try:
        base_dataset = base_dataset.cast_column(audio_column_name, Audio())
    except Exception as e:
        print(f"Warning: Could not cast audio column to Audio() for decode-on-access: {e}")

    # Determine split approach
    if split_by_speakers:
        # Zero-shot approach: Split speakers first, then generate pairs per group
        print(f"\nSplitting speakers for zero-shot evaluation...")

        # Get unique speakers from base_dataset
        unique_speakers = sorted(set(base_dataset["label"]))
        num_speakers = len(unique_speakers)
        print(f"Total speakers: {num_speakers}")

        # Determine split sizes
        if split_percentages:
            # Use provided split percentages
            if isinstance(split_percentages, (list, tuple)):
                # Assume [train%, val%, test%]
                train_pct, val_pct, test_pct = split_percentages
            else:
                # Dict format
                train_pct = split_percentages.get("train", 0.8)
                val_pct = split_percentages.get("validation", 0.1)
                test_pct = split_percentages.get("test", 0.1)
        else:
            # Default: 80/10/10
            train_pct, val_pct, test_pct = 0.8, 0.1, 0.1

        # Calculate speaker counts
        num_train_speakers = int(num_speakers * train_pct)
        num_val_speakers = int(num_speakers * val_pct)
        num_test_speakers = num_speakers - num_train_speakers - num_val_speakers

        print(f"Speaker split: {num_train_speakers} train, {num_val_speakers} val, {num_test_speakers} test")

        # Shuffle and split speakers
        rng = random.Random(seed)
        shuffled_speakers = unique_speakers.copy()
        rng.shuffle(shuffled_speakers)

        train_speakers = set(shuffled_speakers[:num_train_speakers])
        val_speakers = set(shuffled_speakers[num_train_speakers:num_train_speakers + num_val_speakers])
        test_speakers = set(shuffled_speakers[num_train_speakers + num_val_speakers:])

        # Split base dataset by speakers using fast Arrow-based filtering
        # This is MUCH faster than .filter() with lambda for large datasets
        labels = base_dataset["label"]

        train_indices = [i for i, label in enumerate(labels) if label in train_speakers]
        val_indices = [i for i, label in enumerate(labels) if label in val_speakers]
        test_indices = [i for i, label in enumerate(labels) if label in test_speakers]

        train_base = base_dataset.select(train_indices)
        val_base = base_dataset.select(val_indices)
        test_base = base_dataset.select(test_indices)

        print(f"Base dataset splits: {len(train_base)} train, {len(val_base)} val, {len(test_base)} test samples")

        # Generate pairs separately for each split (with optional caching)
        print(f"\nGenerating speaker verification pairs ({pairs_per_speaker} per speaker)...")

        cache_meta = {
            "dataset": dataset_name,
            "config": dataset_config,
            "max_speakers": max_speakers,
            "selected_speakers": selected_speakers,
            "pairs_per_speaker": pairs_per_speaker,
            "max_duration": max_duration,
            "min_duration": min_duration,
            "split_by_speakers": split_by_speakers,
            "split_percentages": split_percentages,
            "seed": seed,
            "base_count": len(base_dataset),
        }
        pairs_cache_dir = _build_pairs_cache_dir(cache_root, cache_meta)

        load_pairs_cache = cache_splits and not force_rebuild
        save_pairs_cache = cache_splits
        train_pairs = _load_cached_pairs(pairs_cache_dir, "train") if load_pairs_cache else None
        val_pairs = _load_cached_pairs(pairs_cache_dir, "validation") if load_pairs_cache else None
        test_pairs = _load_cached_pairs(pairs_cache_dir, "test") if load_pairs_cache else None

        if train_pairs is None and len(train_base) > 0:
            train_pairs = _generate_ver_pairs(train_base, pairs_per_speaker=pairs_per_speaker, seed=seed + 42)
            if save_pairs_cache:
                _save_cached_pairs(pairs_cache_dir, "train", train_pairs)

        if val_pairs is None and len(val_base) > 0:
            val_pairs = _generate_ver_pairs(val_base, pairs_per_speaker=pairs_per_speaker, seed=seed + 43)
            if save_pairs_cache:
                _save_cached_pairs(pairs_cache_dir, "validation", val_pairs)

        if test_pairs is None and len(test_base) > 0:
            test_pairs = _generate_ver_pairs(test_base, pairs_per_speaker=pairs_per_speaker, seed=seed + 44)
            if save_pairs_cache:
                _save_cached_pairs(pairs_cache_dir, "test", test_pairs)

        print(f"✓ Zero-shot split: Val/test speakers are completely unseen during training")

    else:
        # Standard approach: Generate all pairs first, then split randomly
        print(f"\nGenerating speaker verification pairs ({pairs_per_speaker} per speaker)...")

        cache_meta = {
            "dataset": dataset_name,
            "config": dataset_config,
            "max_speakers": max_speakers,
            "selected_speakers": selected_speakers,
            "pairs_per_speaker": pairs_per_speaker,
            "max_duration": max_duration,
            "min_duration": min_duration,
            "split_by_speakers": split_by_speakers,
            "split_percentages": split_percentages,
            "seed": seed,
            "base_count": len(base_dataset),
        }
        pairs_cache_dir = _build_pairs_cache_dir(cache_root, cache_meta)
        load_pairs_cache = cache_splits and not force_rebuild
        save_pairs_cache = cache_splits

        train_pairs = _load_cached_pairs(pairs_cache_dir, "train") if load_pairs_cache else None
        val_pairs = _load_cached_pairs(pairs_cache_dir, "validation") if load_pairs_cache else None
        test_pairs = _load_cached_pairs(pairs_cache_dir, "test") if load_pairs_cache else None

        pairs_dataset = None
        if train_pairs is None or val_pairs is None or test_pairs is None:
            pairs_dataset = _generate_ver_pairs(
                base_dataset,
                pairs_per_speaker=pairs_per_speaker,
                seed=seed + 42,
            )

        # Split into train/val/test at the pair level
        # Note: We don't stratify because the pairs are already balanced (50/50)
        if split_percentages:
            # Convert to DatasetDict for apply_split_percentages
            pairs_dict = DatasetDict({"train": pairs_dataset})
            pairs_dict = apply_split_percentages(
                pairs_dict,
                split_percentages=split_percentages,
                train_split="train",
                seed=seed,
                stratify_by_column=None,  # Don't stratify (pairs already balanced)
            )
            train_pairs = train_pairs or pairs_dict.get("train")
            val_pairs = val_pairs or pairs_dict.get("validation")
            test_pairs = test_pairs or pairs_dict.get("test")
        else:
            # Default: 80/10/10 split (no stratification needed, pairs are balanced)
            splits = pairs_dataset.train_test_split(test_size=0.2, seed=seed)
            train_val_split = splits["train"].train_test_split(test_size=0.125, seed=seed)

            train_pairs = train_pairs or train_val_split["train"]
            val_pairs = val_pairs or train_val_split["test"]
            test_pairs = test_pairs or splits["test"]

        if save_pairs_cache and train_pairs is not None and val_pairs is not None and test_pairs is not None:
            _save_cached_pairs(pairs_cache_dir, "train", train_pairs)
            _save_cached_pairs(pairs_cache_dir, "validation", val_pairs)
            _save_cached_pairs(pairs_cache_dir, "test", test_pairs)

        print(f"⚠ Standard split: Val/test may contain seen speakers (different audio pairs)")

    # Note: We don't need to filter pairs by duration because:
    # 1. The base dataset was already filtered (before pairing)
    # 2. Pairs are made from pre-filtered samples
    # 3. The collator will trim audios to max_audio_length anyway
    # So adding duration columns to pairs is unnecessary

    # Add durations to both audios in pairs (for metadata/debugging only)
    if False:  # Disabled to save processing time
        print("Adding duration information to audio pairs...")
        if train_pairs:
            train_pairs = _add_duration_to_pairs(train_pairs, num_proc=effective_num_proc)
        if val_pairs:
            val_pairs = _add_duration_to_pairs(val_pairs, num_proc=effective_num_proc)
        if test_pairs:
            test_pairs = _add_duration_to_pairs(test_pairs, num_proc=effective_num_proc)

    # Build cache path
    dataset_key = dataset_name.replace("/", "_")
    config_key = dataset_config or "default"
    speakers_key = "all" if max_speakers is None else str(max_speakers).zfill(4)
    pairs_key = str(pairs_per_speaker).zfill(4)

    cache_name = (
        f"{dataset_key}_{config_key}"
        f"_speakers_{speakers_key}_pairs_{pairs_key}"
        f"_train_{_samples_key(max_train_samples)}"
        f"_val_{_samples_key(max_validation_samples)}"
        f"_test_{_samples_key(max_test_samples)}"
        f"_seed_{int(seed)}.json"
    )
    cache_path = cache_root / cache_name

    # Label names for binary classification
    label_names = ["no", "yes"]

    # Create DatasetDict for caching
    dataset_dict = DatasetDict()
    if train_pairs is not None:
        dataset_dict["train"] = train_pairs
    if val_pairs is not None:
        dataset_dict["validation"] = val_pairs
    if test_pairs is not None:
        dataset_dict["test"] = test_pairs

    # Cache and sample splits
    train_subset, validation_subset, test_subset, splits_metadata, payload_seed = cache_and_sample_splits(
        dataset_dict,
        cache_path=cache_path,
        max_train_samples=max_train_samples,
        max_validation_samples=max_validation_samples,
        max_test_samples=max_test_samples,
        seed=seed,
        manifest_fields=MANIFEST_FIELDS,
        audio_column=None,  # We have two audio columns
        cache_splits=cache_splits,
        force_rebuild=force_rebuild,
        additional_metadata={
            "dataset": dataset_name,
            "config": dataset_config,
            "label_names": label_names,
            "pairs_per_speaker": pairs_per_speaker,
            "max_speakers": max_speakers,
            "selected_speakers": selected_speakers,
        },
    )

    # Print summary
    speaker_info = f"{len(selected_speakers)} speakers" if selected_speakers else "all speakers"
    print_dataset_summary(
        task_emoji="🔍",
        task_name="Speaker verification dataset",
        train_subset=train_subset,
        validation_subset=validation_subset,
        test_subset=test_subset,
        splits_metadata=splits_metadata,
        label_names=label_names,
        seed=payload_seed,
        num_proc=effective_num_proc,
        extra_info=f"{speaker_info}, {pairs_per_speaker} pairs/speaker",
    )

    return train_subset, validation_subset, test_subset, label_names


@dataclass
class SpeakerVerCollator:
    """Prepare batches for speaker verification fine-tuning.

    Concatenates two audio clips with a silence gap and uses binary classification (yes/no).
    """

    processor: Any
    sampling_rate: int
    label_names: Sequence[str]  # ["no", "yes"]
    max_audio_length: Optional[float] = 10.0  # Max duration per audio in seconds
    audio_gap_seconds: float = 0.5  # Silence gap between audios

    def _label_to_text(self, value: Any) -> str:
        """Convert label index to text (0 → 'no', 1 → 'yes')."""
        if value is None:
            return ""
        try:
            index = int(value)
        except (TypeError, ValueError):
            return str(value)
        if 0 <= index < len(self.label_names):
            return self.label_names[index]
        return str(index)

    def _build_instruction(self, feature: Dict[str, Any]) -> str:
        """Build the instruction text for the user message."""
        return (
            "Listen to the two audio segments. Do they belong to the same speaker? "
            "Answer 'yes' if they are the same speaker, or 'no' if they are different speakers."
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process batch with audio concatenation."""
        # Extract audio pairs - handle both dict and direct AudioDecoder format
        audio_a_arrays = []
        audio_b_arrays = []
        for feature in features:
            audio_a = feature["audio_a"]
            audio_b = feature["audio_b"]

            # Extract array - handle both dict format and AudioDecoder objects
            if isinstance(audio_a, dict):
                audio_a = audio_a.get("array", np.array([], dtype=np.float32))
            if isinstance(audio_b, dict):
                audio_b = audio_b.get("array", np.array([], dtype=np.float32))

            # If still not a numpy array, use the extraction function
            if not isinstance(audio_a, np.ndarray):
                audio_a = _extract_audio_array(audio_a)
            if not isinstance(audio_b, np.ndarray):
                audio_b = _extract_audio_array(audio_b)

            audio_a_arrays.append(audio_a)
            audio_b_arrays.append(audio_b)

        # Trim both audios to max_audio_length
        if self.max_audio_length is not None:
            max_samples = int(self.max_audio_length * self.sampling_rate)
            audio_a_arrays = [
                arr[:max_samples] if len(arr) > max_samples else arr
                for arr in audio_a_arrays
            ]
            audio_b_arrays = [
                arr[:max_samples] if len(arr) > max_samples else arr
                for arr in audio_b_arrays
            ]

        # Create silence gap
        silence_samples = int(self.audio_gap_seconds * self.sampling_rate)

        # Concatenate audio pairs with silence gap
        concatenated_audios = []
        for audio_a, audio_b in zip(audio_a_arrays, audio_b_arrays):
            # Ensure same dtype
            if audio_a.dtype != audio_b.dtype:
                audio_b = audio_b.astype(audio_a.dtype)

            # Create silence with same dtype
            silence = np.zeros(silence_samples, dtype=audio_a.dtype)

            # Concatenate
            concatenated = np.concatenate([audio_a, silence, audio_b])
            concatenated_audios.append(concatenated)

        # Extract labels
        label_strings = [self._label_to_text(feature.get("label")) for feature in features]

        # Configure tokenizer
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "padding_side", None) != "left":
            tokenizer.padding_side = "left"

        # Build prompts using chat template
        prompts = []
        for feature, label in zip(features, label_strings):
            instruction = self._build_instruction(feature)

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": None},  # Concatenated audio
                        {"type": "text", "text": instruction}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": label}  # "yes" or "no"
                    ]
                }
            ]

            prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=False,
                tokenize=False
            )
            prompts.append(prompt)

        # Process with model processor
        inputs = self.processor(
            audio=concatenated_audios,
            sampling_rate=self.sampling_rate,
            text=prompts,
            return_tensors="pt",
            padding=True,
        )

        # Create labels with masking
        labels = inputs["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        audio_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.audio_token
        )

        # Mask padding and audio tokens
        labels = labels.masked_fill(labels == pad_id, -100)
        labels = labels.masked_fill(labels == audio_token_id, -100)

        # Mask everything except the assistant's response
        for i, label in enumerate(label_strings):
            label_tokens = tokenizer.encode(label, add_special_tokens=False)
            input_ids = inputs["input_ids"][i]
            label_length = len(label_tokens)

            # Search for the label tokens in the sequence, preferring the last match
            found = False
            for j in range(len(input_ids) - label_length, -1, -1):
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
