"""Custom MELD dataset loader from local files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset, DatasetDict, Features, Value, Audio, ClassLabel


# MELD emotion labels (7 classes)
MELD_EMOTIONS = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]

# Expected CSV columns from MELD
CSV_COLUMNS = [
    "Sr No.",
    "Utterance",
    "Speaker",
    "Emotion",
    "Sentiment",
    "Dialogue_ID",
    "Utterance_ID",
    "Season",
    "Episode",
    "StartTime",
    "EndTime",
]


def load_meld_from_local(
    data_dir: str | Path = "data/meld",
    audio_format: str = "mp4",
) -> DatasetDict:
    """
    Load MELD dataset from local files.

    Args:
        data_dir: Path to directory containing CSV files and audio/video files
        audio_format: Format of audio files ('mp4' for video, 'wav' for extracted audio)

    Returns:
        DatasetDict with train, validation, test splits
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        fallback_dir = Path("data") / data_dir
        if fallback_dir.exists():
            print(f"Data directory not found at {data_dir}, using {fallback_dir} instead.")
            data_dir = fallback_dir

    # Load CSV annotations
    train_csv = data_dir / "train_sent_emo.csv"
    dev_csv = data_dir / "dev_sent_emo.csv"
    test_csv = data_dir / "test_sent_emo.csv"

    # Audio/video directories - they're in MELD.Raw subdirectory
    meld_raw_dir = data_dir / "MELD.Raw"
    if audio_format == "mp4":
        train_audio_dir = meld_raw_dir / "train_splits"
        dev_audio_dir = meld_raw_dir / "dev_splits_complete"
        test_audio_dir = meld_raw_dir / "output_repeated_splits_test"
    else:
        # For extracted audio
        train_audio_dir = data_dir / "audio" / "train"
        dev_audio_dir = data_dir / "audio" / "dev"
        test_audio_dir = data_dir / "audio" / "test"

    # Define features
    features = Features({
        "Utterance": Value("string"),
        "Speaker": Value("string"),
        "Emotion": ClassLabel(names=MELD_EMOTIONS),
        "Sentiment": Value("string"),
        "Dialogue_ID": Value("int32"),
        "Utterance_ID": Value("int32"),
        "Season": Value("int32"),
        "Episode": Value("int32"),
        "audio": Audio(sampling_rate=16000),
    })

    # Load each split
    def load_split(csv_path: Path, audio_dir: Path) -> Dataset:
        """Load a single split from CSV and audio files."""
        data: List[Dict[str, Any]] = []
        skipped_count = 0

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dialogue_id = int(row["Dialogue_ID"])
                utterance_id = int(row["Utterance_ID"])

                # Construct audio filename: dia{dialogue_id}_utt{utterance_id}.{format}
                audio_filename = f"dia{dialogue_id}_utt{utterance_id}.{audio_format}"
                audio_path = audio_dir / audio_filename

                # Skip if audio file doesn't exist or is too small (likely corrupted)
                if not audio_path.exists():
                    skipped_count += 1
                    continue

                # Skip files that are suspiciously small (< 1KB, likely corrupted)
                if audio_path.stat().st_size < 1024:
                    skipped_count += 1
                    continue

                # Normalize emotion label to lowercase
                emotion = row["Emotion"].strip().lower()

                data.append({
                    "Utterance": row["Utterance"].strip(),
                    "Speaker": row["Speaker"].strip(),
                    "Emotion": emotion,
                    "Sentiment": row["Sentiment"].strip().lower(),
                    "Dialogue_ID": dialogue_id,
                    "Utterance_ID": utterance_id,
                    "Season": int(row["Season"]),
                    "Episode": int(row["Episode"]),
                    "audio": str(audio_path),
                })

        if skipped_count > 0:
            print(f"  Skipped {skipped_count} missing or corrupted audio files")

        if len(data) == 0:
            raise ValueError(f"No valid audio files found in {audio_dir}")

        return Dataset.from_list(data, features=features)

    # Load all splits
    print(f"Loading MELD from {data_dir}...")
    train_dataset = load_split(train_csv, train_audio_dir)
    dev_dataset = load_split(dev_csv, dev_audio_dir)
    test_dataset = load_split(test_csv, test_audio_dir)

    print(f"Loaded {len(train_dataset)} train, {len(dev_dataset)} dev, {len(test_dataset)} test samples")

    return DatasetDict({
        "train": train_dataset,
        "validation": dev_dataset,
        "test": test_dataset,
    })


__all__ = ["load_meld_from_local", "MELD_EMOTIONS"]
