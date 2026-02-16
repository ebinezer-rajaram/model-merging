#!/usr/bin/env python3
"""Analyze cached speaker verification pairs and count unique speakers.

This script inspects cached speaker-verification pair datasets under:
  artifacts/speaker_ver/datasets/pairs

It estimates the number of distinct speakers present in the cached pairs by
using the `speaker_id` field:
  - For positive pairs (label == 1), speaker_id is the single speaker ID.
  - For negative pairs (label == 0), speaker_id is formatted as "speakerA_speakerB".
    We split on "_" to recover the two speaker IDs when possible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from datasets import Dataset


DEFAULT_CACHE_ROOT = Path("artifacts") / "speaker_ver" / "datasets" / "pairs"


def _find_cache_dirs(cache_root: Path) -> List[Path]:
    if not cache_root.exists():
        return []
    return sorted([p for p in cache_root.iterdir() if p.is_dir()])


def _pick_cache_dir(cache_root: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Cache dir not found: {explicit}")
        return explicit

    candidates = _find_cache_dirs(cache_root)
    if not candidates:
        raise FileNotFoundError(f"No cache directories found under {cache_root}")
    if len(candidates) == 1:
        return candidates[0]

    # Pick most recently modified dir if multiple.
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_split(cache_dir: Path, split: str) -> Optional[Dataset]:
    split_dir = cache_dir / split
    if not split_dir.exists():
        return None
    return Dataset.load_from_disk(str(split_dir))


def _extract_speakers(
    dataset: Dataset,
) -> Tuple[Set[str], int]:
    if "speaker_id" not in dataset.column_names:
        raise ValueError("Dataset does not contain 'speaker_id' column.")

    speaker_ids = dataset["speaker_id"]
    labels = dataset["label"] if "label" in dataset.column_names else None

    speakers: Set[str] = set()
    ambiguous_neg = 0

    for idx, sid in enumerate(speaker_ids):
        label = labels[idx] if labels is not None else None
        sid_str = str(sid)

        if label == 1:
            speakers.add(sid_str)
            continue

        # Negative or unknown: try to split "speakerA_speakerB"
        if "_" in sid_str:
            parts = sid_str.split("_")
            if len(parts) == 2:
                speakers.update(parts)
            else:
                ambiguous_neg += 1
                speakers.add(sid_str)
        else:
            speakers.add(sid_str)

    return speakers, ambiguous_neg


def _iter_splits(split_arg: str) -> Iterable[str]:
    if split_arg == "all":
        return ("train", "validation", "test")
    return (split_arg,)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count unique speakers in cached speaker verification pairs."
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help="Root directory containing cached pairs (default: artifacts/speaker_ver/datasets/pairs)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Specific cache directory to use (overrides --cache-root selection).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test", "all"],
        default="all",
        help="Which split to analyze (default: all).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print results as JSON.",
    )

    args = parser.parse_args()

    cache_dir = _pick_cache_dir(args.cache_root, args.cache_dir)
    splits = list(_iter_splits(args.split))

    per_split: Dict[str, Dict[str, int]] = {}
    all_speakers: Set[str] = set()
    total_ambiguous = 0

    for split in splits:
        ds = _load_split(cache_dir, split)
        if ds is None:
            per_split[split] = {"samples": 0, "unique_speakers": 0, "ambiguous_negative_ids": 0}
            continue

        speakers, ambiguous = _extract_speakers(ds)
        per_split[split] = {
            "samples": len(ds),
            "unique_speakers": len(speakers),
            "ambiguous_negative_ids": ambiguous,
        }
        all_speakers.update(speakers)
        total_ambiguous += ambiguous

    result = {
        "cache_dir": str(cache_dir),
        "splits": per_split,
        "unique_speakers_all_splits": len(all_speakers),
        "ambiguous_negative_ids_total": total_ambiguous,
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"Cache dir: {result['cache_dir']}")
    for split in splits:
        info = per_split[split]
        print(
            f"{split}: {info['samples']} samples, "
            f"{info['unique_speakers']} unique speakers "
            f"(ambiguous negatives: {info['ambiguous_negative_ids']})"
        )
    if len(splits) > 1:
        print(f"All splits (union): {result['unique_speakers_all_splits']} unique speakers")
        if total_ambiguous:
            print(f"Ambiguous negative speaker_id values total: {total_ambiguous}")


if __name__ == "__main__":
    main()
