from __future__ import annotations

import csv
import json
from pathlib import Path
import wave

import pytest

from tasks.vocalsound.dataset import load_vocalsound_dataset


LABEL_NAMES = ["cough", "sneeze", "throat clearing", "sniff", "laugh", "exhale"]


def _write_silent_wav(path: Path, *, sample_rate: int = 16000, duration_ms: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(sample_rate * (duration_ms / 1000.0))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frame_count)


def _write_label_map(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "mid", "display_name"])
        writer.writeheader()
        for idx, label in enumerate(LABEL_NAMES):
            writer.writerow({"index": idx, "mid": f"/m/{idx:02d}", "display_name": label})


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle)


def _build_bundle(tmp_path: Path, *, include_bad_rows: bool = False, unknown_label: bool = False) -> Path:
    root = tmp_path / "vocalsound"
    audio_root = root / "audio"
    _write_silent_wav(audio_root / "train_short.wav", duration_ms=30)
    _write_silent_wav(audio_root / "train_long.wav", duration_ms=240)
    _write_silent_wav(audio_root / "val.wav", duration_ms=40)
    _write_silent_wav(audio_root / "test.wav", duration_ms=50)
    _write_label_map(root / "class_labels_indices_vs.csv")

    train_rows = [
        {"wav": "audio/train_short.wav", "label": "cough"},
        {"wav": "audio/train_long.wav", "label": "sneeze"},
    ]
    val_rows = [{"wav": "audio/val.wav", "label": "sniff"}]
    test_rows = [{"wav": "audio/test.wav", "label": "laugh"}]

    if include_bad_rows:
        train_rows.extend(
            [
                {"wav": "", "label": "cough"},
                {"label": "sneeze"},
                {"wav": "audio/does_not_exist.wav", "label": "cough"},
            ]
        )

    if unknown_label:
        train_rows.append({"wav": "audio/train_short.wav", "label": "unknown_sound"})

    _write_manifest(root / "vocalsound_train_data.json", train_rows)
    _write_manifest(root / "vocalsound_valid_data.json", val_rows)
    _write_manifest(root / "vocalsound_eval_data.json", test_rows)
    return root


def test_vocalsound_loader_loads_local_bundle_and_preserves_label_map_order(tmp_path: Path) -> None:
    data_dir = _build_bundle(tmp_path)
    train_ds, val_ds, test_ds, label_names = load_vocalsound_dataset(
        data_dir=str(data_dir),
        cache_splits=False,
        num_proc=1,
    )

    assert train_ds is not None and val_ds is not None and test_ds is not None
    assert len(train_ds) == 2
    assert len(val_ds) == 1
    assert len(test_ds) == 1
    assert label_names == LABEL_NAMES
    assert train_ds.features["label"].names == LABEL_NAMES


def test_vocalsound_loader_applies_duration_filter_and_sample_caps(tmp_path: Path) -> None:
    data_dir = _build_bundle(tmp_path)
    train_ds, val_ds, test_ds, _ = load_vocalsound_dataset(
        data_dir=str(data_dir),
        max_duration=0.08,
        max_train_samples=1,
        max_validation_samples=1,
        max_test_samples=1,
        cache_splits=False,
        num_proc=1,
    )

    assert train_ds is not None and val_ds is not None and test_ds is not None
    assert len(train_ds) == 1
    assert len(val_ds) == 1
    assert len(test_ds) == 1


def test_vocalsound_loader_fails_fast_on_missing_manifest(tmp_path: Path) -> None:
    data_dir = _build_bundle(tmp_path)
    (data_dir / "vocalsound_valid_data.json").unlink()

    with pytest.raises(FileNotFoundError) as exc_info:
        load_vocalsound_dataset(data_dir=str(data_dir), cache_splits=False, num_proc=1)
    assert "validation manifest" in str(exc_info.value).lower()


def test_vocalsound_loader_skips_bad_rows_but_errors_on_unknown_labels(tmp_path: Path) -> None:
    data_dir_bad = _build_bundle(tmp_path / "bad_rows", include_bad_rows=True)
    train_ds, _, _, _ = load_vocalsound_dataset(
        data_dir=str(data_dir_bad),
        cache_splits=False,
        num_proc=1,
    )
    assert train_ds is not None
    assert len(train_ds) == 2

    data_dir_unknown = _build_bundle(tmp_path / "unknown_labels", unknown_label=True)
    with pytest.raises(ValueError) as exc_info:
        load_vocalsound_dataset(
            data_dir=str(data_dir_unknown),
            cache_splits=False,
            num_proc=1,
        )
    assert "not present in label map" in str(exc_info.value)
