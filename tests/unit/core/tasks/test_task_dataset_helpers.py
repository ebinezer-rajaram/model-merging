from __future__ import annotations

import pytest
from datasets import Dataset, DatasetDict

from core.tasks.dataset import (
    _ensure_audio_column,
    _extract_label_names,
    _normalize_target_count,
    _rename_column,
    _samples_key,
    apply_split_percentages,
    assemble_splits,
    cache_and_sample_splits,
    filter_by_duration,
    load_and_prepare_dataset,
    prepare_classification_dataset,
)
from tests.helpers.core import audio_dataset_dict


def test_private_dataset_column_helpers_validate_inputs() -> None:
    ds = audio_dataset_dict()
    assert _samples_key(None) == "full"
    assert _samples_key(7) == "000007"
    assert _normalize_target_count("max", None) is None
    assert _normalize_target_count("max", 3) == 3
    with pytest.raises(ValueError, match="non-negative"):
        _normalize_target_count("max", -1)

    renamed = _rename_column(ds, "speech", "audio")
    assert "audio" in renamed["train"].column_names
    with pytest.raises(ValueError, match="already exists"):
        _rename_column(renamed, "sentence", "audio")
    with pytest.raises(ValueError, match="not found"):
        _rename_column(ds, "missing", "audio")


def test_prepare_classification_dataset_encodes_labels_and_text() -> None:
    prepared = prepare_classification_dataset(audio_dataset_dict(), label_column="intent", text_column="sentence")
    assert "label" in prepared["train"].column_names
    assert "text" in prepared["train"].column_names
    assert _extract_label_names(prepared["train"]) == ["book", "play"]


def test_split_assembly_percentages_audio_and_duration_filtering(tmp_path) -> None:
    ds = DatasetDict(
        {
            "source": Dataset.from_list([{"audio": {"array": [0.0], "sampling_rate": 1}, "duration": idx} for idx in range(10)]),
            "heldout": Dataset.from_list([{"audio": {"array": [0.0], "sampling_rate": 1}, "duration": 1.0}]),
        }
    )
    assembled = assemble_splits(ds, train_split="source", validation_split=None, test_split="heldout")
    assert set(assembled) == {"train", "test"}

    split = apply_split_percentages(ds, [0.6, 0.2, 0.2], train_split="source", seed=1, stratify_by_column=None)
    assert set(split) == {"train", "validation", "test"}
    assert sum(len(split[name]) for name in split) == 10

    ensured, audio_name = _ensure_audio_column(audio_dataset_dict(), preferred="speech")
    assert audio_name == "audio"
    assert "audio" in ensured["train"].column_names
    with pytest.raises(ValueError, match="Could not find an audio column"):
        _ensure_audio_column(DatasetDict({"train": Dataset.from_list([{"label": 0}])}))

    filtered = filter_by_duration(DatasetDict({"train": ds["source"]}), max_duration=4.0, min_duration=2.0)
    assert filtered["train"]["duration"] == [2, 3, 4]


def test_cache_and_sample_splits_writes_and_reuses_metadata(tmp_path) -> None:
    ds = DatasetDict(
        {
            "train": Dataset.from_list([{"id": str(idx), "duration": 1.0, "audio": {"array": [0.0], "sampling_rate": 1}} for idx in range(5)]),
            "validation": Dataset.from_list([{"id": "v", "duration": 1.0, "audio": {"array": [0.0], "sampling_rate": 1}}]),
        }
    )
    cache_path = tmp_path / "splits.json"
    train, val, test, metadata, seed = cache_and_sample_splits(
        ds,
        cache_path=cache_path,
        max_train_samples=2,
        max_validation_samples=1,
        seed=5,
        manifest_fields=("id",),
    )
    assert len(train) == 2
    assert len(val) == 1
    assert test is None
    assert seed == 5
    assert cache_path.exists()

    train_again, _, _, metadata_again, _ = cache_and_sample_splits(ds, cache_path=cache_path, seed=999)
    assert train_again["id"] == train["id"]
    assert metadata_again["train"]["indices"] == metadata["train"]["indices"]


def test_load_and_prepare_dataset_uses_patched_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.tasks.dataset.load_dataset", lambda *args, **kwargs: audio_dataset_dict())
    dataset, audio_column = load_and_prepare_dataset(
        "dummy",
        label_column="intent",
        text_column="sentence",
        audio_column="speech",
        train_split="train",
        validation_split="validation",
        test_split="test",
    )
    assert audio_column == "audio"
    assert set(dataset) == {"train", "validation", "test"}
    assert "label" in dataset["train"].column_names
