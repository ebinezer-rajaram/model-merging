from __future__ import annotations

from datasets import Dataset, DatasetDict

from core.tasks.dataset import load_and_prepare_dataset
from tests.helpers.core import audio_dataset_dict


def test_load_and_prepare_dataset_standardizes_patched_loader(monkeypatch) -> None:
    monkeypatch.setattr("core.tasks.dataset.load_dataset", lambda *args, **kwargs: audio_dataset_dict())
    dataset, audio_column = load_and_prepare_dataset(
        "synthetic",
        label_column="intent",
        text_column="sentence",
        audio_column="speech",
        train_split="train",
        validation_split="validation",
        test_split="test",
    )
    assert audio_column == "audio"
    assert set(dataset) == {"train", "validation", "test"}
    assert dataset["train"].features["label"].names == ["book", "play"]
    assert "text" in dataset["train"].column_names


def test_load_and_prepare_dataset_applies_split_percentages(monkeypatch) -> None:
    raw = DatasetDict(
        {
            "train": Dataset.from_list(
                [
                    {"audio": {"array": [0.0], "sampling_rate": 1}, "intent": "book" if idx % 2 == 0 else "play", "text": str(idx)}
                    for idx in range(12)
                ]
            )
        }
    )
    monkeypatch.setattr("core.tasks.dataset.load_dataset", lambda *args, **kwargs: raw)
    dataset, audio_column = load_and_prepare_dataset(
        "synthetic",
        label_column="intent",
        split_percentages={"train": 0.5, "validation": 0.25, "test": 0.25},
        stratify_by_column=None,
        seed=13,
    )
    assert audio_column == "audio"
    assert set(dataset) == {"train", "validation", "test"}
    assert sum(len(split) for split in dataset.values()) == 12


def test_load_and_prepare_dataset_rejects_non_datasetdict(monkeypatch) -> None:
    monkeypatch.setattr("core.tasks.dataset.load_dataset", lambda *args, **kwargs: Dataset.from_list([{"x": 1}]))
    try:
        load_and_prepare_dataset("synthetic")
    except TypeError as exc:
        assert "Expected DatasetDict" in str(exc)
    else:
        raise AssertionError("Expected TypeError")
