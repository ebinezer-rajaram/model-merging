from __future__ import annotations

from datasets import Dataset, DatasetDict

import tasks.asr.dataset as asr_dataset
import tasks.st.dataset as st_dataset


def _audio(length: int = 4) -> dict:
    return {"array": [0.0] * length, "sampling_rate": 16000}


def _add_duration(example: dict) -> dict:
    return {**example, "duration": 0.1}


def _add_duration_to_dataset(dataset: DatasetDict, **kwargs) -> DatasetDict:
    del kwargs
    return DatasetDict(
        {
            split: subset if "duration" in subset.column_names else subset.add_column("duration", [0.1] * len(subset))
            for split, subset in dataset.items()
        }
    )


def _filter_by_duration(dataset: DatasetDict, **kwargs) -> DatasetDict:
    del kwargs
    return dataset


def _cache_and_sample_splits(dataset: DatasetDict, **kwargs):
    def _select(name: str, limit):
        if name not in dataset:
            return None
        subset = dataset[name]
        if limit is not None:
            subset = subset.select(range(min(int(limit), len(subset))))
        return subset

    train = _select("train", kwargs.get("max_train_samples"))
    validation = _select("validation", kwargs.get("max_validation_samples"))
    test = _select("test", kwargs.get("max_test_samples"))
    metadata = {
        name: {"hours": 0.0}
        for name, subset in (("train", train), ("validation", validation), ("test", test))
        if subset is not None
    }
    return train, validation, test, metadata, kwargs.get("seed", 0)


def test_asr_loader_builds_train_validation_and_optional_test(monkeypatch, tmp_path) -> None:
    clean = DatasetDict(
        {
            "train.100": Dataset.from_list(
                [
                    {"audio": _audio(), "text": "train one", "id": "tr1", "speaker_id": "s1", "chapter_id": "c1"},
                    {"audio": _audio(), "text": "train two", "id": "tr2", "speaker_id": "s2", "chapter_id": "c2"},
                ]
            ),
            "validation": Dataset.from_list(
                [{"audio": _audio(), "text": "validation", "id": "val1", "speaker_id": "s1", "chapter_id": "c3"}]
            ),
            "test": Dataset.from_list(
                [{"audio": _audio(), "text": "test clean", "id": "te1", "speaker_id": "s3", "chapter_id": "c4"}]
            ),
        }
    )
    other = DatasetDict(
        {
            "test": Dataset.from_list(
                [{"audio": _audio(), "text": "test other", "id": "to1", "speaker_id": "s4", "chapter_id": "c5"}]
            )
        }
    )

    def _load_dataset(name, config):
        assert name == "librispeech_asr"
        return other if config == "other" else clean

    monkeypatch.setattr(asr_dataset, "load_dataset", _load_dataset)
    monkeypatch.setattr(asr_dataset, "add_duration", _add_duration)

    train_ds, val_ds, test_ds = asr_dataset.load_librispeech_subset(
        train_hours=1.0,
        val_hours=1.0,
        return_test_split=True,
        test_split="test-other",
        cache_dir=tmp_path,
        cache_splits=False,
        num_proc=1,
    )

    assert len(train_ds) == 2
    assert len(val_ds) == 1
    assert len(test_ds) == 1
    assert test_ds[0]["text"] == "test other"


def test_st_loader_normalizes_columns_filters_remove_and_samples(monkeypatch, tmp_path) -> None:
    dataset = DatasetDict(
        {
            "train": Dataset.from_list(
                [
                    {"speech": {"bytes": b"abc", "path": None}, "sentence": "hello", "target": "hallo", "id": "ok"},
                    {"speech": {"bytes": b"abc", "path": None}, "sentence": "REMOVE", "target": "skip", "id": "remove"},
                ]
            ),
            "validation": Dataset.from_list(
                [{"speech": {"bytes": b"abc", "path": None}, "sentence": "val", "target": "wert", "id": "val"}]
            ),
            "test": Dataset.from_list(
                [{"speech": {"bytes": b"abc", "path": None}, "sentence": "test", "target": "test", "id": "test"}]
            ),
        }
    )
    monkeypatch.setattr(st_dataset, "load_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(st_dataset, "add_duration_to_dataset", _add_duration_to_dataset)
    monkeypatch.setattr(st_dataset, "filter_by_duration", _filter_by_duration)
    monkeypatch.setattr(st_dataset, "cache_and_sample_splits", _cache_and_sample_splits)

    train_ds, val_ds, test_ds = st_dataset.load_covost2_dataset(
        source_column="sentence",
        translation_column="target",
        max_train_samples=1,
        cache_dir=tmp_path,
        cache_splits=False,
        num_proc=1,
    )

    assert train_ds is not None and len(train_ds) == 1
    assert train_ds[0]["text"] == "hello"
    assert train_ds[0]["translation"] == "hallo"
    assert val_ds is not None and len(val_ds) == 1
    assert test_ds is not None and len(test_ds) == 1
