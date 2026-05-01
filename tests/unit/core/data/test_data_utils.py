from __future__ import annotations

import numpy as np
import pytest
from datasets import Dataset, DatasetDict

from core.data.dataset_utils import (
    add_duration,
    build_manifest,
    build_split_metadata,
    compute_speaker_stats,
    compute_split_hours,
    filter_dataset_columns,
    hours_key,
    hours_to_seconds,
    load_cached_split,
    normalize_audio,
    normalize_split_metadata,
    num_proc_map_kwargs,
    resolve_num_proc,
    save_cached_split,
    select_indices_by_duration,
    select_random_indices,
    subset_dataset_by_metadata,
)


def test_duration_num_proc_and_sampling_helpers_are_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.data.dataset_utils.cpu_count", lambda: 8)
    assert resolve_num_proc("auto") == 4
    assert resolve_num_proc(None) == 4
    assert resolve_num_proc(0) == 1
    with pytest.raises(ValueError, match="Unsupported num_proc"):
        resolve_num_proc("bad")
    assert num_proc_map_kwargs(1) == {}
    assert num_proc_map_kwargs(2) == {"num_proc": 2}

    assert add_duration({"audio": {"array": [0.0] * 160, "sampling_rate": 16000}})["duration"] == pytest.approx(0.01)
    assert add_duration({"audio": {"array": None, "sampling_rate": 16000}})["duration"] == 0.0
    assert hours_to_seconds(0.5) == 1800.0
    assert hours_key(None) == "full"
    assert hours_key(1.25) == "001250"
    assert select_indices_by_duration([1.0, 2.0, 3.0], 0, seed=7) == []
    assert select_indices_by_duration([1.0, 2.0, 3.0], None, seed=7) == [0, 1, 2]
    assert select_random_indices(5, 0, seed=7) == []
    assert sorted(select_random_indices(5, 2, seed=7)) == sorted(select_random_indices(5, 2, seed=7))


def test_audio_stats_normalization_and_split_metadata(tmp_path) -> None:
    ds = Dataset.from_list(
        [
            {"id": "a", "duration": 1.0, "speaker": "s1", "audio": {"array": [1.0, -1.0], "sampling_rate": 2}},
            {"id": "b", "duration": 2.0, "speaker": "s1", "audio": {"array": [0.5, -0.5], "sampling_rate": 2}},
        ]
    )

    stats = compute_speaker_stats(ds, speaker_column="speaker")
    assert stats["s1"]["count"] == 2
    normalized = normalize_audio({"audio": {"array": np.array([0.5, -0.5]), "sampling_rate": 2}})
    assert np.max(np.abs(normalized["audio"]["array"])) <= 1.0
    per_speaker = normalize_audio(
        {"audio": {"array": np.array([0.5, -0.5]), "sampling_rate": 2}, "speaker": "s1"},
        speaker_column="speaker",
        speaker_stats=stats,
        normalize_per_speaker=True,
    )
    assert np.max(np.abs(per_speaker["audio"]["array"])) <= 1.0

    assert compute_split_hours(ds, [0, 99]) == pytest.approx(1.0 / 3600.0)
    manifest = build_manifest(ds, [1, 0], fields=["id"], include_index=True)
    assert manifest[0]["id"] == "b"
    assert manifest[0]["index"] == 1
    metadata = build_split_metadata(ds, [0, 1], manifest_fields=["id"])
    assert metadata["hours"] == pytest.approx(3.0 / 3600.0)

    assert len(subset_dataset_by_metadata(ds, {"indices": [1]})) == 1
    assert len(subset_dataset_by_metadata(ds, {"indices": []})) == 0
    assert normalize_split_metadata({"train": {"indices": ["1"], "hours": "2.5"}})["train"]["indices"] == [1]

    cache_path = tmp_path / "cache" / "split.json"
    assert load_cached_split(cache_path) is None
    save_cached_split(cache_path, {"indices": [0]})
    assert load_cached_split(cache_path) == {"indices": [0]}
    assert filter_dataset_columns(ds, ["id"], always_keep=["duration"]).column_names == ["id", "duration"]


def test_filter_dataset_columns_noops_when_no_drop_needed() -> None:
    ds = Dataset.from_list([{"id": "a"}])
    assert filter_dataset_columns(ds, ["id"]) is ds
