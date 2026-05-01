from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset, DatasetDict

from tasks.speaker_id.dataset import _limit_samples_per_speaker, _select_speakers
from tasks.speaker_ver.dataset import _generate_ver_pairs
from tasks.speech_qa.dataset import (
    _allocate_total_samples,
    _derive_answer_letter,
    _normalize_answers,
    _normalize_answers_for_mmsu,
)
from tasks.st.dataset import _compute_valid_audio_indices
from tasks.vocalsound.dataset import _map_label_to_index, _normalize_label_token, _resolve_audio_path


def test_speech_qa_allocates_total_samples_by_largest_remainder() -> None:
    dataset = DatasetDict(
        {
            "train": Dataset.from_list([{"id": idx} for idx in range(6)]),
            "validation": Dataset.from_list([{"id": idx} for idx in range(3)]),
            "test": Dataset.from_list([{"id": idx} for idx in range(1)]),
        }
    )

    allocation = _allocate_total_samples(dataset, max_total_samples=5)

    assert allocation == {"train": 3, "validation": 2, "test": 0}
    assert sum(value or 0 for value in allocation.values()) == 5


def test_speech_qa_normalizes_answers_and_derives_mmsu_letter() -> None:
    normalized = _normalize_answers({"answers": {"text": [" Beta ", ""]}}, "answers")
    assert normalized["answers"] == ["Beta"]
    assert normalized["label_text"] == "Beta"

    example = {
        "id": "mmsu_1",
        "answer_gt": "Paris",
        "choice_a": "Paris",
        "choice_b": "London",
        "choice_c": "Berlin",
        "choice_d": "Madrid",
    }
    normalized_mmsu = _normalize_answers_for_mmsu(example, "answer_gt")

    assert _derive_answer_letter(example, ["paris"]) == "A"
    assert normalized_mmsu["answers"] == ["Paris"]
    assert normalized_mmsu["answer_letter"] == "A"
    assert normalized_mmsu["label_text"] == "A"


def test_speech_qa_mmsu_normalization_errors_when_answer_is_not_a_choice() -> None:
    with pytest.raises(ValueError, match="Unable to derive MMSU answer letter"):
        _normalize_answers_for_mmsu(
            {
                "id": "bad",
                "answer_gt": "Rome",
                "choice_a": "Paris",
                "choice_b": "London",
                "choice_c": "Berlin",
                "choice_d": "Madrid",
            },
            "answer_gt",
        )


def test_speaker_id_selects_top_speakers_and_limits_per_speaker() -> None:
    dataset = DatasetDict(
        {
            "train": Dataset.from_list(
                [
                    {"label": "a", "audio": {"array": [0.0]}},
                    {"label": "a", "audio": {"array": [0.1]}},
                    {"label": "a", "audio": {"array": [0.2]}},
                    {"label": "b", "audio": {"array": [0.3]}},
                    {"label": "b", "audio": {"array": [0.4]}},
                    {"label": "c", "audio": {"array": [0.5]}},
                ]
            ),
            "validation": Dataset.from_list(
                [
                    {"label": "a", "audio": {"array": [0.6]}},
                    {"label": "c", "audio": {"array": [0.7]}},
                ]
            ),
        }
    )

    selected_dataset, selected = _select_speakers(dataset, max_speakers=2, seed=0, train_split="train")
    limited = _limit_samples_per_speaker(selected_dataset, max_samples=1, seed=123)

    assert selected == ["a", "b"]
    assert selected_dataset["train"]["label"] == ["a", "a", "a", "b", "b"]
    assert selected_dataset["validation"]["label"] == ["a"]
    assert sorted(limited["train"]["label"]) == ["a", "b"]
    assert limited["validation"]["label"] == ["a"]


def test_speaker_ver_pair_generation_is_balanced_and_even() -> None:
    dataset = Dataset.from_list(
        [
            {"label": "speaker_a", "audio": {"array": np.array([0.1, 0.2], dtype=np.float32)}},
            {"label": "speaker_a", "audio": {"array": np.array([0.3, 0.4], dtype=np.float32)}},
            {"label": "speaker_b", "audio": {"array": np.array([0.5, 0.6], dtype=np.float32)}},
            {"label": "speaker_b", "audio": {"array": np.array([0.7, 0.8], dtype=np.float32)}},
        ]
    )

    pairs = _generate_ver_pairs(dataset, total_pairs=5, seed=7)

    assert len(pairs) == 4
    assert pairs["label"].count(1) == 2
    assert pairs["label"].count(0) == 2
    assert {"audio_a", "audio_b", "speaker_id"}.issubset(set(pairs.column_names))


def test_st_valid_audio_indices_accept_content_and_reject_empty_or_missing_audio(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFF")
    missing_path = tmp_path / "missing.wav"
    dataset = Dataset.from_list(
        [
            {"id": "bytes", "audio": {"bytes": b"abc", "path": None}},
            {"id": "path", "audio": {"bytes": None, "path": str(audio_path)}},
            {"id": "empty", "audio": {"bytes": b"", "path": None}},
            {"id": "missing", "audio": {"bytes": None, "path": str(missing_path)}},
            {"id": "none", "audio": None},
        ]
    )

    indices = _compute_valid_audio_indices(
        dataset,
        audio_column_name="audio",
        split_name="train",
        log_corrupted=False,
        validate_absolute_paths=True,
    )

    assert indices == [0, 1]


def test_vocalsound_label_and_path_helpers(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    wav_path = audio_root / "clip.wav"
    wav_path.write_bytes(b"RIFF")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("[]")
    label_names = ["cough", "throat clearing"]
    lookup = {"cough": 0, "throatclearing": 1, "/m/clear": 1}

    assert _normalize_label_token("Throat clearing!") == "throatclearing"
    assert _map_label_to_index("1", label_names, lookup) == 1
    assert _map_label_to_index("Throat Clearing", label_names, lookup) == 1
    assert _map_label_to_index("/m/clear", label_names, lookup) == 1
    assert _map_label_to_index("unknown", label_names, lookup) is None
    assert _resolve_audio_path("clip", manifest_path=manifest_path, audio_root=audio_root) == wav_path.resolve()
