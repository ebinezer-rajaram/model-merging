from __future__ import annotations

import json
from pathlib import Path
import wave

from tasks.speech_qa.dataset import load_speech_qa_dataset


def _write_silent_wav(path: Path, *, sample_rate: int = 16000, duration_ms: int = 50) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    num_frames = int(sample_rate * (duration_ms / 1000.0))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * num_frames)


def _build_spoken_squad_payload(question_id: str, question: str, answer: str) -> dict:
    return {
        "version": "1.1",
        "data": [
            {
                "title": "topic",
                "paragraphs": [
                    {
                        "context": "alpha beta gamma",
                        "qas": [
                            {
                                "id": question_id,
                                "question": question,
                                "answers": [{"text": answer, "answer_start": 6}],
                            }
                        ],
                    }
                ],
            }
        ],
    }


def test_local_spoken_squad_loader_builds_train_and_test_splits(tmp_path: Path) -> None:
    data_dir = tmp_path / "Spoken-SQuAD"
    (data_dir / "wav" / "train").mkdir(parents=True, exist_ok=True)
    (data_dir / "wav" / "test").mkdir(parents=True, exist_ok=True)

    (data_dir / "spoken_train-v1.1.json").write_text(
        json.dumps(_build_spoken_squad_payload("train_q1", "train question", "beta"))
    )
    (data_dir / "spoken_test-v1.1.json").write_text(
        json.dumps(_build_spoken_squad_payload("test_q1", "test question", "beta"))
    )

    _write_silent_wav(data_dir / "wav" / "train" / "1_1_1.wav")
    _write_silent_wav(data_dir / "wav" / "test" / "1_1_1.wav")

    train_ds, val_ds, test_ds, answers_map = load_speech_qa_dataset(
        dataset_name="local_spoken_squad",
        data_dir=data_dir,
        min_wavs_per_split=1,
        max_missing_audio_rate=0.0,
        split_percentages=None,
        train_split="train",
        validation_split=None,
        test_split="test",
        num_proc=1,
    )

    assert train_ds is not None
    assert val_ds is None
    assert test_ds is not None
    assert len(train_ds) == 1
    assert len(test_ds) == 1
    assert train_ds[0]["question"] == "train question"
    assert test_ds[0]["id"] == "test_q1"
    assert answers_map["train"][0] == ["beta"]
    assert answers_map["test"][0] == ["beta"]


def test_local_spoken_squad_loader_uses_noisy_test_variant(tmp_path: Path) -> None:
    data_dir = tmp_path / "Spoken-SQuAD"
    (data_dir / "wav" / "train").mkdir(parents=True, exist_ok=True)
    (data_dir / "wav" / "test").mkdir(parents=True, exist_ok=True)

    (data_dir / "spoken_train-v1.1.json").write_text(
        json.dumps(_build_spoken_squad_payload("train_q1", "train question", "beta"))
    )
    (data_dir / "spoken_test-v1.1_WER44.json").write_text(
        json.dumps(_build_spoken_squad_payload("test_q_wer44", "test noisy", "beta"))
    )

    _write_silent_wav(data_dir / "wav" / "train" / "1_1_1.wav")
    _write_silent_wav(data_dir / "wav" / "test" / "1_1_1.wav")

    _, _, test_ds, _ = load_speech_qa_dataset(
        dataset_name="local_spoken_squad",
        data_dir=data_dir,
        min_wavs_per_split=1,
        max_missing_audio_rate=0.0,
        noisy_test_variant="wer44",
        split_percentages=None,
        train_split="train",
        validation_split=None,
        test_split="test",
        num_proc=1,
    )

    assert test_ds is not None
    assert test_ds[0]["id"] == "test_q_wer44"


def test_local_spoken_squad_loader_fails_when_missing_audio_rate_exceeds_threshold(tmp_path: Path) -> None:
    data_dir = tmp_path / "Spoken-SQuAD"
    (data_dir / "wav" / "train").mkdir(parents=True, exist_ok=True)
    (data_dir / "wav" / "test").mkdir(parents=True, exist_ok=True)

    payload = {
        "version": "1.1",
        "data": [
            {
                "title": "topic",
                "paragraphs": [
                    {
                        "context": "alpha beta gamma",
                        "qas": [
                            {"id": "q1", "question": "q1?", "answers": [{"text": "beta", "answer_start": 6}]},
                            {"id": "q2", "question": "q2?", "answers": [{"text": "beta", "answer_start": 6}]},
                        ],
                    }
                ],
            }
        ],
    }
    (data_dir / "spoken_train-v1.1.json").write_text(json.dumps(payload))
    (data_dir / "spoken_test-v1.1.json").write_text(json.dumps(payload))

    # Provide wav only for test split and only one train/topic-paragraph mapping.
    _write_silent_wav(data_dir / "wav" / "test" / "1_1_1.wav")

    try:
        load_speech_qa_dataset(
            dataset_name="local_spoken_squad",
            data_dir=data_dir,
            min_wavs_per_split=0,
            max_missing_audio_rate=0.0,
            split_percentages=None,
            train_split="train",
            validation_split=None,
            test_split="test",
            num_proc=1,
        )
        assert False, "Expected ValueError due to missing audio mappings."
    except ValueError as exc:
        assert "missing audio mappings" in str(exc)


def test_local_spoken_squad_loader_can_fallback_to_train_only_source(tmp_path: Path) -> None:
    data_dir = tmp_path / "Spoken-SQuAD"
    (data_dir / "wav" / "train").mkdir(parents=True, exist_ok=True)
    (data_dir / "wav" / "test").mkdir(parents=True, exist_ok=True)

    qas = [
        {
            "id": f"q{i}",
            "question": f"question {i}?",
            "answers": [{"text": "beta", "answer_start": 6}],
        }
        for i in range(10)
    ]
    payload = {
        "version": "1.1",
        "data": [
            {
                "title": "topic",
                "paragraphs": [
                    {
                        "context": "alpha beta gamma",
                        "qas": qas,
                    }
                ],
            }
        ],
    }
    (data_dir / "spoken_train-v1.1.json").write_text(json.dumps(payload))
    (data_dir / "spoken_test-v1.1.json").write_text(json.dumps(payload))
    _write_silent_wav(data_dir / "wav" / "train" / "1_1_1.wav")

    train_ds, val_ds, test_ds, answers_map = load_speech_qa_dataset(
        dataset_name="local_spoken_squad",
        data_dir=data_dir,
        min_wavs_per_split=1,
        max_missing_audio_rate=0.0,
        allow_train_only_fallback=True,
        split_percentages={"train": 0.8, "validation": 0.1, "test": 0.1},
        train_split="train",
        validation_split="validation",
        test_split="test",
        num_proc=1,
    )

    assert train_ds is not None
    assert val_ds is not None
    assert test_ds is not None
    assert len(train_ds) + len(val_ds) + len(test_ds) == 10
    metadata = answers_map.get("_metadata", {})
    assert metadata.get("split_origin") == "train_only_fallback"


def test_local_spoken_squad_loader_supports_max_total_samples(tmp_path: Path) -> None:
    data_dir = tmp_path / "Spoken-SQuAD"
    (data_dir / "wav" / "train").mkdir(parents=True, exist_ok=True)
    (data_dir / "wav" / "test").mkdir(parents=True, exist_ok=True)

    qas = [
        {
            "id": f"q{i}",
            "question": f"question {i}?",
            "answers": [{"text": "beta", "answer_start": 6}],
        }
        for i in range(20)
    ]
    payload = {
        "version": "1.1",
        "data": [
            {
                "title": "topic",
                "paragraphs": [
                    {
                        "context": "alpha beta gamma",
                        "qas": qas,
                    }
                ],
            }
        ],
    }
    (data_dir / "spoken_train-v1.1.json").write_text(json.dumps(payload))
    (data_dir / "spoken_test-v1.1.json").write_text(json.dumps(payload))
    _write_silent_wav(data_dir / "wav" / "train" / "1_1_1.wav")

    train_ds, val_ds, test_ds, _ = load_speech_qa_dataset(
        dataset_name="local_spoken_squad",
        data_dir=data_dir,
        min_wavs_per_split=1,
        max_missing_audio_rate=0.0,
        allow_train_only_fallback=True,
        split_percentages={"train": 0.8, "validation": 0.1, "test": 0.1},
        max_total_samples=10,
        train_split="train",
        validation_split="validation",
        test_split="test",
        num_proc=1,
    )

    assert train_ds is not None
    assert val_ds is not None
    assert test_ds is not None
    assert len(train_ds) + len(val_ds) + len(test_ds) == 10


def test_local_spoken_squad_loader_supports_audio_concatenation_policy(tmp_path: Path) -> None:
    data_dir = tmp_path / "Spoken-SQuAD"
    (data_dir / "wav" / "train").mkdir(parents=True, exist_ok=True)
    (data_dir / "wav" / "test").mkdir(parents=True, exist_ok=True)

    payload = {
        "version": "1.1",
        "data": [
            {
                "title": "topic",
                "paragraphs": [
                    {
                        "context": "alpha beta gamma",
                        "qas": [
                            {"id": "q1", "question": "train question 1", "answers": [{"text": "beta", "answer_start": 6}]},
                            {"id": "q2", "question": "train question 2", "answers": [{"text": "beta", "answer_start": 6}]},
                            {"id": "q3", "question": "train question 3", "answers": [{"text": "beta", "answer_start": 6}]},
                            {"id": "q4", "question": "train question 4", "answers": [{"text": "beta", "answer_start": 6}]},
                        ],
                    }
                ],
            }
        ],
    }
    (data_dir / "spoken_train-v1.1.json").write_text(json.dumps(payload))
    (data_dir / "spoken_test-v1.1.json").write_text(json.dumps(payload))

    # Two sentence-level wavs for one paragraph prefix.
    _write_silent_wav(data_dir / "wav" / "train" / "1_1_1.wav", duration_ms=50)
    _write_silent_wav(data_dir / "wav" / "train" / "1_1_2.wav", duration_ms=50)
    _write_silent_wav(data_dir / "wav" / "test" / "1_1_1.wav", duration_ms=50)
    _write_silent_wav(data_dir / "wav" / "test" / "1_1_2.wav", duration_ms=50)

    train_ds, _, _, answers_map = load_speech_qa_dataset(
        dataset_name="local_spoken_squad",
        data_dir=data_dir,
        min_wavs_per_split=1,
        max_missing_audio_rate=0.0,
        split_percentages=None,
        train_split="train",
        validation_split=None,
        test_split="test",
        audio_merge_policy="concatenate_sentences",
        num_proc=1,
    )

    assert train_ds is not None
    assert train_ds[0]["duration"] >= 0.09
    metadata = answers_map.get("_metadata", {})
    assert metadata.get("audio_merge_policy") == "concatenate_sentences"
    train_metadata = metadata.get("train", {})
    assert train_metadata.get("multi_sentence_prefixes") == 1
