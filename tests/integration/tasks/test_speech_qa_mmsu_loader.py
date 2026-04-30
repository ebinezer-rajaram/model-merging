from __future__ import annotations

from pathlib import Path
import wave

from datasets import Audio, Dataset, DatasetDict
import pytest

from tasks.speech_qa.dataset import SpeechQACollator, load_speech_qa_dataset


def _write_silent_wav(path: Path, *, sample_rate: int = 16000, duration_ms: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    num_frames = int(sample_rate * (duration_ms / 1000.0))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * num_frames)


def _build_mmsu_dataset(tmp_path: Path) -> DatasetDict:
    wav_path = tmp_path / "audio" / "sample.wav"
    _write_silent_wav(wav_path)
    train = Dataset.from_list(
        [
            {
                "id": "mmsu_1",
                "question": "What is the capital of France?",
                "answer_gt": "Paris",
                "task_name": "geography",
                "choice_a": "Paris",
                "choice_b": "London",
                "choice_c": "Berlin",
                "choice_d": "Madrid",
                "audio": str(wav_path),
            }
        ]
    ).cast_column("audio", Audio(sampling_rate=16000))
    return DatasetDict({"train": train})


def _add_constant_duration(dataset: DatasetDict, **kwargs) -> DatasetDict:
    del kwargs
    return DatasetDict(
        {
            split: subset.add_column("duration", [0.08] * len(subset))
            for split, subset in dataset.items()
        }
    )


def test_mmsu_loader_uses_answer_gt_and_eval_first_split(monkeypatch, tmp_path: Path) -> None:
    dataset_dict = _build_mmsu_dataset(tmp_path)
    monkeypatch.setattr("tasks.speech_qa.dataset.load_dataset", lambda *args, **kwargs: dataset_dict)
    monkeypatch.setattr("tasks.speech_qa.dataset.add_duration_to_dataset", _add_constant_duration)

    train_ds, val_ds, test_ds, answers_map = load_speech_qa_dataset(
        dataset_name="ddwang2000/MMSU",
        validation_split=None,
        test_split="train",
        split_percentages=None,
        cache_splits=False,
        num_proc=1,
    )

    assert train_ds is not None
    assert val_ds is None
    assert test_ds is not None
    assert test_ds[0]["label_text"] == "A"
    assert test_ds[0]["answer_letter"] == "A"
    assert test_ds[0]["answers"] == ["Paris"]
    assert answers_map["test"][0] == ["Paris"]


def test_mmsu_loader_injects_choice_context_when_enabled(monkeypatch, tmp_path: Path) -> None:
    dataset_dict = _build_mmsu_dataset(tmp_path)
    monkeypatch.setattr("tasks.speech_qa.dataset.load_dataset", lambda *args, **kwargs: dataset_dict)
    monkeypatch.setattr("tasks.speech_qa.dataset.add_duration_to_dataset", _add_constant_duration)

    _, _, test_ds, _ = load_speech_qa_dataset(
        dataset_name="ddwang2000/MMSU",
        validation_split=None,
        test_split="train",
        split_percentages=None,
        include_choices_in_prompt=True,
        cache_splits=False,
        num_proc=1,
    )

    assert test_ds is not None
    expected = "A. Paris\nB. London\nC. Berlin\nD. Madrid"
    assert test_ds[0]["context"] == expected


def test_speech_qa_collator_choice_toggle() -> None:
    feature = {
        "question": "What is the capital of France?",
        "choice_a": "Paris",
        "choice_b": "London",
        "choice_c": "Berlin",
        "choice_d": "Madrid",
    }
    question_only = SpeechQACollator(
        processor=None,
        sampling_rate=16000,
        include_choices_in_prompt=False,
    )._build_instruction(feature)
    with_choices = SpeechQACollator(
        processor=None,
        sampling_rate=16000,
        include_choices_in_prompt=True,
    )._build_instruction(feature)

    assert "Question: What is the capital of France?" in question_only
    assert "Choices:" not in question_only
    assert "Choices:" in with_choices
    assert "A. Paris" in with_choices
    assert "D. Madrid" in with_choices
    assert "Respond with exactly one uppercase letter: A, B, C, or D." in with_choices
    assert "Do not output any other text." in with_choices


def test_mmsu_loader_requires_task_name_for_subtask_analysis(monkeypatch, tmp_path: Path) -> None:
    dataset_dict = _build_mmsu_dataset(tmp_path)
    train_no_subtask = dataset_dict["train"].remove_columns(["task_name"])
    monkeypatch.setattr(
        "tasks.speech_qa.dataset.load_dataset",
        lambda *args, **kwargs: DatasetDict({"train": train_no_subtask}),
    )
    monkeypatch.setattr("tasks.speech_qa.dataset.add_duration_to_dataset", _add_constant_duration)

    with pytest.raises(ValueError) as exc_info:
        load_speech_qa_dataset(
            dataset_name="ddwang2000/MMSU",
            validation_split=None,
            test_split="train",
            split_percentages=None,
            cache_splits=False,
            num_proc=1,
        )
    assert "task_name" in str(exc_info.value)
