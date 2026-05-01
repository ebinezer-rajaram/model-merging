from __future__ import annotations

import csv
import json
from pathlib import Path

from datasets import Audio, Dataset, DatasetDict
import pytest

import tasks.emotion.dataset as emotion_dataset
import tasks.intent.dataset as intent_dataset
import tasks.kws.dataset as kws_dataset
import tasks.langid.dataset as langid_dataset
import tasks.speaker_id.dataset as speaker_id_dataset
import tasks.speaker_ver.dataset as speaker_ver_dataset
from tasks.speech_qa.dataset import load_speech_qa_dataset
from tasks.vocalsound.dataset import load_vocalsound_dataset
from tests.helpers.audio import write_silent_wav


LABEL_NAMES = ["cough", "sneeze", "throat clearing", "sniff", "laugh", "exhale"]


def _audio(length: int = 4) -> dict:
    return {"array": [0.0] * length, "sampling_rate": 16000}


def _add_duration(dataset: DatasetDict, **kwargs) -> DatasetDict:
    del kwargs
    return DatasetDict(
        {
            split: subset if "duration" in subset.column_names else subset.add_column("duration", [0.1] * len(subset))
            for split, subset in dataset.items()
        }
    )


def _filter_by_duration(dataset: DatasetDict, **kwargs) -> DatasetDict:
    max_duration = kwargs.get("max_duration")
    min_duration = kwargs.get("min_duration")
    if max_duration is None and min_duration is None:
        return dataset
    filtered = {}
    for split, subset in dataset.items():
        indices = [
            idx
            for idx, duration in enumerate(subset["duration"])
            if (max_duration is None or duration <= max_duration) and (min_duration is None or duration >= min_duration)
        ]
        filtered[split] = subset.select(indices)
    return DatasetDict(filtered)


def _cache_and_sample_splits(dataset: DatasetDict, **kwargs):
    max_train = kwargs.get("max_train_samples")
    max_val = kwargs.get("max_validation_samples")
    max_test = kwargs.get("max_test_samples")

    def _select(name: str, limit):
        if name not in dataset:
            return None
        subset = dataset[name]
        if limit is not None:
            subset = subset.select(range(min(int(limit), len(subset))))
        return subset

    train = _select("train", max_train)
    validation = _select("validation", max_val)
    test = _select("test", max_test)
    metadata = {
        name: {"hours": 0.0}
        for name, subset in (("train", train), ("validation", validation), ("test", test))
        if subset is not None
    }
    return train, validation, test, metadata, kwargs.get("seed", 0)


def _patch_common_loader_steps(monkeypatch: pytest.MonkeyPatch, module) -> None:
    monkeypatch.setattr(module, "add_duration_to_dataset", _add_duration)
    monkeypatch.setattr(module, "filter_by_duration", _filter_by_duration)
    monkeypatch.setattr(module, "cache_and_sample_splits", _cache_and_sample_splits)


def _standard_classification_splits() -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_list(
                [
                    {"audio": _audio(), "label": "alpha", "text": "train alpha"},
                    {"audio": _audio(), "label": "beta", "text": "train beta"},
                ]
            ),
            "validation": Dataset.from_list([{"audio": _audio(), "label": "alpha", "text": "val alpha"}]),
            "test": Dataset.from_list([{"audio": _audio(), "label": "beta", "text": "test beta"}]),
        }
    )


def test_emotion_loader_uses_shared_classification_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = _standard_classification_splits()
    monkeypatch.setattr(emotion_dataset, "load_and_prepare_dataset", lambda **kwargs: (dataset, "audio"))
    _patch_common_loader_steps(monkeypatch, emotion_dataset)

    train_ds, val_ds, test_ds, labels = emotion_dataset.load_superb_emotion_dataset(
        max_train_samples=1,
        cache_dir=tmp_path,
        cache_splits=False,
        num_proc=1,
    )

    assert train_ds is not None and len(train_ds) == 1
    assert val_ds is not None and test_ds is not None
    assert labels == ["alpha", "beta"]


def test_intent_loader_filters_corrupted_audio_and_keeps_label_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = DatasetDict(
        {
            "train": Dataset.from_list(
                [
                    {"audio": _audio(), "label": "turn_on", "text": "turn it on"},
                    {"audio": None, "label": "turn_off", "text": "turn it off"},
                ]
            ),
            "validation": Dataset.from_list([{"audio": _audio(), "label": "turn_on", "text": "val"}]),
            "test": Dataset.from_list([{"audio": _audio(), "label": "turn_off", "text": "test"}]),
        }
    )
    monkeypatch.setattr(intent_dataset, "load_and_prepare_dataset", lambda **kwargs: (dataset, "audio"))
    _patch_common_loader_steps(monkeypatch, intent_dataset)

    train_ds, _, _, labels = intent_dataset.load_slurp_intent_dataset(
        cache_dir=tmp_path,
        cache_splits=False,
        num_proc=1,
    )

    assert train_ds is not None
    assert len(train_ds) == 1
    assert labels == ["turn_on"]


def test_kws_loader_filters_silence_and_applies_sample_caps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = DatasetDict(
        {
            "train": Dataset.from_list(
                [
                    {"audio": _audio(), "label": 0},
                    {"audio": _audio(), "label": 35},
                    {"audio": _audio(), "label": 1},
                ]
            ),
            "validation": Dataset.from_list([{"audio": _audio(), "label": 0}]),
            "test": Dataset.from_list([{"audio": _audio(), "label": 1}]),
        }
    )
    monkeypatch.setattr(kws_dataset, "load_dataset", lambda *args, **kwargs: dataset)
    _patch_common_loader_steps(monkeypatch, kws_dataset)

    train_ds, _, _, labels = kws_dataset.load_speech_commands_kws_dataset(
        max_train_samples=1,
        cache_dir=tmp_path,
        cache_splits=False,
        num_proc=1,
    )

    assert train_ds is not None
    assert len(train_ds) == 1
    assert 35 not in train_ds["label"]
    assert labels == kws_dataset.KNOWN_COMMANDS


def test_langid_loader_uses_selected_language_splits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = DatasetDict(
        {
            "train": Dataset.from_list(
                [
                    {"audio": _audio(), "label": "english", "lang": "en"},
                    {"audio": _audio(), "label": "german", "lang": "de"},
                ]
            ),
            "validation": Dataset.from_list([{"audio": _audio(), "label": "english", "lang": "en"}]),
            "test": Dataset.from_list([{"audio": _audio(), "label": "german", "lang": "de"}]),
        }
    )
    monkeypatch.setattr(langid_dataset, "_load_fleurs_languages", lambda **kwargs: dataset)
    _patch_common_loader_steps(monkeypatch, langid_dataset)

    train_ds, val_ds, test_ds, labels = langid_dataset.load_fleurs_langid_dataset(
        languages=["en", "de"],
        max_validation_samples=1,
        cache_dir=tmp_path,
        cache_splits=False,
        num_proc=1,
    )

    assert train_ds is not None and len(train_ds) == 2
    assert val_ds is not None and len(val_ds) == 1
    assert test_ds is not None and len(test_ds) == 1
    assert labels == langid_dataset.LANGUAGE_NAMES


def test_speaker_id_loader_selects_speakers_before_sampling(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = DatasetDict(
        {
            "train": Dataset.from_list(
                [
                    {"audio": _audio(), "label": "spk_a", "text": "a1"},
                    {"audio": _audio(), "label": "spk_a", "text": "a2"},
                    {"audio": _audio(), "label": "spk_b", "text": "b1"},
                    {"audio": _audio(), "label": "spk_c", "text": "c1"},
                ]
            ),
            "validation": Dataset.from_list(
                [
                    {"audio": _audio(), "label": "spk_a", "text": "a val"},
                    {"audio": _audio(), "label": "spk_c", "text": "c val"},
                ]
            ),
            "test": Dataset.from_list([{"audio": _audio(), "label": "spk_b", "text": "b test"}]),
        }
    )
    monkeypatch.setattr(speaker_id_dataset, "load_and_prepare_dataset", lambda **kwargs: (dataset, "audio"))
    _patch_common_loader_steps(monkeypatch, speaker_id_dataset)

    train_ds, val_ds, test_ds, labels = speaker_id_dataset.load_voxceleb_speaker_dataset(
        max_speakers=2,
        max_samples_per_speaker=1,
        cache_dir=tmp_path,
        cache_splits=False,
        num_proc=1,
    )

    assert labels == ["spk_a", "spk_b"]
    assert train_ds is not None and sorted(train_ds["label"]) == ["spk_a", "spk_b"]
    assert val_ds is not None and val_ds["label"] == ["spk_a"]
    assert test_ds is not None and test_ds["label"] == ["spk_b"]


def test_speaker_ver_loader_reuses_cached_pairs_without_loading_base_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pair_splits = {
        "train": Dataset.from_list(
            [
                {"audio_a": _audio(), "audio_b": _audio(), "label": 1, "speaker_id": "same"},
                {"audio_a": _audio(), "audio_b": _audio(), "label": 0, "speaker_id": "diff"},
            ]
        ),
        "validation": Dataset.from_list([{"audio_a": _audio(), "audio_b": _audio(), "label": 1, "speaker_id": "val"}]),
        "test": Dataset.from_list([{"audio_a": _audio(), "audio_b": _audio(), "label": 0, "speaker_id": "test"}]),
    }

    def _load_cached_pair(_cache_dir: Path, split_name: str):
        return pair_splits.get(split_name)

    monkeypatch.setattr(speaker_ver_dataset, "_find_cached_pairs_dir", lambda *args, **kwargs: tmp_path / "pairs")
    monkeypatch.setattr(speaker_ver_dataset, "_load_cached_pairs", _load_cached_pair)
    monkeypatch.setattr(
        speaker_ver_dataset,
        "load_and_prepare_dataset",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("base dataset should not be loaded")),
    )
    monkeypatch.setattr(speaker_ver_dataset, "cache_and_sample_splits", _cache_and_sample_splits)

    train_ds, val_ds, test_ds, labels = speaker_ver_dataset.load_speaker_ver_dataset(
        cache_dir=tmp_path,
        cache_splits=True,
        force_rebuild=False,
        max_train_samples=1,
        num_proc=1,
    )

    assert labels == ["no", "yes"]
    assert train_ds is not None and len(train_ds) == 1
    assert val_ds is not None and len(val_ds) == 1
    assert test_ds is not None and len(test_ds) == 1


def _build_mmsu_dataset(tmp_path: Path) -> DatasetDict:
    wav_path = tmp_path / "audio" / "sample.wav"
    write_silent_wav(wav_path, duration_ms=80)
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
        {split: subset.add_column("duration", [0.08] * len(subset)) for split, subset in dataset.items()}
    )


def test_speech_qa_mmsu_loader_uses_answer_gt_and_eval_first_split(monkeypatch, tmp_path: Path) -> None:
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


def test_speech_qa_mmsu_loader_injects_choice_context_when_enabled(monkeypatch, tmp_path: Path) -> None:
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
    assert test_ds[0]["context"] == "A. Paris\nB. London\nC. Berlin\nD. Madrid"


def test_speech_qa_mmsu_loader_requires_task_name_for_subtask_analysis(monkeypatch, tmp_path: Path) -> None:
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


def test_speech_qa_local_spoken_squad_loader_builds_train_and_test_splits(tmp_path: Path) -> None:
    data_dir = tmp_path / "Spoken-SQuAD"
    (data_dir / "wav" / "train").mkdir(parents=True, exist_ok=True)
    (data_dir / "wav" / "test").mkdir(parents=True, exist_ok=True)

    (data_dir / "spoken_train-v1.1.json").write_text(
        json.dumps(_build_spoken_squad_payload("train_q1", "train question", "beta"))
    )
    (data_dir / "spoken_test-v1.1.json").write_text(
        json.dumps(_build_spoken_squad_payload("test_q1", "test question", "beta"))
    )

    write_silent_wav(data_dir / "wav" / "train" / "1_1_1.wav")
    write_silent_wav(data_dir / "wav" / "test" / "1_1_1.wav")

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


def test_speech_qa_local_spoken_squad_loader_uses_noisy_test_variant(tmp_path: Path) -> None:
    data_dir = tmp_path / "Spoken-SQuAD"
    (data_dir / "wav" / "train").mkdir(parents=True, exist_ok=True)
    (data_dir / "wav" / "test").mkdir(parents=True, exist_ok=True)

    (data_dir / "spoken_train-v1.1.json").write_text(
        json.dumps(_build_spoken_squad_payload("train_q1", "train question", "beta"))
    )
    (data_dir / "spoken_test-v1.1_WER44.json").write_text(
        json.dumps(_build_spoken_squad_payload("test_q_wer44", "test noisy", "beta"))
    )

    write_silent_wav(data_dir / "wav" / "train" / "1_1_1.wav")
    write_silent_wav(data_dir / "wav" / "test" / "1_1_1.wav")

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


def test_speech_qa_local_spoken_squad_loader_fails_when_missing_audio_rate_exceeds_threshold(tmp_path: Path) -> None:
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
                        "qas": [{"id": "q1", "question": "q1?", "answers": [{"text": "beta", "answer_start": 6}]}],
                    },
                    {
                        "context": "delta epsilon zeta",
                        "qas": [
                            {"id": "q2", "question": "q2?", "answers": [{"text": "epsilon", "answer_start": 6}]}
                        ],
                    },
                ],
            }
        ],
    }
    (data_dir / "spoken_train-v1.1.json").write_text(json.dumps(payload))
    (data_dir / "spoken_test-v1.1.json").write_text(json.dumps(payload))
    write_silent_wav(data_dir / "wav" / "train" / "1_1_1.wav")
    write_silent_wav(data_dir / "wav" / "test" / "1_1_1.wav")

    with pytest.raises(ValueError, match="missing audio mappings"):
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


def test_speech_qa_local_spoken_squad_loader_can_fallback_to_train_only_source(tmp_path: Path) -> None:
    data_dir = tmp_path / "Spoken-SQuAD"
    (data_dir / "wav" / "train").mkdir(parents=True, exist_ok=True)
    (data_dir / "wav" / "test").mkdir(parents=True, exist_ok=True)

    qas = [{"id": f"q{i}", "question": f"question {i}?", "answers": [{"text": "beta", "answer_start": 6}]} for i in range(10)]
    payload = {
        "version": "1.1",
        "data": [{"title": "topic", "paragraphs": [{"context": "alpha beta gamma", "qas": qas}]}],
    }
    (data_dir / "spoken_train-v1.1.json").write_text(json.dumps(payload))
    (data_dir / "spoken_test-v1.1.json").write_text(json.dumps(payload))
    write_silent_wav(data_dir / "wav" / "train" / "1_1_1.wav")

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
    assert answers_map.get("_metadata", {}).get("split_origin") == "train_only_fallback"


def test_speech_qa_local_spoken_squad_loader_supports_max_total_samples(tmp_path: Path) -> None:
    data_dir = tmp_path / "Spoken-SQuAD"
    (data_dir / "wav" / "train").mkdir(parents=True, exist_ok=True)
    (data_dir / "wav" / "test").mkdir(parents=True, exist_ok=True)

    qas = [{"id": f"q{i}", "question": f"question {i}?", "answers": [{"text": "beta", "answer_start": 6}]} for i in range(20)]
    payload = {
        "version": "1.1",
        "data": [{"title": "topic", "paragraphs": [{"context": "alpha beta gamma", "qas": qas}]}],
    }
    (data_dir / "spoken_train-v1.1.json").write_text(json.dumps(payload))
    (data_dir / "spoken_test-v1.1.json").write_text(json.dumps(payload))
    write_silent_wav(data_dir / "wav" / "train" / "1_1_1.wav")

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


def test_speech_qa_local_spoken_squad_loader_supports_audio_concatenation_policy(tmp_path: Path) -> None:
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

    write_silent_wav(data_dir / "wav" / "train" / "1_1_1.wav", duration_ms=50)
    write_silent_wav(data_dir / "wav" / "train" / "1_1_2.wav", duration_ms=50)
    write_silent_wav(data_dir / "wav" / "test" / "1_1_1.wav", duration_ms=50)
    write_silent_wav(data_dir / "wav" / "test" / "1_1_2.wav", duration_ms=50)

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
    assert metadata.get("train", {}).get("multi_sentence_prefixes") == 1


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
    write_silent_wav(audio_root / "train_short.wav", duration_ms=30)
    write_silent_wav(audio_root / "train_long.wav", duration_ms=240)
    write_silent_wav(audio_root / "val.wav", duration_ms=40)
    write_silent_wav(audio_root / "test.wav", duration_ms=50)
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
