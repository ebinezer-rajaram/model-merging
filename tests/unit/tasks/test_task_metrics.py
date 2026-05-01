from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tasks.asr.metrics import compute_asr_metrics
from tasks.emotion.metrics import compute_emotion_metrics
from tasks.intent.metrics import compute_intent_metrics
from tasks.kws.metrics import compute_kws_metrics
from tasks.langid.metrics import compute_langid_metrics
from tasks.speaker_id.metrics import compute_speaker_id_metrics
from tasks.speaker_ver.metrics import compute_speaker_ver_metrics
from tasks.speech_qa.metrics import compute_speech_qa_metrics
from tasks.st.metrics import compute_st_metrics
from tasks.vocalsound.metrics import compute_vocalsound_metrics
from tests.helpers.speech_qa import FakeSquadMetric
from tests.helpers.tokenizer import DummyProcessor


@pytest.mark.parametrize(
    ("metric_fn", "labels", "prefix"),
    [
        (compute_intent_metrics, ["book flight", "play music"], "intent"),
        (compute_emotion_metrics, ["happy", "sad"], "emotion"),
        (compute_kws_metrics, ["yes", "no"], "keyword"),
        (compute_langid_metrics, ["english", "german"], "language"),
        (compute_speaker_id_metrics, ["1001", "1002"], "speaker"),
    ],
)
def test_classification_task_metric_wrappers(metric_fn, labels, prefix: str) -> None:
    processor = DummyProcessor({1: f"{prefix}: {labels[1]}", 2: labels[1]})

    metrics = metric_fn(
        (np.array([[1]]), np.array([[2, -100]])),
        processor=processor,
        label_names=labels,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["recognized_rate"] == 1.0
    assert metrics["num_samples"] == 1.0


def test_classification_metric_wrapper_tracks_unrecognized_outputs() -> None:
    processor = DummyProcessor(
        {
            1: "intent: book flight",
            2: "not a valid intent",
            3: "book flight",
            4: "play music",
        }
    )

    metrics = compute_intent_metrics(
        (np.array([[1], [2]]), np.array([[3, -100], [4, -100]])),
        processor=processor,
        label_names=["book flight", "play music"],
        store_predictions=True,
    )

    assert metrics["accuracy"] == 0.5
    assert metrics["recognized_rate"] == 0.5
    assert metrics["_predictions"] == [0, -1]
    assert metrics["_labels"] == [0, 1]


def test_asr_metrics_strip_assistant_prefix_and_support_multiple_normalizations() -> None:
    processor = DummyProcessor(
        {
            1: "assistant\nhello world",
            2: "assistant\nhello, world!",
            3: "assistant\nhello world",
        }
    )

    metrics = compute_asr_metrics(
        (np.array([[1], [2]]), np.array([[3], [3]])),
        processor=processor,
        wer_normalization="all",
    )

    assert metrics["wer_default"] > 0.0
    assert metrics["wer_aggressive"] == 0.0
    assert set(metrics) == {"wer_default", "wer_standardize", "wer_aggressive"}


def test_st_metrics_return_bleu_and_chrf_for_exact_translation() -> None:
    text = "assistant\nThis is a sufficiently long exact speech translation output"
    processor = DummyProcessor({1: text, 2: text})

    metrics = compute_st_metrics(
        (np.array([[1]]), np.array([[2]])),
        processor=processor,
        target_lang="en",
    )

    assert metrics["bleu"] > 99.0
    assert metrics["chrf"] > 99.0


def test_speaker_ver_wrapper_direct_label_parsing_and_prediction_storage() -> None:
    processor = DummyProcessor({1: "yes", 2: "no", 3: "yes", 4: "yes"})

    metrics = compute_speaker_ver_metrics(
        (np.array([[1], [2]]), np.array([[3, -100], [4, -100]])),
        processor=processor,
        label_names=["no", "yes"],
        store_predictions=True,
    )

    assert metrics["accuracy"] == 0.5
    assert metrics["recognized_rate"] == 1.0
    assert metrics["_predictions"] == [1, 0]
    assert metrics["_labels"] == [1, 1]


def test_vocalsound_metrics_returns_classification_scores_and_predictions() -> None:
    processor = DummyProcessor(
        {
            11: "assistant\ncough",
            12: "assistant\nsneeze",
            21: "assistant\ncough",
            22: "assistant\nsneeze",
        }
    )
    preds = np.array([[11], [12]])
    labels = np.array([[21, -100], [22, -100]])

    metrics = compute_vocalsound_metrics(
        (preds, labels),
        processor=processor,
        label_names=["cough", "sneeze"],
        store_predictions=True,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert metrics["recognized_rate"] == 1.0
    assert metrics["num_samples"] == 2.0
    assert "_predictions" in metrics
    assert "_labels" in metrics


def test_speech_qa_metrics_uses_wrapper_stripping(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = DummyProcessor({1: "Answer: ", 2: "C", 3: "C"})
    monkeypatch.setattr("tasks.speech_qa.metrics._SQUAD_METRIC", FakeSquadMetric())

    metrics = compute_speech_qa_metrics(
        (np.array([[1, 2]]), np.array([[3, -100]])),
        processor=processor,
        reference_answers=[["Denver"]],
        reference_choice_maps=[{"A": "Austin", "B": "Boston", "C": "Denver", "D": "Seattle"}],
        reference_subtasks=["geo"],
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["recognized_rate"] == 1.0
    assert metrics["exact_match"] == 100.0
    assert metrics["f1"] == 100.0
    assert metrics["num_samples"] == 1.0
    assert metrics["subtask_accuracy"]["geo"] == 1.0
    assert metrics["subtask_num_samples"]["geo"] == 1


def test_speech_qa_metrics_writes_audit_dump(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    processor = DummyProcessor(
        {
            1: "The answer is ",
            2: "B",
            3: "C",
            4: "B",
            5: "C",
        }
    )
    monkeypatch.setattr("tasks.speech_qa.metrics._SQUAD_METRIC", FakeSquadMetric())
    dump_path = tmp_path / "audit.jsonl"

    compute_speech_qa_metrics(
        (np.array([[1, 2], [3, 0]]), np.array([[4, -100], [5, -100]])),
        processor=processor,
        reference_answers=[["beta"], ["gamma"]],
        reference_ids=["id_1", "id_2"],
        reference_questions=["q1?", "q2?"],
        reference_choice_maps=[
            {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
            {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
        ],
        reference_subtasks=["t1", "t2"],
        audit_dump_path=dump_path,
        audit_samples=2,
        split_name="validation",
    )

    lines = dump_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["split"] == "validation"
    assert first["id"] == "id_1"
    assert first["question"] == "q1?"
    assert first["prediction_text"] == "beta"
    assert first["gold_answers"] == ["beta"]
    assert first["predicted_letter"] == "B"
    assert first["gold_letter"] == "B"
    assert first["task_name"] == "t1"
    assert "best_f1" in first
    assert "exact_match" in first


def test_speech_qa_metrics_accepts_debug_dump_aliases(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    processor = DummyProcessor({1: "The answer is ", 2: "B", 3: "B"})
    monkeypatch.setattr("tasks.speech_qa.metrics._SQUAD_METRIC", FakeSquadMetric())
    dump_path = tmp_path / "debug_audit.jsonl"

    compute_speech_qa_metrics(
        (np.array([[1, 2]]), np.array([[3, -100]])),
        processor=processor,
        reference_answers=[["beta"]],
        reference_choice_maps=[{"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"}],
        debug_eval_dump_path=dump_path,
        debug_eval_dump_samples=1,
        split_name="validation",
    )

    lines = dump_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["prediction_text"] == "beta"


def test_speech_qa_metrics_parses_letter_variants_and_tracks_recognized_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = DummyProcessor(
        {
            1: "C",
            2: "Answer: C",
            3: "The answer is C.",
            4: "unknown",
            5: "C",
        }
    )
    monkeypatch.setattr("tasks.speech_qa.metrics._SQUAD_METRIC", FakeSquadMetric())

    metrics = compute_speech_qa_metrics(
        (np.array([[1], [2], [3], [4]]), np.array([[5, -100], [5, -100], [5, -100], [5, -100]])),
        processor=processor,
        reference_answers=[["gamma"], ["gamma"], ["gamma"], ["gamma"]],
        reference_choice_maps=[
            {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
            {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
            {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
            {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
        ],
        reference_subtasks=["s1", "s1", "s2", "s2"],
    )

    assert metrics["accuracy"] == 0.75
    assert metrics["recognized_rate"] == 0.75
    assert metrics["exact_match"] == 75.0
    assert metrics["f1"] == 75.0
    assert metrics["subtask_accuracy"]["s1"] == 1.0
    assert metrics["subtask_accuracy"]["s2"] == 0.5
    assert metrics["subtask_num_samples"]["s1"] == 2
    assert metrics["subtask_num_samples"]["s2"] == 2
