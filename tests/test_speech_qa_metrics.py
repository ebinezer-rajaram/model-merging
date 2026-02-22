from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tasks.speech_qa.metrics import compute_speech_qa_metrics


class _DummyTokenizer:
    def __init__(self, token_map: dict[int, str]):
        self._token_map = token_map

    def decode(self, token_ids, skip_special_tokens: bool = True):
        return "".join(self._token_map.get(int(token_id), "") for token_id in token_ids)


class _DummyProcessor:
    def __init__(self, token_map: dict[int, str]):
        self.tokenizer = _DummyTokenizer(token_map)


class _FakeSquadMetric:
    def compute(self, *, predictions, references):
        em_scores = []
        f1_scores = []
        for pred, ref in zip(predictions, references):
            pred_text = str(pred["prediction_text"]).strip().lower()
            gold_answers = [str(x).strip().lower() for x in ref["answers"]["text"]]
            exact = 100.0 if pred_text in gold_answers else 0.0
            # Keep fake F1 simple for test determinism.
            f1 = exact
            em_scores.append(exact)
            f1_scores.append(f1)
        return {
            "exact_match": float(sum(em_scores) / max(1, len(em_scores))),
            "f1": float(sum(f1_scores) / max(1, len(f1_scores))),
        }


def test_speech_qa_metrics_uses_wrapper_stripping(monkeypatch) -> None:
    token_map = {
        1: "Answer: ",
        2: "C",
        3: "C",
    }
    processor = _DummyProcessor(token_map)
    monkeypatch.setattr("tasks.speech_qa.metrics._SQUAD_METRIC", _FakeSquadMetric())

    preds = np.array([[1, 2]])
    labels = np.array([[3, -100]])
    metrics = compute_speech_qa_metrics(
        (preds, labels),
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


def test_speech_qa_metrics_writes_audit_dump(monkeypatch, tmp_path: Path) -> None:
    token_map = {
        1: "The answer is ",
        2: "B",
        3: "C",
        4: "B",
        5: "C",
    }
    processor = _DummyProcessor(token_map)
    monkeypatch.setattr("tasks.speech_qa.metrics._SQUAD_METRIC", _FakeSquadMetric())

    preds = np.array([[1, 2], [3]])
    labels = np.array([[4, -100], [5, -100]])
    dump_path = tmp_path / "audit.jsonl"

    compute_speech_qa_metrics(
        (preds, labels),
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


def test_speech_qa_metrics_accepts_debug_dump_aliases(monkeypatch, tmp_path: Path) -> None:
    token_map = {
        1: "The answer is ",
        2: "B",
        3: "B",
    }
    processor = _DummyProcessor(token_map)
    monkeypatch.setattr("tasks.speech_qa.metrics._SQUAD_METRIC", _FakeSquadMetric())

    preds = np.array([[1, 2]])
    labels = np.array([[3, -100]])
    dump_path = tmp_path / "debug_audit.jsonl"

    compute_speech_qa_metrics(
        (preds, labels),
        processor=processor,
        reference_answers=[["beta"]],
        reference_choice_maps=[{"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"}],
        debug_eval_dump_path=dump_path,
        debug_eval_dump_samples=1,
        split_name="validation",
    )

    lines = dump_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["prediction_text"] == "beta"


def test_speech_qa_metrics_parses_letter_variants_and_tracks_recognized_rate(monkeypatch) -> None:
    token_map = {
        1: "C",
        2: "Answer: C",
        3: "The answer is C.",
        4: "unknown",
        5: "C",
    }
    processor = _DummyProcessor(token_map)
    monkeypatch.setattr("tasks.speech_qa.metrics._SQUAD_METRIC", _FakeSquadMetric())

    preds = np.array([[1], [2], [3], [4]])
    labels = np.array([[5, -100], [5, -100], [5, -100], [5, -100]])
    metrics = compute_speech_qa_metrics(
        (preds, labels),
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
