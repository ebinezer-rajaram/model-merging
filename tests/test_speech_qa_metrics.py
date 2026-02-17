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
        2: "Denver",
        3: "Denver",
    }
    processor = _DummyProcessor(token_map)
    monkeypatch.setattr("tasks.speech_qa.metrics._SQUAD_METRIC", _FakeSquadMetric())

    preds = np.array([[1, 2]])
    labels = np.array([[3, -100]])
    metrics = compute_speech_qa_metrics(
        (preds, labels),
        processor=processor,
        reference_answers=[["Denver"]],
    )

    assert metrics["exact_match"] == 100.0
    assert metrics["f1"] == 100.0
    assert metrics["num_samples"] == 1.0


def test_speech_qa_metrics_writes_audit_dump(monkeypatch, tmp_path: Path) -> None:
    token_map = {
        1: "The answer is ",
        2: "beta",
        3: "gamma",
        4: "beta",
        5: "gamma",
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
    assert "best_f1" in first
    assert "exact_match" in first
