from __future__ import annotations

import csv
from types import SimpleNamespace

from core.training.trainer import sanitize_generation_kwargs, save_history_to_csv
from tests.helpers.core import FakeTrainerForCsv, FakeTrainerState


def test_sanitize_generation_kwargs_removes_sampling_args_for_greedy() -> None:
    sanitized = sanitize_generation_kwargs({"max_new_tokens": 32, "do_sample": False, "temperature": 0.0, "top_p": 0.95, "top_k": 40})
    assert sanitized["max_new_tokens"] == 32
    assert sanitized["do_sample"] is False
    assert "temperature" not in sanitized
    assert "top_p" not in sanitized
    assert "top_k" not in sanitized


def test_sanitize_generation_kwargs_keeps_sampling_args_when_sampling() -> None:
    sanitized = sanitize_generation_kwargs({"do_sample": True, "temperature": 0.7, "top_p": 0.9, "top_k": 50})
    assert sanitized["do_sample"] is True
    assert sanitized["temperature"] == 0.7
    assert sanitized["top_p"] == 0.9
    assert sanitized["top_k"] == 50


def test_save_history_to_csv_normalizes_rows_and_deduplicates(tmp_path) -> None:
    trainer = FakeTrainerForCsv(
        FakeTrainerState(
            [
                {"step": 1, "loss": 0.5, "eval_runtime": 3.0},
                {"step": 1, "loss": 0.5, "eval_runtime": 3.0},
                {"step": 2, "eval_accuracy": "0.9", "bad": "skip"},
            ]
        )
    )
    rows = save_history_to_csv(trainer, tmp_path / "history.csv", extra_rows=[{"step": 0, "train_loss": 0.7}])
    assert len(rows) == 3
    with (tmp_path / "history.csv").open("r", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["step"] == "0"
    assert "eval_runtime" not in csv_rows[0]


def test_save_history_to_csv_noops_without_metrics(tmp_path, capsys) -> None:
    rows = save_history_to_csv(SimpleNamespace(state=SimpleNamespace(log_history=[{"step": 1, "note": "skip"}])), tmp_path / "empty.csv")
    assert rows == []
    assert "No training metrics" in capsys.readouterr().out
