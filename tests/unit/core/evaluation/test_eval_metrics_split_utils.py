from __future__ import annotations

import csv

import pytest

from core.evaluation import plotting
from core.evaluation.metrics import compute_wer_from_texts, decode_tokens, sanitize_token_array
from core.evaluation.split_utils import (
    apply_task_split_overrides,
    asr_resolved_librispeech_split,
    canonical_output_split,
    normalize_split_name,
    task_data_split,
)


def test_token_metrics_sanitize_decode_and_wer_fallbacks() -> None:
    assert sanitize_token_array([1, -100, 3], pad_id=0) == [1, 0, 3]

    class FallbackTokenizer:
        pad_token_id = 0
        eos_token_id = 0
        unk_token_id = 99
        vocab_size = 10
        all_special_tokens = ["<pad>"]

        def decode(self, tokens, skip_special_tokens=True):
            raise TypeError("fallback")

        def convert_ids_to_tokens(self, tokens):
            return ["A" if int(tok) == 1 else "<pad>" for tok in tokens]

        def convert_tokens_to_string(self, tokens):
            return " ".join(tokens)

    assert decode_tokens([1, 999], FallbackTokenizer()) == "A"
    assert compute_wer_from_texts(["Hello, world"], ["hello world"], normalization="aggressive") == 0.0
    assert compute_wer_from_texts(["hello"], ["hello"], normalization="standardize") == 0.0
    assert compute_wer_from_texts(["hello"], ["hello"], normalization="default") == 0.0


def test_split_aliases_and_asr_overrides_do_not_mutate_input() -> None:
    assert normalize_split_name(" Test.Other ") == "test.other"
    assert canonical_output_split("test-other") == "test_other"
    assert canonical_output_split("test.clean") == "test"
    assert task_data_split("emotion", "test-other") == "test"
    assert asr_resolved_librispeech_split("test-other") == "test.other"
    assert asr_resolved_librispeech_split("test-clean") == "test"

    config = {"dataset": {"test_split": "test"}}
    updated, metadata = apply_task_split_overrides(task="asr", config=config, requested_split="test-other")
    assert updated["dataset"]["test_split"] == "test-other"
    assert config["dataset"]["test_split"] == "test"
    assert metadata["resolved_split"] == "test.other"
    assert metadata["output_split"] == "test_other"

    non_asr, non_asr_meta = apply_task_split_overrides(task="emotion", config={}, requested_split="validation")
    assert non_asr == {}
    assert non_asr_meta["data_split"] == "validation"


def test_plotting_collects_series_and_handles_missing_or_empty_inputs(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    missing_plot = tmp_path / "missing.png"
    plotting.plot_loss_and_wer(tmp_path / "missing.csv", missing_plot)
    assert "not found" in capsys.readouterr().out or plotting.plt is None

    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("step\n", encoding="utf-8")
    plotting.plot_loss_and_wer(empty_csv, tmp_path / "empty.png")
    assert "No metric columns" in capsys.readouterr().out or plotting.plt is None

    history_csv = tmp_path / "history.csv"
    with history_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "train_loss", "eval_wer", "learning_rate"])
        writer.writeheader()
        writer.writerow({"step": "1", "train_loss": "2.0", "eval_wer": "0.5", "learning_rate": "1e-4"})
        writer.writerow({"step": "1", "train_loss": "1.5", "eval_wer": "0.4", "learning_rate": "1e-4"})
        writer.writerow({"step": "bad", "train_loss": "", "eval_wer": "none", "learning_rate": "1e-4"})

    steps, series = plotting._collect_series(history_csv)
    assert steps == [1, 1, None]
    assert series["train_loss"] == [2.0, 1.5, None]
    assert series["eval_wer"] == [0.5, 0.4, None]

    out_plot = tmp_path / "history.png"
    plotting.plot_loss_and_wer(history_csv, out_plot)
    if plotting.plt is not None:
        assert out_plot.exists()


def test_plot_confusion_matrix_filters_invalid_predictions(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    empty_plot = tmp_path / "empty_cm.png"
    plotting.plot_confusion_matrix([-1], [-1], ["a"], empty_plot)
    assert "No valid predictions" in capsys.readouterr().out or plotting.plt is None

    out_plot = tmp_path / "cm.png"
    plotting.plot_confusion_matrix([0, 1, -1, 1], [0, 0, -1, 1], ["no", "yes"], out_plot, normalize=False)
    if plotting.plt is not None:
        assert out_plot.exists()
