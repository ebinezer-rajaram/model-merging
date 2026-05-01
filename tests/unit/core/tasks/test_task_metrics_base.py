from __future__ import annotations

import numpy as np
import pytest

from core.tasks.metrics import (
    _compute_macro_f1,
    _compute_weighted_f1,
    _decode_labels,
    _safe_decode_ids,
    compute_classification_metrics,
    match_label_in_text,
    post_process_chat_template_output,
)
from tests.helpers.tokenizer import DummyProcessor


def test_safe_decode_falls_back_to_token_conversion() -> None:
    class FallbackTokenizer:
        all_special_tokens = ["<pad>"]

        def decode(self, token_ids, skip_special_tokens=True):
            raise TypeError("decode failed")

        def convert_ids_to_tokens(self, token_ids, skip_special_tokens=True):
            return ["A" if int(token) == 1 else "<pad>" for token in token_ids]

        def convert_tokens_to_string(self, tokens):
            return "".join(tokens)

    assert _safe_decode_ids(FallbackTokenizer(), [1, None, 0]) == "A<pad>"
    assert _decode_labels(np.array([[1, -100]]), FallbackTokenizer()) == ["A"]


def test_label_matching_prefers_candidate_span_then_full_text() -> None:
    label_map = {"book flight": 0, "play music": 1}
    assert match_label_in_text("intent: play music.", label_map, pattern_prefix="intent") == 1
    assert match_label_in_text("assistant says book flight", label_map, pattern_prefix="intent") == 0
    assert match_label_in_text("", label_map) is None


def test_f1_helpers_treat_unknown_predictions_as_false_negatives() -> None:
    assert _compute_macro_f1([0, 1], [0, -1], 2) == pytest.approx(0.5)
    assert _compute_weighted_f1([0, 1], [0, -1], 2) == pytest.approx(0.5)
    assert _compute_macro_f1([], [], 0) == 0.0
    assert _compute_weighted_f1([-1], [0], 2) == 0.0


def test_compute_classification_metrics_tracks_unknown_predictions_and_storage() -> None:
    processor = DummyProcessor(
        {
            1: "intent: book flight",
            2: "unmapped",
            3: "book flight",
            4: "play music",
        }
    )
    metrics = compute_classification_metrics(
        (np.array([[1], [2]]), np.array([[3, -100], [4, -100]])),
        processor=processor,
        label_names=["book flight", "play music"],
        pattern_prefix="intent",
        store_predictions=True,
    )
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["recognized_rate"] == pytest.approx(0.5)
    assert metrics["num_samples"] == 2.0
    assert metrics["_predictions"] == [0, -1]
    assert metrics["_labels"] == [0, 1]


def test_post_process_chat_template_output_strips_assistant_prefix() -> None:
    assert post_process_chat_template_output(["user\nx\nassistant\nAnswer"]) == ["Answer"]
