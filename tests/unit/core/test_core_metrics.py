from __future__ import annotations

import numpy as np
import pytest

from core.tasks.metrics import (
    compute_classification_metrics,
    match_label_in_text,
    post_process_chat_template_output,
)
from tests.helpers.tokenizer import DummyProcessor


def test_match_label_prefers_candidate_span_then_full_text() -> None:
    label_map = {"book flight": 0, "play music": 1}

    assert match_label_in_text("intent: play music.", label_map, pattern_prefix="intent") == 1
    assert match_label_in_text("assistant says book flight", label_map, pattern_prefix="intent") == 0
    assert match_label_in_text("", label_map) is None


def test_compute_classification_metrics_tracks_unknown_predictions() -> None:
    processor = DummyProcessor(
        {
            1: "intent: book flight",
            2: "unmapped",
            3: "book flight",
            4: "play music",
        }
    )
    preds = np.array([[1], [2]])
    labels = np.array([[3, -100], [4, -100]])

    metrics = compute_classification_metrics(
        (preds, labels),
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
