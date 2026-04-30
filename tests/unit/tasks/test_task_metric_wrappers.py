from __future__ import annotations

import numpy as np
import pytest

from tasks.emotion.metrics import compute_emotion_metrics
from tasks.intent.metrics import compute_intent_metrics
from tasks.kws.metrics import compute_kws_metrics
from tasks.langid.metrics import compute_langid_metrics
from tests.helpers.tokenizer import DummyProcessor


@pytest.mark.parametrize(
    ("metric_fn", "labels", "prefix"),
    [
        (compute_intent_metrics, ["book flight", "play music"], "intent"),
        (compute_emotion_metrics, ["happy", "sad"], "emotion"),
        (compute_kws_metrics, ["yes", "no"], "keyword"),
        (compute_langid_metrics, ["english", "german"], "language"),
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
