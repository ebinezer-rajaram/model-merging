"""Metric computation utilities for emotion recognition."""

from __future__ import annotations

from typing import Any, Dict, Sequence

from tasks.base.metrics import compute_classification_metrics


def compute_emotion_metrics(
    eval_pred: Any,
    *,
    processor,
    label_names: Sequence[str],
    store_predictions: bool = False,
) -> Dict[str, float]:
    """Compute accuracy and macro-F1 for generated emotion labels.

    Args:
        eval_pred: Tuple of (predictions, labels)
        processor: Model processor containing tokenizer
        label_names: List of emotion label names
        store_predictions: If True, store predictions and labels for confusion matrix

    Returns:
        Dictionary containing metrics and optionally prediction data
    """
    # Use the generic classification metrics with "emotion" as the pattern prefix
    return compute_classification_metrics(
        eval_pred,
        processor=processor,
        label_names=label_names,
        pattern_prefix="emotion",
        store_predictions=store_predictions,
    )


__all__ = ["compute_emotion_metrics"]
