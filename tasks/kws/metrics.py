"""Metric computation utilities for keyword spotting."""

from __future__ import annotations

from typing import Any, Dict, Sequence

from core.tasks.metrics import compute_classification_metrics


def compute_kws_metrics(
    eval_pred: Any,
    *,
    processor,
    label_names: Sequence[str],
    store_predictions: bool = False,
) -> Dict[str, float]:
    """Compute accuracy and macro-F1 for generated keyword labels.

    Args:
        eval_pred: Tuple of (predictions, labels)
        processor: Model processor containing tokenizer
        label_names: List of keyword label names
        store_predictions: If True, store predictions and labels for confusion matrix

    Returns:
        Dictionary containing metrics and optionally prediction data
    """
    # Use the generic classification metrics with "keyword" as the pattern prefix
    return compute_classification_metrics(
        eval_pred,
        processor=processor,
        label_names=label_names,
        pattern_prefix="keyword",
        store_predictions=store_predictions,
    )


__all__ = ["compute_kws_metrics"]
