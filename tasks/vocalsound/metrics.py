"""Metric computation utilities for VocalSound classification."""

from __future__ import annotations

from typing import Any, Dict, Sequence

from core.tasks.metrics import compute_classification_metrics


def compute_vocalsound_metrics(
    eval_pred: Any,
    *,
    processor,
    label_names: Sequence[str],
    store_predictions: bool = False,
) -> Dict[str, float]:
    """Compute accuracy and F1 metrics for VocalSound labels."""
    return compute_classification_metrics(
        eval_pred,
        processor=processor,
        label_names=label_names,
        pattern_prefix="sound",
        store_predictions=store_predictions,
    )


__all__ = ["compute_vocalsound_metrics"]
