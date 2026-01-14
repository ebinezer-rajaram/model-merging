"""Metrics for speaker verification task."""

from functools import partial

from core.tasks.metrics import compute_classification_metrics


def compute_speaker_ver_metrics(
    eval_pred, *, processor, label_names, store_predictions=False
):
    """Compute binary classification metrics for speaker verification.

    Args:
        eval_pred: EvalPrediction with predictions and label_ids
        processor: Model processor for decoding
        label_names: List of label names (["no", "yes"])
        store_predictions: Whether to save predictions to file

    Returns:
        Dict with accuracy, macro_f1, weighted_f1, recognized_rate
    """
    return compute_classification_metrics(
        eval_pred,
        processor=processor,
        label_names=label_names,  # ["no", "yes"]
        pattern_prefix=None,  # Direct parsing, no prefix like "speaker:"
        store_predictions=store_predictions,
    )
