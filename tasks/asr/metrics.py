"""Task-specific metric computation for ASR."""

from typing import Any, Dict

import numpy as np

from core import compute_wer_from_texts, decode_tokens


def compute_asr_metrics(eval_pred: Any, processor) -> Dict[str, float]:
    """Compute word error rate for ASR evaluation."""
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    if isinstance(labels, tuple):
        labels = labels[0]

    preds = np.array(preds)
    labels = np.array(labels)

    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id or 0

    preds = np.where((preds < 0) | ~np.isfinite(preds), pad_id, preds).astype(np.int64)
    labels = np.where((labels < 0) | ~np.isfinite(labels), pad_id, labels).astype(np.int64)

    pred_texts = [decode_tokens(p, processor.tokenizer) for p in preds]
    label_texts = [decode_tokens(l, processor.tokenizer) for l in labels]

    return {"wer": compute_wer_from_texts(label_texts, pred_texts)}
