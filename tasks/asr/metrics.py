"""Task-specific metric computation for ASR."""

from typing import Any, Dict, Literal

import numpy as np

from core import compute_wer_from_texts


def compute_asr_metrics(
    eval_pred: Any,
    processor,
    wer_normalization: Literal["default", "standardize", "both"] = "default",
) -> Dict[str, float]:
    """Compute word error rate for ASR evaluation.

    Args:
        eval_pred: Evaluation predictions (preds, labels) from trainer
        processor: Model processor with tokenizer
        wer_normalization: WER normalization mode
            - 'default': Minimal normalization (remove extra spaces, strip whitespace)
            - 'standardize': Aggressive normalization (lowercase, remove punctuation,
                           expand contractions, remove Kaldi non-words)
            - 'both': Compute both metrics (returns wer_default and wer_standardize)

    Returns:
        Dictionary with WER metric(s):
            - If 'default' or 'standardize': {"wer": <value>}
            - If 'both': {"wer_default": <value>, "wer_standardize": <value>}
    """
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

    # Decode using batch_decode (matching reference test script)
    pred_texts_raw = processor.batch_decode(preds, skip_special_tokens=True)
    label_texts_raw = processor.batch_decode(labels, skip_special_tokens=True)

    # Post-process to remove assistant prefix and clean up (matching reference test script)
    pred_texts = []
    for text in pred_texts_raw:
        text = text.strip()
        # Extract transcription from chat response format if needed
        if "assistant\n" in text:
            text = text.split("assistant\n", 1)[1].strip()
        pred_texts.append(text)

    label_texts = []
    for text in label_texts_raw:
        text = text.strip()
        # Extract transcription from chat response format if needed
        if "assistant\n" in text:
            text = text.split("assistant\n", 1)[1].strip()
        label_texts.append(text)

    if wer_normalization == "both":
        return {
            "wer_default": compute_wer_from_texts(label_texts, pred_texts, normalization="default"),
            "wer_standardize": compute_wer_from_texts(label_texts, pred_texts, normalization="standardize"),
        }
    else:
        return {"wer": compute_wer_from_texts(label_texts, pred_texts, normalization=wer_normalization)}
