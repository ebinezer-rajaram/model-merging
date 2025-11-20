"""Task-specific metric computation for Speech Translation."""

from typing import Any, Dict

import numpy as np
import sacrebleu


def compute_st_metrics(
    eval_pred: Any,
    processor,
) -> Dict[str, float]:
    """Compute case-sensitive detokenized BLEU for speech translation evaluation.

    Args:
        eval_pred: Evaluation predictions (preds, labels) from trainer
        processor: Model processor with tokenizer

    Returns:
        Dictionary with BLEU score: {"bleu": <value>}
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

    # Decode using batch_decode
    pred_texts_raw = processor.batch_decode(preds, skip_special_tokens=True)
    label_texts_raw = processor.batch_decode(labels, skip_special_tokens=True)

    # Post-process: remove chat template artifacts and strip whitespace (minimal normalization)
    def clean_text(text: str) -> str:
        """Clean chat template artifacts while preserving case and punctuation."""
        text = text.strip()
        # Extract translation from chat response format if needed
        if "assistant\n" in text:
            text = text.split("assistant\n", 1)[1].strip()
        return text

    pred_texts = [clean_text(text) for text in pred_texts_raw]
    label_texts = [clean_text(text) for text in label_texts_raw]

    # Compute case-sensitive detokenized BLEU with smoothing
    # sacrebleu expects references as list of lists (multiple references per prediction)
    references = [[text] for text in label_texts]

    # lowercase=False: case-sensitive evaluation
    # tokenize='13a': international tokenization (standard for MT evaluation)
    # smooth_method='exp': exponential smoothing (BLEU+1) to handle short sentences and rare n-grams
    # smooth_value=0.01: small epsilon for smoothing (default but explicit)
    bleu = sacrebleu.corpus_bleu(
        pred_texts,
        references,
        lowercase=False,
        tokenize='13a',
        smooth_method='exp',
        smooth_value=0.01
    )

    return {"bleu": bleu.score}
