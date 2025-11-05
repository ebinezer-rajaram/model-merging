"""Metric computation utilities for speech question answering."""

from __future__ import annotations

import collections
import re
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


_ARTICLES = {"a", "an", "the"}


def _normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation/articles/extra whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^0-9a-z\s]", " ", text)
    tokens = [token for token in text.split() if token and token not in _ARTICLES]
    return " ".join(tokens)


def _f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    truth_tokens = _normalize_answer(ground_truth).split()
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))


def _safe_decode_ids(tokenizer, token_ids, *, skip_special_tokens: bool = True) -> str:
    """Decode token ids into text while gracefully handling None tokens."""
    if token_ids is None:
        return ""
    if isinstance(token_ids, np.ndarray):
        token_ids = token_ids.tolist()
    if isinstance(token_ids, (list, tuple)):
        token_ids = [int(token) for token in token_ids if token is not None]
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    except (TypeError, ValueError):
        tokens = tokenizer.convert_ids_to_tokens(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )
        tokens = [token for token in tokens if token is not None]
        if not tokens:
            return ""
        try:
            return tokenizer.convert_tokens_to_string(tokens)
        except (TypeError, ValueError):
            return "".join(token for token in tokens if token is not None)


def _safe_batch_decode(tokenizer, sequences, *, skip_special_tokens: bool = True) -> Sequence[str]:
    """Batch decode sequences with None-token protection."""
    decoded: list[str] = []
    for sequence in sequences:
        decoded.append(_safe_decode_ids(tokenizer, sequence, skip_special_tokens=skip_special_tokens))
    return decoded


def _decode_labels(array: np.ndarray, tokenizer) -> Sequence[str]:
    decoded: list[str] = []
    for row in array:
        tokens = [token for token in row.tolist() if token != -100]
        decoded.append(_safe_decode_ids(tokenizer, tokens, skip_special_tokens=True))
    return decoded


def _best_over_ground_truths(metric_fn, prediction: str, ground_truths: Iterable[str]) -> float:
    scores = [metric_fn(prediction, truth) for truth in ground_truths]
    return max(scores) if scores else 0.0


def compute_speech_qa_metrics(
    eval_pred: Any,
    *,
    processor,
    reference_answers: Sequence[Sequence[str]],
) -> Dict[str, float]:
    """Compute exact match and F1 metrics for generated answers."""
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    if isinstance(labels, tuple):
        labels = labels[0]

    preds = np.array(preds)
    labels = np.array(labels)

    tokenizer = processor.tokenizer
    pred_texts = _safe_batch_decode(tokenizer, preds, skip_special_tokens=True)
    label_texts = _decode_labels(labels, tokenizer)

    total = min(len(pred_texts), len(reference_answers))
    if total == 0:
        return {"exact_match": 0.0, "f1": 0.0, "num_samples": 0.0}

    exact_scores: List[float] = []
    f1_scores: List[float] = []

    for idx in range(total):
        prediction = pred_texts[idx]
        target_answers = reference_answers[idx]
        if not target_answers:
            target_answers = [label_texts[idx]]

        exact = _best_over_ground_truths(_exact_match_score, prediction, target_answers)
        f1 = _best_over_ground_truths(_f1_score, prediction, target_answers)
        exact_scores.append(exact)
        f1_scores.append(f1)

    exact_match = float(np.mean(exact_scores)) if exact_scores else 0.0
    f1_mean = float(np.mean(f1_scores)) if f1_scores else 0.0

    return {
        "exact_match": exact_match * 100.0,
        "f1": f1_mean * 100.0,
        "num_samples": float(total),
    }


__all__ = ["compute_speech_qa_metrics"]
