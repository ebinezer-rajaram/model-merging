"""Metric computation utilities for speaker identification."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Sequence

import numpy as np


def _normalize_label(text: str) -> str:
    """Lowercase and collapse non-alphanumeric characters."""
    normalized = re.sub(r"[^0-9a-z]+", " ", text.lower())
    return " ".join(normalized.split())


def _extract_candidate_span(text: str) -> str:
    """Return the portion of `text` most likely containing the predicted speaker."""
    if not text:
        return ""

    match = re.search(r"speaker\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    candidate = match.group(1) if match else text
    candidate = candidate.strip()

    if not candidate:
        return ""

    candidate = candidate.splitlines()[0].strip()
    candidate = re.split(r"[.!?]", candidate, maxsplit=1)[0].strip()
    return candidate


def _find_first_label_match(normalized_text: str, label_map: Dict[str, int]) -> int | None:
    """Return the label index corresponding to the first occurrence in `normalized_text`."""
    if not normalized_text:
        return None

    best_idx: int | None = None
    best_pos: int | None = None
    for name, idx in label_map.items():
        if not name:
            continue
        pos = normalized_text.find(name)
        if pos == -1:
            continue
        if best_pos is None or pos < best_pos:
            best_pos = pos
            best_idx = idx
    return best_idx


def _match_label(text: str, label_map: Dict[str, int]) -> int | None:
    """Map a generated string back to a known label index."""
    candidate = _normalize_label(_extract_candidate_span(text))
    idx = _find_first_label_match(candidate, label_map)
    if idx is not None:
        return idx

    normalized = _normalize_label(text)
    if not normalized:
        return None
    return _find_first_label_match(normalized, label_map)


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


def _compute_macro_f1(y_true: Iterable[int], y_pred: Iterable[int], num_classes: int) -> float:
    """Compute macro-F1 treating out-of-vocabulary predictions as false negatives."""
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label < 0 or true_label >= num_classes:
            continue
        if 0 <= pred_label < num_classes:
            if pred_label == true_label:
                tp[true_label] += 1
            else:
                fp[pred_label] += 1
                fn[true_label] += 1
        else:
            fn[true_label] += 1

    scores = []
    for idx in range(num_classes):
        precision_den = tp[idx] + fp[idx]
        recall_den = tp[idx] + fn[idx]
        precision = tp[idx] / precision_den if precision_den > 0 else 0.0
        recall = tp[idx] / recall_den if recall_den > 0 else 0.0
        if precision + recall == 0.0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))

    return float(np.mean(scores)) if scores else 0.0


def _compute_weighted_f1(y_true: Iterable[int], y_pred: Iterable[int], num_classes: int) -> float:
    """Compute weighted-F1 treating out-of-vocabulary predictions as false negatives."""
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label < 0 or true_label >= num_classes:
            continue
        if 0 <= pred_label < num_classes:
            if pred_label == true_label:
                tp[true_label] += 1
            else:
                fp[pred_label] += 1
                fn[true_label] += 1
        else:
            fn[true_label] += 1

    weighted_sum = 0.0
    total_support = 0
    for idx in range(num_classes):
        support = tp[idx] + fn[idx]
        if support == 0:
            continue

        precision_den = tp[idx] + fp[idx]
        recall_den = tp[idx] + fn[idx]
        precision = tp[idx] / precision_den if precision_den > 0 else 0.0
        recall = tp[idx] / recall_den if recall_den > 0 else 0.0

        if precision + recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        weighted_sum += f1 * support
        total_support += support

    return float(weighted_sum / total_support) if total_support > 0 else 0.0


def compute_speaker_id_metrics(
    eval_pred: Any,
    *,
    processor,
    label_names: Sequence[str],
    store_predictions: bool = False,
) -> Dict[str, float]:
    """Compute accuracy and macro-F1 for generated speaker labels.

    Args:
        eval_pred: Tuple of (predictions, labels)
        processor: Model processor containing tokenizer
        label_names: List of speaker label names
        store_predictions: If True, store predictions and labels for confusion matrix

    Returns:
        Dictionary containing metrics and optionally prediction data
    """
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    if isinstance(labels, tuple):
        labels = labels[0]

    preds = np.array(preds)
    labels = np.array(labels)

    tokenizer = processor.tokenizer
    pred_texts_raw = _safe_batch_decode(tokenizer, preds, skip_special_tokens=True)
    label_texts_raw = _decode_labels(labels, tokenizer)

    # Post-process to remove chat template artifacts (matching ASR approach)
    pred_texts = []
    for text in pred_texts_raw:
        text = text.strip()
        # Extract speaker from chat response format if needed
        if "assistant\n" in text:
            text = text.split("assistant\n", 1)[1].strip()
        pred_texts.append(text)

    label_texts = []
    for text in label_texts_raw:
        text = text.strip()
        # Extract speaker from chat response format if needed
        if "assistant\n" in text:
            text = text.split("assistant\n", 1)[1].strip()
        label_texts.append(text)

    label_map = {_normalize_label(name): idx for idx, name in enumerate(label_names)}
    default_total = len(label_names)

    target_indices = []
    for text in label_texts:
        matched = _match_label(text, label_map)
        if matched is None:
            target_indices.append(-1)
        else:
            target_indices.append(matched)

    pred_indices = []
    recognized = 0
    for text in pred_texts:
        matched = _match_label(text, label_map)
        if matched is not None:
            pred_indices.append(matched)
            recognized += 1
        else:
            pred_indices.append(-1)

    total = sum(1 for idx in target_indices if idx >= 0)
    correct = sum(
        1 for target, pred in zip(target_indices, pred_indices) if target >= 0 and target == pred
    )
    accuracy = float(correct / total) if total else 0.0

    macro_f1 = _compute_macro_f1(target_indices, pred_indices, default_total)
    weighted_f1 = _compute_weighted_f1(target_indices, pred_indices, default_total)

    recognized_rate = float(recognized / total) if total else 0.0

    result = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "recognized_rate": recognized_rate,
        "num_samples": float(total),
    }

    # Store predictions for confusion matrix if requested
    if store_predictions:
        result["_predictions"] = pred_indices
        result["_labels"] = target_indices

    return result


__all__ = ["compute_speaker_id_metrics"]
