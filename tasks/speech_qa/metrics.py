"""Metric computation utilities for speech question answering."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence

import evaluate
import numpy as np
from evaluate import EvaluationModule


_SQUAD_METRIC: Optional[EvaluationModule] = None
_AUDIT_WRAPPERS = (
    r"^\s*answer\s*:\s*",
    r"^\s*the answer is\s*",
    r"^\s*final answer\s*:\s*",
)


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


def _sanitize_prediction_ids(array: np.ndarray, tokenizer) -> np.ndarray:
    """Replace invalid/padded generation ids with pad token before decoding."""
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    numeric = np.array(array)
    invalid = (numeric < 0) | ~np.isfinite(numeric)
    if invalid.any():
        numeric = np.where(invalid, int(pad_id), numeric)
    return numeric.astype(np.int64, copy=False)


def _strip_chat_artifacts(text: str) -> str:
    text = (text or "").strip()
    if "assistant\n" in text:
        text = text.split("assistant\n", 1)[1].strip()
    return text


def _normalize_prediction_text(text: str) -> str:
    """Normalize prediction wrappers before official SQuAD scoring."""
    text = _strip_chat_artifacts(text)
    for pattern in _AUDIT_WRAPPERS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = text.strip().strip("\"'`")
    return text


def _get_squad_metric() -> EvaluationModule:
    global _SQUAD_METRIC
    if _SQUAD_METRIC is None:
        _SQUAD_METRIC = evaluate.load("squad")
    return _SQUAD_METRIC


def _build_squad_references(
    reference_answers: Sequence[Sequence[str]],
    label_texts: Sequence[str],
) -> List[Dict[str, object]]:
    references: List[Dict[str, object]] = []
    for idx, answers in enumerate(reference_answers):
        answer_texts = [str(answer).strip() for answer in (answers or []) if str(answer).strip()]
        if not answer_texts and idx < len(label_texts):
            fallback = (label_texts[idx] or "").strip()
            answer_texts = [fallback] if fallback else [""]
        references.append(
            {
                "id": str(idx),
                "answers": {
                    "text": answer_texts if answer_texts else [""],
                    "answer_start": [0] * max(1, len(answer_texts)),
                },
            }
        )
    return references


def _write_eval_audit_dump(
    *,
    dump_path: Path,
    samples_to_dump: int,
    predictions: Sequence[str],
    references: Sequence[Dict[str, object]],
    ids: Optional[Sequence[str]],
    questions: Optional[Sequence[str]],
    split_name: Optional[str],
) -> None:
    if samples_to_dump <= 0:
        return
    limit = min(int(samples_to_dump), len(predictions), len(references))
    if limit <= 0:
        return

    metric = _get_squad_metric()
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, object]] = []

    for idx in range(limit):
        ref = references[idx]
        pred = predictions[idx]
        single_result = metric.compute(
            predictions=[{"id": ref["id"], "prediction_text": pred}],
            references=[ref],
        )
        ref_answers = ref.get("answers", {}).get("text", []) if isinstance(ref, dict) else []
        records.append(
            {
                "split": split_name or "eval",
                "index": idx,
                "id": (ids[idx] if ids and idx < len(ids) else str(ref.get("id", idx))),
                "question": (questions[idx] if questions and idx < len(questions) else ""),
                "prediction_text": pred,
                "gold_answers": list(ref_answers),
                "normalized_prediction": pred,
                "best_f1": float(single_result.get("f1", 0.0)),
                "exact_match": float(single_result.get("exact_match", 0.0)),
            }
        )

    with dump_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def compute_speech_qa_metrics(
    eval_pred: Any,
    *,
    processor,
    reference_answers: Sequence[Sequence[str]],
    reference_ids: Optional[Sequence[str]] = None,
    reference_questions: Optional[Sequence[str]] = None,
    audit_dump_path: Optional[str | Path] = None,
    audit_samples: int = 0,
    split_name: Optional[str] = None,
) -> Dict[str, float]:
    """Compute SQuAD exact-match/F1 in 0-100 scale using the official HF SQuAD metric."""
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    if isinstance(labels, tuple):
        labels = labels[0]

    preds = np.array(preds)
    labels = np.array(labels)
    preds = _sanitize_prediction_ids(preds, processor.tokenizer)

    tokenizer = processor.tokenizer
    pred_texts_raw = _safe_batch_decode(tokenizer, preds, skip_special_tokens=True)
    label_texts_raw = _decode_labels(labels, tokenizer)

    pred_texts = [_normalize_prediction_text(text) for text in pred_texts_raw]
    label_texts = [_strip_chat_artifacts(text) for text in label_texts_raw]

    total = min(len(pred_texts), len(reference_answers))
    if total == 0:
        return {"exact_match": 0.0, "f1": 0.0, "num_samples": 0.0}

    pred_texts = pred_texts[:total]
    label_texts = label_texts[:total]
    references = _build_squad_references(reference_answers[:total], label_texts)
    predictions = [{"id": str(idx), "prediction_text": pred_texts[idx]} for idx in range(total)]

    metric = _get_squad_metric()
    result = metric.compute(predictions=predictions, references=references)
    exact_match = float(result.get("exact_match", 0.0))
    f1_mean = float(result.get("f1", 0.0))

    if audit_dump_path:
        _write_eval_audit_dump(
            dump_path=Path(audit_dump_path),
            samples_to_dump=int(max(0, audit_samples)),
            predictions=pred_texts,
            references=references,
            ids=list(reference_ids) if reference_ids is not None else None,
            questions=list(reference_questions) if reference_questions is not None else None,
            split_name=split_name,
        )

    return {
        "exact_match": exact_match,
        "f1": f1_mean,
        "num_samples": float(total),
    }


__all__ = ["compute_speech_qa_metrics"]
