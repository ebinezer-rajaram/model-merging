"""Metric computation utilities for speech question answering."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

import evaluate
import numpy as np
from evaluate import EvaluationModule


_SQUAD_METRIC: Optional[EvaluationModule] = None
_AUDIT_WRAPPERS = (
    r"^\s*answer\s*:\s*",
    r"^\s*the answer is\s*",
    r"^\s*final answer\s*:\s*",
)
_OPTION_STRICT_FULL = re.compile(r"^\s*[\(\[\{\"\']?\s*([ABCD])\s*[\)\]\}\"'\.\:\;\!\?]?\s*$", flags=re.IGNORECASE)
_OPTION_EXPLICIT = re.compile(
    r"(?:^|\b)(?:answer|final answer|option|choice)\s*(?:is|:)?\s*([ABCD])\b",
    flags=re.IGNORECASE,
)
_OPTION_LINE_PREFIX = re.compile(r"^\s*([ABCD])(?:[\)\]\}\.\:\-]|$)", flags=re.IGNORECASE)


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


def _normalize_choice_text(text: Any) -> str:
    value = str(text or "").strip().lower()
    return re.sub(r"\s+", " ", value)


def _extract_option_letter(text: str) -> Optional[str]:
    cleaned = _normalize_prediction_text(text)
    if not cleaned:
        return None

    # 1) Strict single-letter outputs like "C", "(C)", "C."
    full = _OPTION_STRICT_FULL.match(cleaned)
    if full:
        return full.group(1).upper()

    # 2) Explicit answer patterns like "Answer: C", "final answer is B", "option D".
    explicit_matches = list(_OPTION_EXPLICIT.finditer(cleaned))
    if explicit_matches:
        return explicit_matches[-1].group(1).upper()

    # 3) Last non-empty line starts with a label token like "C)" or "B:"
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        line_match = _OPTION_LINE_PREFIX.match(lines[-1])
        if line_match:
            return line_match.group(1).upper()

    return None


def _resolve_prediction_text_for_squad(
    *,
    prediction_text: str,
    predicted_letter: Optional[str],
    choice_map: Optional[Mapping[str, str]],
) -> str:
    if predicted_letter and choice_map:
        value = str(choice_map.get(predicted_letter, "")).strip()
        if value:
            return value
    return _normalize_prediction_text(prediction_text)


def _resolve_gold_letter(
    *,
    gold_answers: Sequence[str],
    decoded_label_text: str,
    choice_map: Optional[Mapping[str, str]],
) -> Optional[str]:
    label_candidate = str(decoded_label_text or "").strip().upper()
    if label_candidate in {"A", "B", "C", "D"}:
        return label_candidate
    if not choice_map:
        return None
    normalized_answers = {_normalize_choice_text(answer) for answer in gold_answers if str(answer).strip()}
    for label in ("A", "B", "C", "D"):
        choice_value = _normalize_choice_text(choice_map.get(label, ""))
        if choice_value and choice_value in normalized_answers:
            return label
    return None


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
    predicted_letters: Optional[Sequence[Optional[str]]] = None,
    gold_letters: Optional[Sequence[Optional[str]]] = None,
    subtasks: Optional[Sequence[str]] = None,
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
                "predicted_letter": (
                    predicted_letters[idx] if predicted_letters and idx < len(predicted_letters) else None
                ),
                "gold_letter": (
                    gold_letters[idx] if gold_letters and idx < len(gold_letters) else None
                ),
                "task_name": (
                    subtasks[idx] if subtasks and idx < len(subtasks) else ""
                ),
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
    reference_choice_maps: Optional[Sequence[Mapping[str, str]]] = None,
    reference_subtasks: Optional[Sequence[str]] = None,
    audit_dump_path: Optional[str | Path] = None,
    audit_samples: int = 0,
    debug_eval_dump_path: Optional[str | Path] = None,
    debug_eval_dump_samples: int = 0,
    split_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute letter accuracy (primary) with SQuAD EM/F1 diagnostics."""
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
        return {
            "accuracy": 0.0,
            "recognized_rate": 0.0,
            "exact_match": 0.0,
            "f1": 0.0,
            "num_samples": 0.0,
            "subtask_accuracy": {},
            "subtask_num_samples": {},
        }

    pred_texts = pred_texts[:total]
    label_texts = label_texts[:total]
    choice_maps: List[Mapping[str, str]] = []
    if reference_choice_maps is None:
        choice_maps = [{} for _ in range(total)]
    else:
        for idx in range(total):
            if idx < len(reference_choice_maps) and isinstance(reference_choice_maps[idx], Mapping):
                choice_maps.append(reference_choice_maps[idx])
            else:
                choice_maps.append({})

    subtasks: List[str] = []
    if reference_subtasks is None:
        subtasks = ["" for _ in range(total)]
    else:
        for idx in range(total):
            subtasks.append(str(reference_subtasks[idx]) if idx < len(reference_subtasks) else "")

    pred_letters: List[Optional[str]] = []
    gold_letters: List[Optional[str]] = []
    squad_pred_texts: List[str] = []

    for idx in range(total):
        pred_letter = _extract_option_letter(pred_texts[idx])
        choice_map = choice_maps[idx]
        gold_answers = list(reference_answers[idx] or [])
        gold_letter = _resolve_gold_letter(
            gold_answers=gold_answers,
            decoded_label_text=label_texts[idx],
            choice_map=choice_map,
        )
        pred_letters.append(pred_letter)
        gold_letters.append(gold_letter)
        squad_pred_texts.append(
            _resolve_prediction_text_for_squad(
                prediction_text=pred_texts[idx],
                predicted_letter=pred_letter,
                choice_map=choice_map,
            )
        )

    references = _build_squad_references(reference_answers[:total], label_texts)
    predictions = [{"id": str(idx), "prediction_text": squad_pred_texts[idx]} for idx in range(total)]

    metric = _get_squad_metric()
    result = metric.compute(predictions=predictions, references=references)
    exact_match = float(result.get("exact_match", 0.0))
    f1_mean = float(result.get("f1", 0.0))

    recognized = 0
    correct = 0
    subtask_totals: Dict[str, int] = {}
    subtask_correct: Dict[str, int] = {}
    for idx in range(total):
        pred_letter = pred_letters[idx]
        gold_letter = gold_letters[idx]
        if pred_letter is not None:
            recognized += 1
        if pred_letter is not None and gold_letter is not None and pred_letter == gold_letter:
            correct += 1
            subtask = subtasks[idx]
            if subtask:
                subtask_correct[subtask] = subtask_correct.get(subtask, 0) + 1
        subtask = subtasks[idx]
        if subtask:
            subtask_totals[subtask] = subtask_totals.get(subtask, 0) + 1

    subtask_accuracy = {
        key: (float(subtask_correct.get(key, 0)) / float(count))
        for key, count in sorted(subtask_totals.items())
        if count > 0
    }
    subtask_num_samples = {key: int(count) for key, count in sorted(subtask_totals.items())}

    resolved_dump_path = audit_dump_path or debug_eval_dump_path
    resolved_samples = int(max(0, audit_samples))
    if resolved_samples <= 0:
        resolved_samples = int(max(0, debug_eval_dump_samples))

    if resolved_dump_path:
        _write_eval_audit_dump(
            dump_path=Path(resolved_dump_path),
            samples_to_dump=resolved_samples,
            predictions=squad_pred_texts,
            references=references,
            ids=list(reference_ids) if reference_ids is not None else None,
            questions=list(reference_questions) if reference_questions is not None else None,
            split_name=split_name,
            predicted_letters=pred_letters,
            gold_letters=gold_letters,
            subtasks=subtasks,
        )

    return {
        "accuracy": float(correct / total),
        "recognized_rate": float(recognized / total),
        "exact_match": exact_match,
        "f1": f1_mean,
        "num_samples": float(total),
        "subtask_accuracy": subtask_accuracy,
        "subtask_num_samples": subtask_num_samples,
    }


__all__ = ["compute_speech_qa_metrics"]
