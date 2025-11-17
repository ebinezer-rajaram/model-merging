"""Dataset helpers and data collator for speech question answering tasks."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from datasets import Dataset, DatasetDict

from core import resolve_num_proc
from core.tasks.dataset import (
    _column_exists,
    _normalize_target_count,
    _rename_column,
    _samples_key,
    _select_candidate_column,
    add_duration_to_dataset,
    assemble_splits,
    cache_and_sample_splits,
    load_dataset,
    print_dataset_summary,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATASET_CACHE_ROOT = PACKAGE_ROOT / "artifacts" / "speech_qa" / "datasets"

DEFAULT_DATASET_NAME = "kresnik/spoken_squad"
FALLBACK_QUESTION_COLUMNS: Tuple[str, ...] = ("question", "query")
FALLBACK_TRANSCRIPT_COLUMNS: Tuple[str, ...] = ("transcript", "text", "sentence", "context")
FALLBACK_CONTEXT_COLUMNS: Tuple[str, ...] = ("context", "passage", "paragraph")
FALLBACK_ID_COLUMNS: Tuple[str, ...] = ("id", "qid", "example_id")
FALLBACK_ANSWER_COLUMNS: Tuple[str, ...] = ("answers", "answer")

MANIFEST_FIELDS: Tuple[str, ...] = (
    "id",
    "question",
    "label_text",
    "answers",
    "context",
    "transcript",
    "path",
)


def _normalize_answers(example: Dict[str, Any], answer_column: str) -> Dict[str, Any]:
    """Normalize answer formats to a consistent list of strings."""
    answers = example.get(answer_column)
    normalized: List[str] = []
    if isinstance(answers, dict):
        texts = answers.get("text") or []
        normalized.extend(texts)
    elif isinstance(answers, (list, tuple)):
        for entry in answers:
            if isinstance(entry, str):
                normalized.append(entry)
            elif isinstance(entry, Mapping):
                maybe_text = entry.get("text")
                if isinstance(maybe_text, str):
                    normalized.append(maybe_text)
                elif isinstance(maybe_text, (list, tuple)):
                    normalized.extend(str(item) for item in maybe_text if item)
    elif answers:
        normalized.append(str(answers))

    normalized = [text.strip() for text in normalized if isinstance(text, str) and text.strip()]
    if not normalized:
        normalized = [""]

    example["answers"] = normalized
    example["label_text"] = normalized[0]
    return example


def load_speech_qa_dataset(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_config: Optional[str] = None,
    max_train_samples: Optional[int] = None,
    max_validation_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    max_duration: Optional[float] = None,
    min_duration: Optional[float] = None,
    seed: int = 0,
    num_proc: Optional[int | str] = None,
    cache_dir: Optional[Path | str] = None,
    cache_splits: bool = True,
    force_rebuild: bool = False,
    audio_column: Optional[str] = None,
    question_column: Optional[str] = None,
    transcript_column: Optional[str] = None,
    context_column: Optional[str] = None,
    id_column: Optional[str] = None,
    answer_column: Optional[str] = None,
    split_percentages: Optional[Mapping[str, float] | Sequence[float]] = None,
    train_split: str = "train",
    validation_split: Optional[str] = "validation",
    test_split: Optional[str] = "test",
) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset], Dict[str, List[List[str]]]]:
    """
    Load an audio question answering dataset with optional sub-sampling.

    Returns train, validation, test splits (None when missing) and a mapping of per-split answer sets.
    """
    # Normalize sample counts
    max_train_samples = _normalize_target_count("max_train_samples", max_train_samples)
    max_validation_samples = _normalize_target_count("max_validation_samples", max_validation_samples)
    max_test_samples = _normalize_target_count("max_test_samples", max_test_samples)

    # Load dataset
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config)
    else:
        dataset = load_dataset(dataset_name)

    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected DatasetDict, received {type(dataset)!r}")

    # Normalize column names (question, transcript, context, id, answers)
    if question_column and question_column != "question":
        dataset = _rename_column(dataset, question_column, "question")
    elif not _column_exists(dataset, "question"):
        fallback_question = _select_candidate_column(dataset, FALLBACK_QUESTION_COLUMNS)
        if fallback_question and fallback_question != "question":
            dataset = _rename_column(dataset, fallback_question, "question")
        else:
            raise ValueError("Dataset must contain a question column. Provide 'question_column'.")

    if transcript_column and transcript_column != "transcript":
        dataset = _rename_column(dataset, transcript_column, "transcript")
    elif not _column_exists(dataset, "transcript"):
        fallback_transcript = _select_candidate_column(dataset, FALLBACK_TRANSCRIPT_COLUMNS)
        if fallback_transcript and fallback_transcript != "transcript":
            dataset = _rename_column(dataset, fallback_transcript, "transcript")

    if context_column and context_column != "context":
        dataset = _rename_column(dataset, context_column, "context")
    elif not _column_exists(dataset, "context"):
        fallback_context = _select_candidate_column(dataset, FALLBACK_CONTEXT_COLUMNS)
        if fallback_context and fallback_context != "context":
            dataset = _rename_column(dataset, fallback_context, "context")

    if id_column and id_column != "id":
        dataset = _rename_column(dataset, id_column, "id")
    elif not _column_exists(dataset, "id"):
        fallback_id = _select_candidate_column(dataset, FALLBACK_ID_COLUMNS)
        if fallback_id and fallback_id != "id":
            dataset = _rename_column(dataset, fallback_id, "id")

    target_answer_column = answer_column or _select_candidate_column(dataset, FALLBACK_ANSWER_COLUMNS)
    if not target_answer_column:
        raise ValueError("Dataset must include an answers column. Provide 'answer_column'.")

    # Apply split percentages or assemble standard splits
    # (Speech QA typically doesn't need stratification, so we skip that complexity)
    dataset = assemble_splits(
        dataset,
        train_split=train_split,
        validation_split=validation_split,
        test_split=test_split,
    )

    # Ensure audio column
    from core.tasks.dataset import _ensure_audio_column
    dataset, audio_column_name = _ensure_audio_column(dataset, preferred=audio_column)

    # Normalize answers
    normalize_fn = partial(_normalize_answers, answer_column=target_answer_column)
    dataset = dataset.map(normalize_fn)

    # Add duration information
    effective_num_proc = resolve_num_proc(num_proc)
    dataset = add_duration_to_dataset(dataset, audio_column=audio_column_name, num_proc=effective_num_proc)

    # Filter by duration if specified
    if max_duration is not None or min_duration is not None:
        def _keep_duration(example: dict) -> bool:
            duration = example.get("duration") or 0.0
            duration_float = float(duration)
            if max_duration is not None and duration_float > max_duration:
                return False
            if min_duration is not None and duration_float < min_duration:
                return False
            return True

        for split_name in list(dataset.keys()):
            before = len(dataset[split_name])
            dataset[split_name] = dataset[split_name].filter(_keep_duration)
            after = len(dataset[split_name])
            if after != before:
                filtered_count = before - after
                duration_info = []
                if max_duration is not None:
                    duration_info.append(f">{max_duration:.1f}s")
                if min_duration is not None:
                    duration_info.append(f"<{min_duration:.1f}s")
                duration_str = " or ".join(duration_info)
                print(f"⏱️ Filtered {filtered_count} {split_name} samples ({duration_str}).")

    # Build cache path
    cache_root = Path(cache_dir) if cache_dir is not None else DATASET_CACHE_ROOT
    dataset_key = dataset_name.replace("/", "_")
    config_key = dataset_config or "default"
    cache_name = (
        f"{dataset_key}_{config_key}"
        f"_train_{_samples_key(max_train_samples)}"
        f"_val_{_samples_key(max_validation_samples)}"
        f"_test_{_samples_key(max_test_samples)}"
        f"_seed_{int(seed)}.json"
    )
    cache_path = cache_root / cache_name

    # Cache and sample splits
    train_subset, validation_subset, test_subset, splits_metadata, payload_seed = cache_and_sample_splits(
        dataset,
        cache_path=cache_path,
        max_train_samples=max_train_samples,
        max_validation_samples=max_validation_samples,
        max_test_samples=max_test_samples,
        seed=seed,
        manifest_fields=MANIFEST_FIELDS,
        audio_column=audio_column_name,
        cache_splits=cache_splits,
        force_rebuild=force_rebuild,
        additional_metadata={
            "dataset": dataset_name,
            "config": dataset_config,
            "audio_column": audio_column_name,
        },
    )

    # Collect answers for evaluation
    answers_map: Dict[str, List[List[str]]] = {}

    def _collect_answers(ds: Optional[Dataset]) -> List[List[str]]:
        if ds is None:
            return []
        column = ds["answers"] if "answers" in ds.column_names else [[] for _ in range(len(ds))]
        collected: List[List[str]] = []
        for entry in column:
            if isinstance(entry, (list, tuple)):
                collected.append([str(item).strip() for item in entry if str(item).strip()])
            else:
                collected.append([str(entry).strip()] if entry else [""])
        return collected

    answers_map["train"] = _collect_answers(train_subset)
    answers_map["validation"] = _collect_answers(validation_subset)
    answers_map["test"] = _collect_answers(test_subset)

    # Print summary
    print_dataset_summary(
        task_emoji="❓",
        task_name="Speech-QA dataset",
        train_subset=train_subset,
        validation_subset=validation_subset,
        test_subset=test_subset,
        splits_metadata=splits_metadata,
        seed=payload_seed,
        num_proc=effective_num_proc,
    )

    return train_subset, validation_subset, test_subset, answers_map


@dataclass
class SpeechQACollator:
    """Prepare batches for speech question answering fine-tuning.

    Always uses chat template with both user message (audio + instruction) and assistant response (ground truth).
    During evaluation, the CustomTrainer's prediction_step will strip out the ground truth before generation.
    """

    processor: Any
    sampling_rate: int
    include_transcript: bool = True
    include_context: bool = False

    def _select_label(self, feature: Dict[str, Any]) -> str:
        if "label_text" in feature and feature["label_text"]:
            return str(feature["label_text"])
        answers = feature.get("answers") or []
        if isinstance(answers, (list, tuple)) and answers:
            return str(answers[0])
        return ""

    def _build_instruction(self, feature: Dict[str, Any]) -> str:
        """Build the instruction text for the user message."""
        question = str(feature.get("question", "")).strip()
        transcript = str(feature.get("transcript", "") or "").strip()
        context = str(feature.get("context", "") or "").strip()

        instruction = "Listen to the audio segment and answer the question."
        if question:
            instruction += f"\nQuestion: {question}"
        if transcript and self.include_transcript:
            instruction += f"\nTranscript: {transcript}"
        if context and self.include_context:
            instruction += f"\nContext: {context}"

        return instruction

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_arrays = [feature["audio"]["array"] for feature in features]
        labels = [self._select_label(feature) for feature in features]

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "padding_side", None) != "left":
            tokenizer.padding_side = "left"

        # Build prompts using chat template format
        prompts = []
        for feature, label in zip(features, labels):
            instruction = self._build_instruction(feature)

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": None},
                        {"type": "text", "text": instruction}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": label}
                    ]
                }
            ]
            prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=False,
                tokenize=False
            )
            prompts.append(prompt)

        inputs = self.processor(
            audio=audio_arrays,
            sampling_rate=self.sampling_rate,
            text=prompts,
            return_tensors="pt",
            padding=True,
        )

        label_ids = inputs["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        audio_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.audio_token
        )

        # Mask padding and audio tokens
        label_ids = label_ids.masked_fill(label_ids == pad_id, -100)
        label_ids = label_ids.masked_fill(label_ids == audio_token_id, -100)

        # Mask everything except the assistant's answer response
        for i, label in enumerate(labels):
            label_tokens = tokenizer.encode(label, add_special_tokens=False)
            input_ids = inputs["input_ids"][i]
            label_length = len(label_tokens)

            # Search for the label tokens in the sequence
            found = False
            for j in range(len(input_ids) - label_length + 1):
                if torch.all(input_ids[j:j + label_length] == torch.tensor(label_tokens, device=input_ids.device)):
                    label_ids[i, :j] = -100
                    found = True
                    break

            # Fallback: mask based on sequence structure
            if not found:
                non_masked = (label_ids[i] != -100).nonzero(as_tuple=False)
                if len(non_masked) > label_length:
                    mask_until = non_masked[-label_length].item()
                    label_ids[i, :mask_until] = -100

        inputs["labels"] = label_ids
        return inputs


__all__ = ["load_speech_qa_dataset", "SpeechQACollator"]
