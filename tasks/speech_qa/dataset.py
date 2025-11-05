"""Dataset helpers and data collator for speech question answering tasks."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from datasets import Dataset, DatasetDict, load_dataset

from core.dataset_utils import (
    add_duration,
    build_split_metadata,
    load_cached_split,
    normalize_split_metadata,
    num_proc_map_kwargs,
    resolve_num_proc,
    save_cached_split,
    select_random_indices,
    subset_dataset_by_metadata,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATASET_CACHE_ROOT = PACKAGE_ROOT / "artifacts" / "speech_qa" / "datasets"

DEFAULT_DATASET_NAME = "kresnik/spoken_squad"
DEFAULT_AUDIO_COLUMN = "audio"
FALLBACK_AUDIO_COLUMNS: Tuple[str, ...] = ("speech",)
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


def _samples_key(value: Optional[int]) -> str:
    return "full" if value is None else str(int(value)).zfill(6)


def _normalize_target_count(name: str, value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative, received {value!r}")
    return parsed


def _column_exists(dataset: DatasetDict, column: str) -> bool:
    try:
        sample_split = next(iter(dataset.values()))
    except StopIteration:
        return False
    return column in sample_split.column_names


def _rename_column(dataset: DatasetDict, source: str, target: str) -> DatasetDict:
    if not source or source == target:
        return dataset
    if not _column_exists(dataset, source):
        raise ValueError(f"Column {source!r} not found in dataset splits.")
    if _column_exists(dataset, target):
        raise ValueError(
            f"Cannot rename column {source!r} to {target!r} because {target!r} already exists."
        )
    return dataset.rename_column(source, target)


def _ensure_audio_column(dataset: DatasetDict, preferred: Optional[str] = None) -> Tuple[DatasetDict, str]:
    """Ensure the dataset has an audio column named DEFAULT_AUDIO_COLUMN."""
    try:
        sample_split = next(iter(dataset.values()))
    except StopIteration as exc:
        raise ValueError("Dataset has no splits to infer audio column from.") from exc

    if preferred:
        if preferred == DEFAULT_AUDIO_COLUMN and DEFAULT_AUDIO_COLUMN in sample_split.column_names:
            return dataset, DEFAULT_AUDIO_COLUMN
        if preferred in sample_split.column_names:
            if preferred != DEFAULT_AUDIO_COLUMN:
                dataset = dataset.rename_column(preferred, DEFAULT_AUDIO_COLUMN)
            return dataset, DEFAULT_AUDIO_COLUMN

    if DEFAULT_AUDIO_COLUMN in sample_split.column_names:
        return dataset, DEFAULT_AUDIO_COLUMN

    for candidate in FALLBACK_AUDIO_COLUMNS:
        if candidate in sample_split.column_names:
            dataset = dataset.rename_column(candidate, DEFAULT_AUDIO_COLUMN)
            return dataset, DEFAULT_AUDIO_COLUMN

    raise ValueError(
        f"Could not find an audio column in dataset splits. Checked {DEFAULT_AUDIO_COLUMN!r}"
        f" and {FALLBACK_AUDIO_COLUMNS!r}."
    )


def _select_candidate_column(dataset: DatasetDict, candidates: Iterable[str]) -> Optional[str]:
    for name in candidates:
        if _column_exists(dataset, name):
            return name
    return None


def _normalize_answers(example: Dict[str, Any], answer_column: str) -> Dict[str, Any]:
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

    max_train_samples = _normalize_target_count("max_train_samples", max_train_samples)
    max_validation_samples = _normalize_target_count("max_validation_samples", max_validation_samples)
    max_test_samples = _normalize_target_count("max_test_samples", max_test_samples)

    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config)
    else:
        dataset = load_dataset(dataset_name)

    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected DatasetDict, received {type(dataset)!r}")

    if question_column and question_column != "question":
        dataset = _rename_column(dataset, question_column, "question")
    elif not _column_exists(dataset, "question"):
        fallback_question = _select_candidate_column(dataset, FALLBACK_QUESTION_COLUMNS)
        if fallback_question and fallback_question != "question":
            dataset = _rename_column(dataset, fallback_question, "question")
        else:
            raise ValueError(
                "Dataset must contain a question column. Provide 'question_column'."
            )

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

    preferred_audio_column = audio_column

    split_ratios: Optional[Tuple[float, float, float]] = None
    if split_percentages is not None:
        if isinstance(split_percentages, Mapping):
            train_ratio = float(split_percentages.get("train", 0.0))
            val_ratio = float(split_percentages.get("validation", 0.0))
            test_ratio = float(split_percentages.get("test", 0.0))
            split_ratios = (train_ratio, val_ratio, test_ratio)
        else:
            ratios = list(split_percentages)
            if len(ratios) != 3:
                raise ValueError("split_percentages must contain exactly three values (train/val/test).")
            split_ratios = tuple(float(value) for value in ratios)

        total = sum(split_ratios)
        if total <= 0:
            raise ValueError("split_percentages must sum to a positive value.")
        split_ratios = tuple(value / total for value in split_ratios)

    if split_ratios is not None:
        if not train_split or train_split not in dataset:
            raise ValueError(f"Requested train split {train_split!r} not found in dataset.")
        base_train = dataset[train_split]

        train_ratio, val_ratio, test_ratio = split_ratios

        if not (0.0 <= test_ratio < 1.0):
            raise ValueError("Test split percentage must be in the range [0, 1).")

        if test_ratio > 0.0:
            first_split = base_train.train_test_split(
                test_size=test_ratio,
                seed=seed,
            )
            train_portion = first_split["train"]
            test_portion = first_split["test"]
        else:
            train_portion = base_train
            test_portion = None

        if val_ratio > 0.0:
            if train_ratio + val_ratio <= 0:
                raise ValueError("Train and validation split percentages must be positive.")
            val_fraction = val_ratio / (train_ratio + val_ratio)
            second_split = train_portion.train_test_split(
                test_size=val_fraction,
                seed=seed + 1,
            )
            train_portion = second_split["train"]
            val_portion = second_split["test"]
        else:
            val_portion = None

        new_splits: Dict[str, Dataset] = {"train": train_portion}
        if val_portion is not None:
            new_splits["validation"] = val_portion
        if test_portion is not None:
            new_splits["test"] = test_portion
        dataset = DatasetDict(new_splits)
    else:
        assembled: Dict[str, Dataset] = {}
        if train_split and train_split in dataset:
            assembled["train"] = dataset[train_split]
        if validation_split and validation_split in dataset:
            assembled["validation"] = dataset[validation_split]
        if test_split and test_split in dataset:
            assembled["test"] = dataset[test_split]
        dataset = DatasetDict(assembled)

    dataset, audio_column_name = _ensure_audio_column(dataset, preferred=preferred_audio_column)

    normalize_fn = partial(_normalize_answers, answer_column=target_answer_column)
    dataset = dataset.map(normalize_fn)

    effective_num_proc = resolve_num_proc(num_proc)
    map_kwargs = num_proc_map_kwargs(effective_num_proc)
    duration_fn = partial(add_duration, audio_column=audio_column_name)
    dataset = dataset.map(duration_fn, **map_kwargs)

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

    payload = load_cached_split(cache_path) if cache_splits and not force_rebuild else None
    payload_seed = seed

    splits_metadata: Dict[str, Dict[str, Any]]
    if payload is None:
        split_targets = {
            "train": (max_train_samples, seed),
            "validation": (max_validation_samples, seed + 1),
            "test": (max_test_samples, seed + 2),
        }

        splits_metadata = {}
        for split_name, (target_count, split_seed) in split_targets.items():
            if split_name not in dataset:
                continue
            split_ds = dataset[split_name]
            indices = select_random_indices(len(split_ds), target_count, split_seed)
            splits_metadata[split_name] = build_split_metadata(
                split_ds,
                indices,
                manifest_fields=MANIFEST_FIELDS,
                audio_column=audio_column_name,
            )

        if cache_splits:
            save_cached_split(
                cache_path,
                {
                    "seed": seed,
                    "dataset": dataset_name,
                    "config": dataset_config,
                    "audio_column": audio_column_name,
                    "splits": splits_metadata,
                },
            )
    else:
        splits_metadata = normalize_split_metadata(payload.get("splits"))
        payload_seed = int(payload.get("seed", seed))

    def _select(split_name: str) -> Optional[Dataset]:
        if split_name not in dataset:
            return None
        return subset_dataset_by_metadata(dataset[split_name], splits_metadata.get(split_name))

    train_subset = _select("train")
    validation_subset = _select("validation")
    test_subset = _select("test")

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

    def _summarize(split_name: str, subset: Optional[Dataset], label: Optional[str] = None) -> str:
        if subset is None:
            return f"{label or split_name}=∅"
        hours = float(splits_metadata.get(split_name, {}).get("hours", 0.0) or 0.0)
        return f"{label or split_name}={len(subset)} (~{hours:.2f} h)"

    summary = ", ".join(
        _summarize(split_name, subset, display)
        for split_name, subset, display in (
            ("train", train_subset, "train"),
            ("validation", validation_subset, "val"),
            ("test", test_subset, "test"),
        )
    )

    print(
        "❓ Speech-QA dataset:"
        f" {summary} (seed={payload_seed}, num_proc={effective_num_proc})."
    )

    return train_subset, validation_subset, test_subset, answers_map


@dataclass
class SpeechQACollator:
    """Prepare batches for speech question answering fine-tuning."""

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

    def _build_prompt(self, feature: Dict[str, Any]) -> str:
        question = str(feature.get("question", "")).strip()
        transcript = str(feature.get("transcript", "") or "").strip()
        context = str(feature.get("context", "") or "").strip()

        prompt = (
            f"{self.processor.audio_token}"
            "Listen to the audio segment and answer the question."
        )
        if question:
            prompt += f"\nQuestion: {question}"
        if transcript and self.include_transcript:
            prompt += f"\nTranscript: {transcript}"
        if context and self.include_context:
            prompt += f"\nContext: {context}"
        prompt += "\nAnswer:"
        return prompt

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_arrays = [feature["audio"]["array"] for feature in features]
        labels = [self._select_label(feature) for feature in features]
        prompts = [self._build_prompt(feature) for feature in features]

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "padding_side", None) != "left":
            tokenizer.padding_side = "left"

        full_texts = [
            f"{prompt} {label}".strip()
            for prompt, label in zip(prompts, labels)
        ]

        inputs = self.processor(
            audio=audio_arrays,
            sampling_rate=self.sampling_rate,
            text=full_texts,
            return_tensors="pt",
            padding=True,
        )

        label_ids = inputs["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            label_ids = label_ids.masked_fill(label_ids == pad_id, -100)

        audio_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.audio_token
        )
        if audio_token_id is not None:
            label_ids = label_ids.masked_fill(label_ids == audio_token_id, -100)

        prompt_token_ids = self.processor.tokenizer(
            prompts,
            add_special_tokens=False,
        )["input_ids"]

        for row_idx, tokens in enumerate(prompt_token_ids):
            prompt_len = len(tokens)
            if prompt_len > 0:
                label_ids[row_idx, :prompt_len] = -100

        inputs["labels"] = label_ids
        return inputs


__all__ = ["load_speech_qa_dataset", "SpeechQACollator"]
