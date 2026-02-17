"""Dataset helpers and data collator for speech question answering tasks."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import re
import wave
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from concurrent.futures import Future, ThreadPoolExecutor

import torch
from datasets import Audio, Dataset, DatasetDict
from tqdm.auto import tqdm

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
    filter_by_duration,
    load_dataset,
    print_dataset_summary,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATASET_CACHE_ROOT = PACKAGE_ROOT / "artifacts" / "speech_qa" / "datasets"

LOCAL_SPOKEN_SQUAD_DATASET_NAME = "local_spoken_squad"
DEFAULT_DATASET_NAME = LOCAL_SPOKEN_SQUAD_DATASET_NAME
DEFAULT_LOCAL_DATA_DIR = PACKAGE_ROOT / "data" / "datasets" / "Spoken-SQuAD"
DEFAULT_LOCAL_TRAIN_JSON = "spoken_train-v1.1.json"
DEFAULT_LOCAL_TEST_JSON = "spoken_test-v1.1.json"
NOISY_TEST_JSON_MAP: Dict[str, str] = {
    "wer44": "spoken_test-v1.1_WER44.json",
    "wer54": "spoken_test-v1.1_WER54.json",
}
FALLBACK_QUESTION_COLUMNS: Tuple[str, ...] = ("question", "query", "instruction")
# Keep "context" out of transcript fallbacks to avoid clobbering the QA passage.
FALLBACK_TRANSCRIPT_COLUMNS: Tuple[str, ...] = ("transcript", "text", "sentence")
FALLBACK_CONTEXT_COLUMNS: Tuple[str, ...] = ("context", "passage", "paragraph")
FALLBACK_ID_COLUMNS: Tuple[str, ...] = ("id", "qid", "example_id")
FALLBACK_ANSWER_COLUMNS: Tuple[str, ...] = ("answers", "answer")
AUDIO_MERGE_POLICIES: Tuple[str, ...] = ("first_sentence", "concatenate_sentences")

MANIFEST_FIELDS: Tuple[str, ...] = (
    "id",
    "question",
    "label_text",
    "answers",
    "context",
    "transcript",
    "path",
)


def _allocate_total_samples(
    dataset: DatasetDict,
    max_total_samples: int,
) -> Dict[str, Optional[int]]:
    """Allocate a global sample budget across available splits by split sizes."""
    split_names = [name for name in ("train", "validation", "test") if name in dataset]
    if not split_names:
        return {"train": None, "validation": None, "test": None}

    split_sizes = {name: len(dataset[name]) for name in split_names}
    total_available = sum(split_sizes.values())
    if total_available <= 0:
        return {"train": 0, "validation": 0, "test": 0}

    target_total = min(int(max_total_samples), total_available)
    if target_total <= 0:
        return {"train": 0, "validation": 0, "test": 0}

    # Largest-remainder allocation to preserve total exactly.
    raw = {
        name: (target_total * split_sizes[name] / total_available)
        for name in split_names
    }
    base = {name: int(raw[name]) for name in split_names}
    remainder = target_total - sum(base.values())
    if remainder > 0:
        order = sorted(
            split_names,
            key=lambda name: (raw[name] - base[name], split_sizes[name]),
            reverse=True,
        )
        idx = 0
        while remainder > 0:
            chosen = order[idx % len(order)]
            if base[chosen] < split_sizes[chosen]:
                base[chosen] += 1
                remainder -= 1
            idx += 1

    allocation = {"train": None, "validation": None, "test": None}
    for name in split_names:
        allocation[name] = min(base[name], split_sizes[name])
    return allocation


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


def _resolve_local_spoken_squad_paths(
    *,
    data_dir: Path,
    train_json: str,
    test_json: str,
    noisy_test_variant: str,
    audio_root: Optional[Path],
) -> Dict[str, Path]:
    """Resolve local Spoken-SQuAD json/audio paths and validate required files."""
    root = data_dir.resolve()
    if not root.exists():
        raise FileNotFoundError(f"local_spoken_squad data_dir does not exist: {root}")

    train_json_path = (root / train_json).resolve()
    test_json_name = NOISY_TEST_JSON_MAP.get(noisy_test_variant, test_json)
    test_json_path = (root / test_json_name).resolve()

    if not train_json_path.exists():
        raise FileNotFoundError(f"Spoken-SQuAD train json not found: {train_json_path}")
    if not test_json_path.exists():
        raise FileNotFoundError(f"Spoken-SQuAD test json not found: {test_json_path}")

    root_audio = (audio_root or (root / "wav")).resolve()
    train_audio_dir = (root_audio / "train").resolve()
    test_audio_dir = (root_audio / "test").resolve()
    return {
        "root": root,
        "train_json": train_json_path,
        "test_json": test_json_path,
        "audio_root": root_audio,
        "train_audio_dir": train_audio_dir,
        "test_audio_dir": test_audio_dir,
    }


def _build_spoken_squad_audio_index(split_audio_dir: Path) -> Dict[Tuple[int, int], List[Tuple[int, Path]]]:
    """Index Spoken-SQuAD wav files by (topic_idx, paragraph_idx)."""
    index: Dict[Tuple[int, int], List[Tuple[int, Path]]] = {}
    if not split_audio_dir.exists():
        return index
    for wav_path in split_audio_dir.rglob("*.wav"):
        match = re.match(r"^(\d+)_(\d+)(?:_(\d+))?$", wav_path.stem)
        if not match:
            continue
        topic_idx = int(match.group(1))
        para_idx = int(match.group(2))
        sent_idx = int(match.group(3)) if match.group(3) is not None else 0
        index.setdefault((topic_idx, para_idx), []).append((sent_idx, wav_path.resolve()))
    for key in index:
        index[key].sort(key=lambda pair: pair[0])
    return index


def _load_spoken_squad_split_from_local(
    *,
    json_path: Path,
    split_name: str,
    audio_index: Dict[Tuple[int, int], List[Tuple[int, Path]]],
    max_missing_audio_rate: float,
    audio_merge_policy: str,
    concatenated_audio_dir: Optional[Path],
) -> Tuple[Dataset, Dict[str, Any]]:
    """Load one Spoken-SQuAD split from local SQuAD-format JSON + wav index."""
    with json_path.open("r") as handle:
        payload = json.load(handle)
    data_items = list(payload.get("data") or [])

    rows: List[Dict[str, Any]] = []
    missing_audio = 0
    total_qas = 0
    multi_sentence_prefixes = 0
    concatenated_audio_files = 0
    concatenation_failures = 0

    zero_based_matches = 0
    one_based_matches = 0
    unresolved_prefixes = 0

    total_paragraphs = sum(len(list(article.get("paragraphs") or [])) for article in data_items)
    show_concat_progress = (audio_merge_policy == "concatenate_sentences") and (total_paragraphs > 0)
    paragraph_progress = tqdm(
        total=total_paragraphs,
        desc=f"🔊 Concatenating {split_name} audio",
        unit="para",
        leave=False,
        disable=not show_concat_progress,
    )
    progress_refresh_interval = 128
    processed_paragraphs = 0

    concat_futures: Dict[Tuple[int, int], Future] = {}
    concat_fallback_path: Dict[Tuple[int, int], Path] = {}
    concat_row_indices: Dict[Tuple[int, int], List[int]] = {}
    concat_success_keys: set[Tuple[int, int]] = set()
    concat_requested_keys: set[Tuple[int, int]] = set()
    concat_executor: Optional[ThreadPoolExecutor] = None
    if audio_merge_policy == "concatenate_sentences":
        max_workers = min(8, max(1, int(os.cpu_count() or 1)))
        concat_executor = ThreadPoolExecutor(max_workers=max_workers)
    if show_concat_progress:
        paragraph_progress.set_postfix(concatenated=0, concat_failures=0, missing_qas=0)

    def _build_concat(target_path: Path, source_wavs: List[Path]) -> Path:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        _concatenate_wavs(source_wavs, target_path)
        return target_path

    for topic_idx, article in enumerate(data_items):
        paragraphs = list(article.get("paragraphs") or [])
        for para_idx, paragraph in enumerate(paragraphs):
            context = str(paragraph.get("context") or "")
            qas = list(paragraph.get("qas") or [])
            # Spoken-SQuAD wav archives are commonly indexed with 0-based prefixes
            # (topic_idx_paragraph_idx[_sentence_idx].wav), while some local exports
            # are 1-based. Resolve 0-based first, then fallback to 1-based.
            zero_based_key = (topic_idx, para_idx)
            one_based_key = (topic_idx + 1, para_idx + 1)
            matched_key: Optional[Tuple[int, int]] = None

            audio_candidates = audio_index.get(zero_based_key, [])
            if audio_candidates:
                matched_key = zero_based_key
                zero_based_matches += 1
            else:
                audio_candidates = audio_index.get(one_based_key, [])
                if audio_candidates:
                    matched_key = one_based_key
                    one_based_matches += 1
                else:
                    unresolved_prefixes += 1

            chosen_audio: Optional[Path] = None
            concat_key_for_rows: Optional[Tuple[int, int]] = None
            if audio_candidates:
                chosen_audio = audio_candidates[0][1]
                if len(audio_candidates) > 1:
                    multi_sentence_prefixes += 1
                    if audio_merge_policy == "concatenate_sentences":
                        if concatenated_audio_dir is None:
                            raise ValueError(
                                "concatenated_audio_dir must be provided when using "
                                "audio_merge_policy='concatenate_sentences'."
                            )
                        if matched_key is None:
                            raise ValueError("matched_key must be resolved when audio candidates exist.")

                        concat_key_for_rows = matched_key
                        concat_requested_keys.add(concat_key_for_rows)
                        concat_fallback_path[concat_key_for_rows] = audio_candidates[0][1]
                        concat_path = concatenated_audio_dir / f"{matched_key[0]}_{matched_key[1]}_concat.wav"
                        if concat_path.exists():
                            concat_success_keys.add(concat_key_for_rows)
                        elif concat_executor is not None and concat_key_for_rows not in concat_futures:
                            concat_futures[concat_key_for_rows] = concat_executor.submit(
                                _build_concat,
                                concat_path,
                                [path for _, path in audio_candidates],
                            )
                        chosen_audio = concat_path.resolve()

            for qa in qas:
                total_qas += 1
                if chosen_audio is None:
                    missing_audio += 1
                    continue
                answers_raw = list(qa.get("answers") or [])
                answers = [
                    str(answer.get("text", "")).strip()
                    for answer in answers_raw
                    if isinstance(answer, Mapping) and str(answer.get("text", "")).strip()
                ]
                if not answers:
                    answers = [""]
                rows.append(
                    {
                        "id": str(qa.get("id") or f"{split_name}_{topic_idx}_{para_idx}_{len(rows)}"),
                        "question": str(qa.get("question") or "").strip(),
                        "answers": answers,
                        "label_text": answers[0],
                        "context": context,
                        "audio": str(chosen_audio),
                    }
                )
                if concat_key_for_rows is not None:
                    concat_row_indices.setdefault(concat_key_for_rows, []).append(len(rows) - 1)

            processed_paragraphs += 1
            if show_concat_progress and (processed_paragraphs % progress_refresh_interval == 0):
                paragraph_progress.set_postfix(
                    concatenated=int(len(concat_success_keys)),
                    concat_failures=int(concatenation_failures),
                    missing_qas=int(missing_audio),
                )
            paragraph_progress.update(1)

    failed_concat_keys: set[Tuple[int, int]] = set()
    for concat_key, future in concat_futures.items():
        try:
            future.result()
            concat_success_keys.add(concat_key)
        except (RuntimeError, ValueError, OSError):
            failed_concat_keys.add(concat_key)

    if concat_executor is not None:
        concat_executor.shutdown(wait=True)

    if failed_concat_keys:
        for concat_key in failed_concat_keys:
            fallback = concat_fallback_path.get(concat_key)
            if fallback is None:
                continue
            for row_idx in concat_row_indices.get(concat_key, []):
                rows[row_idx]["audio"] = str(fallback)
        concatenation_failures = len(failed_concat_keys)

    concatenated_audio_files = len(concat_success_keys)

    if show_concat_progress:
        paragraph_progress.set_postfix(
            concatenated=int(concatenated_audio_files),
            concat_failures=int(concatenation_failures),
            missing_qas=int(missing_audio),
        )
    paragraph_progress.close()

    if total_qas == 0:
        raise ValueError(f"No QA items found in local Spoken-SQuAD split: {json_path}")
    if not rows:
        raise ValueError(
            f"All QA items were dropped for split '{split_name}' because audio files were unresolved."
        )

    missing_rate = float(missing_audio / max(1, total_qas))
    if missing_rate > float(max_missing_audio_rate):
        raise ValueError(
            f"Too many missing audio mappings for split '{split_name}': "
            f"{missing_audio}/{total_qas} ({missing_rate:.2%}) exceeds "
            f"max_missing_audio_rate={max_missing_audio_rate:.2%}."
        )

    if missing_audio > 0:
        print(
            f"⚠️  local_spoken_squad {split_name}: dropped {missing_audio}/{total_qas} QA rows "
            f"({missing_rate:.2%}) due to unresolved audio."
        )
    if multi_sentence_prefixes > 0:
        if audio_merge_policy == "concatenate_sentences":
            print(
                f"ℹ️  local_spoken_squad {split_name}: {multi_sentence_prefixes} paragraph prefixes had "
                "multiple sentence wav files; concatenating sentence wavs per prefix."
            )
            if concatenation_failures > 0:
                print(
                    f"⚠️  local_spoken_squad {split_name}: failed to concatenate "
                    f"{concatenation_failures}/{multi_sentence_prefixes} prefixes; "
                    "fell back to first sentence audio."
                )
        else:
            print(
                f"ℹ️  local_spoken_squad {split_name}: {multi_sentence_prefixes} paragraph prefixes had "
                "multiple sentence wav files; using the first sentence file per prefix."
            )

    split_ds = Dataset.from_list(rows)
    split_ds = split_ds.cast_column("audio", Audio(sampling_rate=16000))
    split_metadata = {
        "split": split_name,
        "source_json": str(json_path),
        "audio_merge_policy": audio_merge_policy,
        "audio_index_key_matches": {
            "zero_based": int(zero_based_matches),
            "one_based": int(one_based_matches),
            "unresolved": int(unresolved_prefixes),
        },
        "total_qas": int(total_qas),
        "rows_loaded": int(len(rows)),
        "dropped_missing_audio": int(missing_audio),
        "missing_audio_rate": float(missing_rate),
        "multi_sentence_prefixes": int(multi_sentence_prefixes),
        "concatenated_audio_files": int(concatenated_audio_files),
        "concatenation_failures": int(concatenation_failures),
    }
    return split_ds, split_metadata


def _concatenate_wavs(source_paths: Sequence[Path], output_path: Path) -> None:
    """Concatenate wav files with identical format into one wav."""
    if not source_paths:
        raise ValueError("source_paths must not be empty")

    params = None
    pcm_chunks: List[bytes] = []
    for path in source_paths:
        with wave.open(str(path), "rb") as handle:
            current_params = (handle.getnchannels(), handle.getsampwidth(), handle.getframerate())
            if params is None:
                params = current_params
            elif current_params != params:
                raise ValueError(
                    "Cannot concatenate wav files with different formats: "
                    f"{params!r} vs {current_params!r}"
                )
            pcm_chunks.append(handle.readframes(handle.getnframes()))

    if params is None:
        raise ValueError("Failed to infer wav params for concatenation.")

    nchannels, sampwidth, framerate = params
    with wave.open(str(output_path), "wb") as out:
        out.setnchannels(int(nchannels))
        out.setsampwidth(int(sampwidth))
        out.setframerate(int(framerate))
        out.writeframes(b"".join(pcm_chunks))


def _load_local_spoken_squad_dataset(
    *,
    data_dir: Optional[Path | str],
    train_json: str,
    test_json: str,
    audio_root: Optional[Path | str],
    noisy_test_variant: str,
    min_wavs_per_split: int,
    max_missing_audio_rate: float,
    allow_train_only_fallback: bool,
    audio_merge_policy: str,
) -> Tuple[DatasetDict, Dict[str, Any]]:
    """Load local Spoken-SQuAD json + wav archives into a DatasetDict."""
    noisy = str(noisy_test_variant or "none").strip().lower()
    if noisy not in {"none", *NOISY_TEST_JSON_MAP.keys()}:
        raise ValueError("noisy_test_variant must be one of: none|wer44|wer54.")
    root = Path(data_dir).expanduser() if data_dir is not None else DEFAULT_LOCAL_DATA_DIR
    resolved = _resolve_local_spoken_squad_paths(
        data_dir=root,
        train_json=train_json,
        test_json=test_json,
        noisy_test_variant=noisy,
        audio_root=(Path(audio_root).expanduser() if audio_root is not None else None),
    )
    train_audio_dir = resolved["train_audio_dir"]
    test_audio_dir = resolved["test_audio_dir"]

    train_wavs = list(train_audio_dir.rglob("*.wav")) if train_audio_dir.exists() else []
    test_wavs = list(test_audio_dir.rglob("*.wav")) if test_audio_dir.exists() else []

    min_wavs = int(min_wavs_per_split)
    if min_wavs < 0:
        raise ValueError("min_wavs_per_split must be >= 0.")
    if len(train_wavs) < min_wavs:
        raise FileNotFoundError(
            "Local Spoken-SQuAD wav files look incomplete. Expected at least "
            f"{min_wavs} wavs per split under '{resolved['audio_root']}'. "
            f"Found train={len(train_wavs)}, test={len(test_wavs)}. "
            "Ensure archives are extracted into wav/train and wav/test."
        )

    use_train_only_source = len(test_wavs) < min_wavs and bool(allow_train_only_fallback)
    if len(test_wavs) < min_wavs and not use_train_only_source:
        raise FileNotFoundError(
            "Local Spoken-SQuAD wav files look incomplete. Expected at least "
            f"{min_wavs} wavs per split under '{resolved['audio_root']}'. "
            f"Found train={len(train_wavs)}, test={len(test_wavs)}. "
            "Ensure archives are extracted into wav/train and wav/test."
        )
    if use_train_only_source:
        print(
            "ℹ️  local_spoken_squad: test wavs missing/incomplete "
            f"(train={len(train_wavs)}, test={len(test_wavs)}). "
            "Using configured train-only fallback (allow_train_only_fallback=true); "
            "this is expected when intentionally deriving train/validation/test from train audio only."
        )

    merge_policy = str(audio_merge_policy or "first_sentence").strip().lower()
    if merge_policy not in AUDIO_MERGE_POLICIES:
        raise ValueError(
            f"audio_merge_policy must be one of {AUDIO_MERGE_POLICIES}, received {audio_merge_policy!r}."
        )
    concatenated_audio_root = resolved["audio_root"] / "_concatenated"

    train_audio_index = _build_spoken_squad_audio_index(train_audio_dir)
    if not train_audio_index:
        raise ValueError(
            "Failed to index local Spoken-SQuAD wav files. Expected filenames like "
            "'TopicIndex_ParagraphIndex[_SentenceIndex].wav'."
        )
    test_audio_index: Dict[Tuple[int, int], List[Tuple[int, Path]]] = {}
    if not use_train_only_source:
        test_audio_index = _build_spoken_squad_audio_index(test_audio_dir)
        if not test_audio_index:
            raise ValueError(
                "Failed to index local Spoken-SQuAD wav files. Expected filenames like "
                "'TopicIndex_ParagraphIndex[_SentenceIndex].wav'."
            )

    print(
        "📦 Loading local Spoken-SQuAD from "
        f"{resolved['root']} (train_wavs={len(train_wavs)}, test_wavs={len(test_wavs)})"
    )

    train_split, train_metadata = _load_spoken_squad_split_from_local(
        json_path=resolved["train_json"],
        split_name="train",
        audio_index=train_audio_index,
        max_missing_audio_rate=max_missing_audio_rate,
        audio_merge_policy=merge_policy,
        concatenated_audio_dir=concatenated_audio_root / "train",
    )
    if use_train_only_source:
        metadata = {
            "split_origin": "train_only_fallback",
            "audio_merge_policy": merge_policy,
            "train": train_metadata,
            "test": None,
        }
        return DatasetDict({"train": train_split}), metadata

    test_split, test_metadata = _load_spoken_squad_split_from_local(
        json_path=resolved["test_json"],
        split_name="test",
        audio_index=test_audio_index,
        max_missing_audio_rate=max_missing_audio_rate,
        audio_merge_policy=merge_policy,
        concatenated_audio_dir=concatenated_audio_root / "test",
    )
    metadata = {
        "split_origin": "standard_train_test",
        "audio_merge_policy": merge_policy,
        "train": train_metadata,
        "test": test_metadata,
    }
    return DatasetDict({"train": train_split, "test": test_split}), metadata


def load_speech_qa_dataset(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_config: Optional[str] = None,
    max_train_samples: Optional[int] = None,
    max_validation_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    max_total_samples: Optional[int] = None,
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
    data_dir: Optional[Path | str] = None,
    train_json: str = DEFAULT_LOCAL_TRAIN_JSON,
    test_json: str = DEFAULT_LOCAL_TEST_JSON,
    audio_root: Optional[Path | str] = None,
    noisy_test_variant: str = "none",
    min_wavs_per_split: int = 100,
    max_missing_audio_rate: float = 0.01,
    allow_train_only_fallback: bool = True,
    audio_merge_policy: str = "first_sentence",
) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset], Dict[str, Any]]:
    """
    Load an audio question answering dataset with optional sub-sampling.

    Returns train, validation, test splits (None when missing) and a mapping with:
    - train/validation/test answer sets
    - _metadata: dataset provenance (for local Spoken-SQuAD).
    """
    # Normalize sample counts
    max_train_samples = _normalize_target_count("max_train_samples", max_train_samples)
    max_validation_samples = _normalize_target_count("max_validation_samples", max_validation_samples)
    max_test_samples = _normalize_target_count("max_test_samples", max_test_samples)
    max_total_samples = _normalize_target_count("max_total_samples", max_total_samples)

    dataset_provenance: Dict[str, Any] = {}
    # Load dataset
    if dataset_name == LOCAL_SPOKEN_SQUAD_DATASET_NAME:
        dataset, dataset_provenance = _load_local_spoken_squad_dataset(
            data_dir=data_dir,
            train_json=train_json,
            test_json=test_json,
            audio_root=audio_root,
            noisy_test_variant=noisy_test_variant,
            min_wavs_per_split=min_wavs_per_split,
            max_missing_audio_rate=max_missing_audio_rate,
            allow_train_only_fallback=allow_train_only_fallback,
            audio_merge_policy=audio_merge_policy,
        )
    elif dataset_config:
        dataset = load_dataset(dataset_name, dataset_config)
    else:
        dataset = load_dataset(dataset_name)

    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected DatasetDict, received {type(dataset)!r}")

    # Ensure audio column first (before other column normalizations)
    # This prevents 'context' from being treated as text context instead of audio
    from core.tasks.dataset import _ensure_audio_column
    dataset, audio_column_name = _ensure_audio_column(dataset, preferred=audio_column)

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

    # Apply split percentages if specified
    if split_percentages:
        from core.tasks.dataset import apply_split_percentages
        dataset = apply_split_percentages(
            dataset,
            split_percentages=split_percentages,
            train_split=train_split,
            seed=seed,
            stratify_by_column=None,  # No stratification for QA
        )
        # After apply_split_percentages, splits are already named correctly
        # Just ensure they exist
        dataset = assemble_splits(
            dataset,
            train_split="train",
            validation_split="validation",
            test_split="test",
        )
    else:
        # Assemble standard splits
        dataset = assemble_splits(
            dataset,
            train_split=train_split,
            validation_split=validation_split,
            test_split=test_split,
        )

    if max_total_samples is not None:
        explicit_any = any(
            value is not None for value in (max_train_samples, max_validation_samples, max_test_samples)
        )
        if explicit_any:
            raise ValueError(
                "Use either max_total_samples or per-split max_*_samples, not both."
            )
        allocation = _allocate_total_samples(dataset, int(max_total_samples))
        max_train_samples = allocation["train"]
        max_validation_samples = allocation["validation"]
        max_test_samples = allocation["test"]

    # Normalize answers
    normalize_fn = partial(_normalize_answers, answer_column=target_answer_column)
    dataset = dataset.map(normalize_fn)

    # Build cache root path
    cache_root = Path(cache_dir) if cache_dir is not None else DATASET_CACHE_ROOT

    # Add duration information (with caching)
    effective_num_proc = resolve_num_proc(num_proc)
    dataset = add_duration_to_dataset(
        dataset,
        audio_column=audio_column_name,
        num_proc=effective_num_proc,
        cache_dir=cache_root if not force_rebuild else None,
    )

    # Filter by duration if specified (with caching)
    dataset = filter_by_duration(
        dataset,
        max_duration=max_duration,
        min_duration=min_duration,
        cache_dir=cache_root if not force_rebuild else None,
    )

    # Build cache path
    dataset_key = dataset_name.replace("/", "_")
    config_key = dataset_config or "default"
    if dataset_name == LOCAL_SPOKEN_SQUAD_DATASET_NAME:
        local_source = {
            "data_dir": str((Path(data_dir).expanduser() if data_dir is not None else DEFAULT_LOCAL_DATA_DIR).resolve()),
            "train_json": str(train_json),
            "test_json": str(test_json),
            "noisy_test_variant": str(noisy_test_variant),
            "audio_merge_policy": str(audio_merge_policy),
        }
        source_hash = hashlib.md5(
            json.dumps(local_source, sort_keys=True).encode("utf-8")
        ).hexdigest()[:10]
        config_key = f"local_{source_hash}"
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
            "data_dir": str(data_dir) if data_dir is not None else None,
            "train_json": train_json if dataset_name == LOCAL_SPOKEN_SQUAD_DATASET_NAME else None,
            "test_json": test_json if dataset_name == LOCAL_SPOKEN_SQUAD_DATASET_NAME else None,
            "noisy_test_variant": noisy_test_variant if dataset_name == LOCAL_SPOKEN_SQUAD_DATASET_NAME else None,
            "audio_merge_policy": audio_merge_policy if dataset_name == LOCAL_SPOKEN_SQUAD_DATASET_NAME else None,
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
    answers_map["_metadata"] = dataset_provenance

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

        instruction = "Copy the shortest exact answer span from the audio. Output only those words."
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
