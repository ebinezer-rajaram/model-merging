"""Dataset helpers and data collator for VocalSound classification."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from datasets import Audio, ClassLabel, Dataset, DatasetDict

from core import resolve_num_proc
from core.tasks.collator import build_strict_label_mask
from core.tasks.dataset import (
    _normalize_target_count,
    _samples_key,
    add_duration_to_dataset,
    cache_and_sample_splits,
    filter_by_duration,
    print_dataset_summary,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATASET_CACHE_ROOT = PACKAGE_ROOT / "artifacts" / "vocalsound" / "datasets"
DEFAULT_DATA_DIR = PACKAGE_ROOT / "data" / "datasets" / "vocalsound"

DEFAULT_TRAIN_MANIFEST = "vocalsound_train_data.json"
DEFAULT_VALIDATION_MANIFEST = "vocalsound_valid_data.json"
DEFAULT_TEST_MANIFEST = "vocalsound_eval_data.json"
DEFAULT_LABEL_MAP_FILE = "class_labels_indices_vs.csv"

FALLBACK_TRAIN_MANIFESTS: Tuple[str, ...] = ("datafiles/tr.json", "tr.json")
FALLBACK_VALIDATION_MANIFESTS: Tuple[str, ...] = ("datafiles/val.json", "val.json")
FALLBACK_TEST_MANIFESTS: Tuple[str, ...] = ("datafiles/te.json", "te.json")
FALLBACK_LABEL_MAP_FILES: Tuple[str, ...] = ("labels/class_labels_indices_vs.csv",)

PATH_KEY_CANDIDATES: Tuple[str, ...] = (
    "wav",
    "audio",
    "audio_path",
    "path",
    "wav_path",
    "file",
    "filename",
)
LABEL_KEY_CANDIDATES: Tuple[str, ...] = (
    "labels",
    "label",
    "mid",
    "class",
    "class_name",
    "category",
    "display_name",
    "sound",
    "target",
    "label_index",
    "index",
)

LABEL_NAME_CANDIDATES: Tuple[str, ...] = ("display_name", "label", "class_name", "name")
LABEL_INDEX_CANDIDATES: Tuple[str, ...] = ("index", "class_index", "id", "label_id")
LABEL_ALIAS_CANDIDATES: Tuple[str, ...] = ("mid", "m_id", "class_mid", "class_id")

MANIFEST_FIELDS: Tuple[str, ...] = (
    "label",
    "label_name",
    "audio_path",
)


def _normalize_label_token(value: str) -> str:
    normalized = re.sub(r"[^0-9a-z]+", "", str(value).lower())
    return normalized.strip()


def _resolve_path(
    *,
    data_dir: Path,
    requested: str,
    fallback_candidates: Sequence[str],
    default_value: str,
    kind: str,
) -> Path:
    """Resolve manifest/CSV path with default-aware fallbacks."""
    requested_path = Path(requested)

    if requested_path.is_absolute():
        if not requested_path.exists():
            raise FileNotFoundError(f"{kind} not found: {requested_path}")
        return requested_path

    if requested != default_value:
        explicit_path = data_dir / requested_path
        if not explicit_path.exists():
            raise FileNotFoundError(
                f"{kind} not found at {explicit_path}. "
                f"Pass a valid path via the corresponding dataset config field."
            )
        return explicit_path

    candidates = [requested] + list(fallback_candidates)
    for candidate in candidates:
        candidate_path = data_dir / candidate
        if candidate_path.exists():
            return candidate_path

    joined = "\n  - ".join(str(data_dir / item) for item in candidates)
    raise FileNotFoundError(f"{kind} not found. Checked:\n  - {joined}")


def _load_label_schema(label_map_path: Path) -> Tuple[List[str], Dict[str, int]]:
    with label_map_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Label map file is empty: {label_map_path}")

    extracted: List[Tuple[int, str, List[str]]] = []
    for row in rows:
        label_name = None
        for key in LABEL_NAME_CANDIDATES:
            value = row.get(key)
            if value is not None and str(value).strip():
                label_name = str(value).strip()
                break
        if not label_name:
            continue

        label_index = None
        for key in LABEL_INDEX_CANDIDATES:
            value = row.get(key)
            if value is None or str(value).strip() == "":
                continue
            try:
                label_index = int(float(value))
                break
            except (TypeError, ValueError):
                continue
        if label_index is None:
            label_index = len(extracted)

        aliases: List[str] = [label_name]
        for key in LABEL_ALIAS_CANDIDATES:
            raw_alias = row.get(key)
            if raw_alias is None:
                continue
            alias = str(raw_alias).strip()
            if alias:
                aliases.append(alias)
        extracted.append((int(label_index), label_name, aliases))

    if not extracted:
        raise ValueError(
            f"Could not parse labels from {label_map_path}. "
            f"Expected one of columns: {LABEL_NAME_CANDIDATES}."
        )

    extracted.sort(key=lambda item: item[0])
    label_names = [name for _, name, _ in extracted]
    label_lookup: Dict[str, int] = {}
    for index, _, aliases in extracted:
        for alias in aliases:
            token = _normalize_label_token(alias)
            if token:
                label_lookup[token] = index
    return label_names, label_lookup


def _read_manifest_entries(manifest_path: Path) -> List[Dict[str, Any]]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        for key in ("data", "items", "entries", "samples"):
            value = payload.get(key)
            if isinstance(value, list):
                entries = value
                break
        else:
            values = list(payload.values())
            if values and all(isinstance(item, dict) for item in values):
                entries = values
            else:
                raise ValueError(
                    f"Unsupported manifest structure in {manifest_path}. "
                    "Expected a list or dict containing a list."
                )
    else:
        raise ValueError(f"Unsupported manifest payload type in {manifest_path}: {type(payload)!r}")

    normalized_entries: List[Dict[str, Any]] = []
    for item in entries:
        if isinstance(item, dict):
            normalized_entries.append(item)
    return normalized_entries


def _build_cache_namespace(
    *,
    train_manifest_path: Path,
    validation_manifest_path: Path,
    test_manifest_path: Path,
    label_map_path: Path,
    audio_root: Optional[Path],
) -> str:
    fingerprint = "|".join(
        (
            str(train_manifest_path.resolve()),
            str(validation_manifest_path.resolve()),
            str(test_manifest_path.resolve()),
            str(label_map_path.resolve()),
            str(audio_root.resolve()) if audio_root is not None else "",
        )
    )
    digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:12]
    return f"vocalsound_local_{digest}"


def _extract_value(
    entry: Mapping[str, Any],
    *,
    preferred_key: Optional[str],
    fallback_keys: Iterable[str],
) -> Any:
    if preferred_key:
        return entry.get(preferred_key)
    for key in fallback_keys:
        if key in entry:
            return entry.get(key)
    return None


def _resolve_audio_path(raw_path: Any, *, manifest_path: Path, audio_root: Optional[Path]) -> Optional[Path]:
    if raw_path is None:
        return None
    value = str(raw_path).strip()
    if not value:
        return None

    path = Path(value).expanduser()
    candidates: List[Path] = []

    if path.is_absolute():
        candidates.append(path)
        if audio_root is not None:
            # Official VocalSound manifests ship absolute source-machine paths.
            # Rebase to local audio root by filename when absolute paths are not portable.
            candidates.append(audio_root / path.name)
    else:
        if audio_root is not None:
            candidates.append(audio_root / path)
            candidates.append(audio_root / path.name)
        candidates.append(manifest_path.parent / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
        if candidate.suffix == "":
            with_wav = candidate.with_suffix(".wav")
            if with_wav.exists():
                return with_wav.resolve()

    return None


def _map_label_to_index(raw_label: Any, label_names: Sequence[str], label_lookup: Mapping[str, int]) -> Optional[int]:
    if raw_label is None:
        return None

    if isinstance(raw_label, bool):
        return None

    if isinstance(raw_label, (int, float)):
        index = int(raw_label)
        if 0 <= index < len(label_names):
            return index
        return None

    text = str(raw_label).strip()
    if not text:
        return None

    if re.fullmatch(r"-?\d+(\.0+)?", text):
        index = int(float(text))
        if 0 <= index < len(label_names):
            return index
        return None

    token = _normalize_label_token(text)
    if token in label_lookup:
        return label_lookup[token]
    if text in label_lookup:
        return label_lookup[text]
    return None


def _prepare_split_records(
    *,
    manifest_path: Path,
    label_names: Sequence[str],
    label_lookup: Mapping[str, int],
    audio_root: Optional[Path],
    path_key: Optional[str],
    label_key: Optional[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    entries = _read_manifest_entries(manifest_path)

    records: List[Dict[str, Any]] = []
    stats = {"missing_path": 0, "missing_label": 0, "missing_audio_file": 0}
    unknown_labels: Dict[str, int] = {}

    for idx, entry in enumerate(entries):
        raw_path = _extract_value(entry, preferred_key=path_key, fallback_keys=PATH_KEY_CANDIDATES)
        raw_label = _extract_value(entry, preferred_key=label_key, fallback_keys=LABEL_KEY_CANDIDATES)

        if raw_path is None:
            stats["missing_path"] += 1
            continue
        if raw_label is None:
            stats["missing_label"] += 1
            continue

        label_index = _map_label_to_index(raw_label, label_names, label_lookup)
        if label_index is None:
            label_token = str(raw_label).strip()
            unknown_labels[label_token] = unknown_labels.get(label_token, 0) + 1
            continue

        resolved_audio_path = _resolve_audio_path(
            raw_path,
            manifest_path=manifest_path,
            audio_root=audio_root,
        )
        if resolved_audio_path is None:
            stats["missing_audio_file"] += 1
            continue

        row: Dict[str, Any] = {
            "audio": str(resolved_audio_path),
            "audio_path": str(resolved_audio_path),
            "label": int(label_index),
            "label_name": str(label_names[label_index]),
        }
        for key, value in entry.items():
            if key in row or key in {path_key, label_key}:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[key] = value
            else:
                row[key] = str(value)
        row["row_index"] = int(idx)
        records.append(row)

    if unknown_labels:
        sample_items = ", ".join(f"{k} ({v})" for k, v in list(unknown_labels.items())[:8])
        raise ValueError(
            f"Manifest {manifest_path} contains labels not present in label map "
            f"{list(label_names)}. Examples: {sample_items}"
        )

    return records, stats


def _build_dataset_from_records(records: List[Dict[str, Any]], split_name: str) -> Dataset:
    if not records:
        raise ValueError(f"No usable samples for split '{split_name}'.")
    return Dataset.from_list(records)


def _cast_label_and_audio(dataset: Dataset, label_names: Sequence[str]) -> Dataset:
    dataset = dataset.cast_column("label", ClassLabel(names=list(label_names)))
    return dataset.cast_column("audio", Audio(sampling_rate=16000))


def load_vocalsound_dataset(
    *,
    dataset_name: str = "vocalsound",
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
    label_column: Optional[str] = None,
    text_column: Optional[str] = None,
    audio_column: Optional[str] = None,
    split_percentages: Optional[Mapping[str, float] | Sequence[float]] = None,
    train_split: str = "train",
    validation_split: Optional[str] = "validation",
    test_split: Optional[str] = "test",
    stratify_by_column: Optional[str] = None,
    data_dir: Optional[str] = None,
    train_manifest: str = DEFAULT_TRAIN_MANIFEST,
    validation_manifest: str = DEFAULT_VALIDATION_MANIFEST,
    test_manifest: str = DEFAULT_TEST_MANIFEST,
    label_map_file: str = DEFAULT_LABEL_MAP_FILE,
    audio_root: Optional[str] = None,
    path_key: Optional[str] = None,
    label_key: Optional[str] = None,
) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset], List[str]]:
    """Load VocalSound dataset from official local manifests."""
    del dataset_name, dataset_config, label_column, text_column, audio_column
    del split_percentages, train_split, validation_split, test_split, stratify_by_column

    max_train_samples = _normalize_target_count("max_train_samples", max_train_samples)
    max_validation_samples = _normalize_target_count("max_validation_samples", max_validation_samples)
    max_test_samples = _normalize_target_count("max_test_samples", max_test_samples)

    data_root = Path(data_dir).expanduser() if data_dir else DEFAULT_DATA_DIR
    if not data_root.is_absolute():
        data_root = (PACKAGE_ROOT / data_root).resolve()

    train_manifest_path = _resolve_path(
        data_dir=data_root,
        requested=str(train_manifest),
        fallback_candidates=FALLBACK_TRAIN_MANIFESTS,
        default_value=DEFAULT_TRAIN_MANIFEST,
        kind="VocalSound train manifest",
    )
    validation_manifest_path = _resolve_path(
        data_dir=data_root,
        requested=str(validation_manifest),
        fallback_candidates=FALLBACK_VALIDATION_MANIFESTS,
        default_value=DEFAULT_VALIDATION_MANIFEST,
        kind="VocalSound validation manifest",
    )
    test_manifest_path = _resolve_path(
        data_dir=data_root,
        requested=str(test_manifest),
        fallback_candidates=FALLBACK_TEST_MANIFESTS,
        default_value=DEFAULT_TEST_MANIFEST,
        kind="VocalSound test manifest",
    )
    label_map_path = _resolve_path(
        data_dir=data_root,
        requested=str(label_map_file),
        fallback_candidates=FALLBACK_LABEL_MAP_FILES,
        default_value=DEFAULT_LABEL_MAP_FILE,
        kind="VocalSound label map CSV",
    )

    label_names, label_lookup = _load_label_schema(label_map_path)
    if len(label_names) != 6:
        print(
            f"⚠️ VocalSound label map contains {len(label_names)} classes (expected 6). "
            "Proceeding with CSV-defined classes."
        )

    inferred_audio_root = None
    if audio_root:
        inferred_audio_root = Path(audio_root).expanduser()
        if not inferred_audio_root.is_absolute():
            inferred_audio_root = (data_root / inferred_audio_root).resolve()
    else:
        for candidate in ("audio_16k", "audio", "wav", "waves"):
            candidate_path = data_root / candidate
            if candidate_path.exists():
                inferred_audio_root = candidate_path.resolve()
                break

    train_records, train_stats = _prepare_split_records(
        manifest_path=train_manifest_path,
        label_names=label_names,
        label_lookup=label_lookup,
        audio_root=inferred_audio_root,
        path_key=path_key,
        label_key=label_key,
    )
    validation_records, validation_stats = _prepare_split_records(
        manifest_path=validation_manifest_path,
        label_names=label_names,
        label_lookup=label_lookup,
        audio_root=inferred_audio_root,
        path_key=path_key,
        label_key=label_key,
    )
    test_records, test_stats = _prepare_split_records(
        manifest_path=test_manifest_path,
        label_names=label_names,
        label_lookup=label_lookup,
        audio_root=inferred_audio_root,
        path_key=path_key,
        label_key=label_key,
    )

    def _print_split_stats(name: str, stats: Mapping[str, int]) -> None:
        skipped = int(sum(stats.values()))
        if skipped <= 0:
            return
        print(
            f"⚠️ VocalSound {name}: skipped {skipped} rows "
            f"(missing_path={stats.get('missing_path', 0)}, "
            f"missing_label={stats.get('missing_label', 0)}, "
            f"missing_audio_file={stats.get('missing_audio_file', 0)})."
        )

    _print_split_stats("train", train_stats)
    _print_split_stats("validation", validation_stats)
    _print_split_stats("test", test_stats)

    dataset = DatasetDict(
        {
            "train": _build_dataset_from_records(train_records, "train"),
            "validation": _build_dataset_from_records(validation_records, "validation"),
            "test": _build_dataset_from_records(test_records, "test"),
        }
    )

    for split_name in list(dataset.keys()):
        dataset[split_name] = _cast_label_and_audio(dataset[split_name], label_names)

    cache_root = Path(cache_dir) if cache_dir is not None else DATASET_CACHE_ROOT
    cache_root = cache_root / _build_cache_namespace(
        train_manifest_path=train_manifest_path,
        validation_manifest_path=validation_manifest_path,
        test_manifest_path=test_manifest_path,
        label_map_path=label_map_path,
        audio_root=inferred_audio_root,
    )
    effective_num_proc = resolve_num_proc(num_proc)

    dataset = add_duration_to_dataset(
        dataset,
        audio_column="audio",
        num_proc=effective_num_proc,
        cache_dir=cache_root if not force_rebuild else None,
    )

    dataset = filter_by_duration(
        dataset,
        max_duration=max_duration,
        min_duration=min_duration,
        cache_dir=cache_root if not force_rebuild else None,
    )

    cache_name = (
        "vocalsound_local"
        f"_train_{_samples_key(max_train_samples)}"
        f"_val_{_samples_key(max_validation_samples)}"
        f"_test_{_samples_key(max_test_samples)}"
        f"_seed_{int(seed)}.json"
    )
    cache_path = cache_root / cache_name

    train_subset, validation_subset, test_subset, splits_metadata, payload_seed = cache_and_sample_splits(
        dataset,
        cache_path=cache_path,
        max_train_samples=max_train_samples,
        max_validation_samples=max_validation_samples,
        max_test_samples=max_test_samples,
        seed=seed,
        manifest_fields=MANIFEST_FIELDS,
        audio_column="audio",
        cache_splits=cache_splits,
        force_rebuild=force_rebuild,
        additional_metadata={
            "label_names": list(label_names),
            "label_map_file": str(label_map_path),
            "train_manifest": str(train_manifest_path),
            "validation_manifest": str(validation_manifest_path),
            "test_manifest": str(test_manifest_path),
            "audio_root": str(inferred_audio_root) if inferred_audio_root is not None else None,
        },
    )

    print_dataset_summary(
        task_emoji="🔊",
        task_name="VocalSound dataset",
        train_subset=train_subset,
        validation_subset=validation_subset,
        test_subset=test_subset,
        splits_metadata=splits_metadata,
        label_names=list(label_names),
        seed=payload_seed,
        num_proc=effective_num_proc,
    )

    return train_subset, validation_subset, test_subset, list(label_names)


@dataclass
class VocalSoundCollator:
    """Prepare batches for VocalSound classification."""

    processor: Any
    sampling_rate: int
    label_names: Sequence[str]
    warn_on_label_mask_fallback: bool = True

    def _label_to_text(self, value: Any) -> str:
        if value is None:
            return ""
        try:
            index = int(value)
        except (TypeError, ValueError):
            return str(value)
        if 0 <= index < len(self.label_names):
            return str(self.label_names[index])
        return str(index)

    def _build_instruction(self) -> str:
        class_options = ", ".join(self.label_names)
        return (
            "Identify the human vocal sound in the audio. "
            f"Choose from: {class_options}. Output only the label."
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        valid_features: List[Dict[str, Any]] = []
        for feature in features:
            try:
                _ = feature["audio"]["array"]
                valid_features.append(feature)
            except (RuntimeError, Exception) as exc:
                print(f"Warning: Skipping corrupted audio sample: {exc}")
                continue

        if not valid_features:
            raise RuntimeError("All audio samples in batch are corrupted")

        audio_arrays = [feature["audio"]["array"] for feature in valid_features]
        label_strings = [self._label_to_text(feature.get("label")) for feature in valid_features]

        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "padding_side", None) != "left":
            tokenizer.padding_side = "left"

        prompts = []
        for label in label_strings:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": None},
                        {"type": "text", "text": self._build_instruction()},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": label},
                    ],
                },
            ]
            prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=False,
                tokenize=False,
            )
            prompts.append(prompt)

        inputs = self.processor(
            audio=audio_arrays,
            sampling_rate=self.sampling_rate,
            text=prompts,
            return_tensors="pt",
            padding=True,
        )

        labels = inputs["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        audio_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.audio_token
        )

        labels = labels.masked_fill(labels == pad_id, -100)
        labels = labels.masked_fill(labels == audio_token_id, -100)

        for row_idx, label in enumerate(label_strings):
            label_tokens = tokenizer.encode(label, add_special_tokens=False)
            input_ids = inputs["input_ids"][row_idx]
            mask_stats = build_strict_label_mask(
                input_ids=input_ids,
                labels=labels[row_idx],
                label_tokens=label_tokens,
                ignore_index=-100,
            )
            if self.warn_on_label_mask_fallback and bool(mask_stats.get("used_fallback", False)):
                reason = str(mask_stats.get("fallback_reason"))
                print(
                    "[vocalsound-collator] label mask fallback used "
                    f"(reason={reason}, label='{label}', kept={mask_stats.get('kept_token_count', 0)})"
                )

        inputs["labels"] = labels
        return inputs


__all__ = ["load_vocalsound_dataset", "VocalSoundCollator"]
