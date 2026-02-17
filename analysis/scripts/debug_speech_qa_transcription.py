#!/usr/bin/env python3
"""Debug Speech-QA audio by transcribing concatenated audio and scoring against dataset text."""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from tqdm.auto import tqdm

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core import compute_wer_from_texts, load_config, load_model_and_processor
from core.tasks.dataset import apply_split_percentages
from tasks.speech_qa import get_config_path
from tasks.speech_qa.dataset import _load_local_spoken_squad_dataset


def _normalize_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_audio_array(audio_obj: Any) -> Sequence[float]:
    if isinstance(audio_obj, dict):
        array = audio_obj.get("array")
        if array is None:
            raise ValueError("Audio dict missing 'array'.")
        return array
    if hasattr(audio_obj, "get_all_samples"):
        samples = audio_obj.get_all_samples()
        data = getattr(samples, "data", None)
        if data is None:
            raise ValueError("AudioDecoder samples missing 'data'.")
        if hasattr(data, "detach"):
            data = data.detach().cpu()
        if hasattr(data, "numpy"):
            data = data.numpy()
        # Expected shape [channels, time] for torchcodec audio.
        if getattr(data, "ndim", 1) == 2:
            data = data[0]
        return data
    array = getattr(audio_obj, "array", None)
    if array is None:
        raise ValueError(f"Unsupported audio object type: {type(audio_obj)!r}")
    return array


def _infer_reference_column(dataset, requested: Optional[str]) -> str:
    if requested:
        if requested not in dataset.column_names:
            raise ValueError(f"Requested reference column {requested!r} not found in {dataset.column_names}.")
        return requested

    if "transcript" in dataset.column_names:
        transcript_values = dataset["transcript"]
        non_empty = sum(1 for value in transcript_values if str(value or "").strip())
        if non_empty > 0:
            return "transcript"
    if "context" in dataset.column_names:
        return "context"
    raise ValueError("Could not infer reference column. Dataset has neither 'transcript' nor 'context'.")


def _build_loader_kwargs(config: Dict[str, Any], force_concatenate: bool) -> Dict[str, Any]:
    dataset_cfg = dict(config.get("dataset", {}))
    kwargs: Dict[str, Any] = {
        "data_dir": dataset_cfg.get("data_dir"),
        "train_json": dataset_cfg.get("train_json", "spoken_train-v1.1.json"),
        "test_json": dataset_cfg.get("test_json", "spoken_test-v1.1.json"),
        "audio_root": dataset_cfg.get("audio_root"),
        "noisy_test_variant": dataset_cfg.get("noisy_test_variant", "none"),
        "min_wavs_per_split": dataset_cfg.get("min_wavs_per_split", 100),
        "max_missing_audio_rate": dataset_cfg.get("max_missing_audio_rate", 0.01),
        "allow_train_only_fallback": dataset_cfg.get("allow_train_only_fallback", True),
        "audio_merge_policy": (
            "concatenate_sentences"
            if force_concatenate
            else "first_sentence"
        ),
    }
    return kwargs


def _select_split(
    split_name: str,
    train_ds,
    val_ds,
    test_ds,
):
    split_map = {
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    }
    if split_name not in split_map:
        raise ValueError(f"Unsupported split {split_name!r}.")
    split_ds = split_map[split_name]
    if split_ds is None:
        raise ValueError(f"Split {split_name!r} is unavailable.")
    return split_ds


def _batched_indices(total: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    start = 0
    while start < total:
        end = min(total, start + batch_size)
        yield start, end
        start = end


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe Speech-QA concatenated audio and compare against dataset transcript/context."
    )
    parser.add_argument("--config", default=None, help="Optional config filename override.")
    parser.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter path.")
    parser.add_argument(
        "--reference-column",
        default=None,
        choices=("transcript", "context"),
        help="Reference text source. Default: transcript if non-empty else context.",
    )
    parser.add_argument("--no-concatenate", action="store_true", help="Disable forced concatenated audio policy.")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument(
        "--instruction",
        default="Only output the exact transcription of this audio. Do not add explanations.",
    )
    parser.add_argument("--output", default=None, help="Output JSONL path.")
    parser.add_argument("--device", default=None, choices=("cpu", "cuda"), help="Force device.")
    parser.add_argument("--no-print", action="store_true", help="Disable per-sample terminal printing.")
    parser.add_argument("--print-limit", type=int, default=50, help="Max samples to print to terminal.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = get_config_path(PACKAGE_ROOT, args.config)
    config = load_config(config_path)
    model_cfg = config.get("model", {})

    model_path = (PACKAGE_ROOT / model_cfg.get("path", "data/models/Qwen2.5-Omni-3B")).resolve()
    adapter_path = Path(args.adapter).expanduser().resolve() if args.adapter else None
    device_override = args.device
    device_map_override = {"": device_override} if device_override in {"cpu", "cuda"} else None
    dtype_override = torch.float32 if device_override == "cpu" else None
    model, processor = load_model_and_processor(
        model_path,
        adapter_path=adapter_path,
        device_map_override=device_map_override,
        torch_dtype_override=dtype_override,
    )

    dataset_cfg = config.get("dataset", {})
    if str(dataset_cfg.get("dataset_name", "local_spoken_squad")) != "local_spoken_squad":
        raise ValueError(
            "This debug script currently supports dataset_name=local_spoken_squad only."
        )

    loader_kwargs = _build_loader_kwargs(config, force_concatenate=not args.no_concatenate)
    raw_dataset, _ = _load_local_spoken_squad_dataset(**loader_kwargs)

    if "validation" not in raw_dataset and "test" not in raw_dataset:
        split_percentages = dataset_cfg.get("split_percentages", {"train": 0.8, "validation": 0.1, "test": 0.1})
        raw_dataset = apply_split_percentages(
            raw_dataset,
            split_percentages=split_percentages,
            train_split=str(dataset_cfg.get("train_split", "train")),
            seed=int(args.seed),
            stratify_by_column=None,
        )

    train_ds = raw_dataset["train"] if "train" in raw_dataset else None
    val_ds = raw_dataset["validation"] if "validation" in raw_dataset else None
    test_ds = raw_dataset["test"] if "test" in raw_dataset else None
    split_ds = _select_split(args.split, train_ds, val_ds, test_ds)

    if args.shuffle:
        split_ds = split_ds.shuffle(seed=args.seed)
    target_count = min(int(args.num_samples), len(split_ds))
    split_ds = split_ds.select(range(target_count))

    reference_column = _infer_reference_column(split_ds, args.reference_column)
    references = [str(value or "").strip() for value in split_ds[reference_column]]
    ids = [str(value) for value in split_ds["id"]] if "id" in split_ds.column_names else [str(i) for i in range(len(split_ds))]
    questions = [str(value or "") for value in split_ds["question"]] if "question" in split_ds.column_names else ["" for _ in range(len(split_ds))]

    tokenizer = processor.tokenizer
    if getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"

    predictions: List[str] = []

    total_batches = (len(split_ds) + int(args.batch_size) - 1) // int(args.batch_size)
    batch_iterator = _batched_indices(len(split_ds), int(args.batch_size))
    progress = tqdm(batch_iterator, total=total_batches, desc="🧠 Transcribing", unit="batch")
    for start, end in progress:
        batch_rows = [split_ds[idx] for idx in range(start, end)]
        audio_arrays = [_extract_audio_array(row["audio"]) for row in batch_rows]
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": None},
                        {"type": "text", "text": args.instruction},
                    ],
                }
            ]
            for _ in batch_rows
        ]
        prompts = [
            processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            for conv in conversations
        ]

        inputs = processor(
            audio=audio_arrays,
            sampling_rate=getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000),
            text=prompts,
            return_tensors="pt",
            padding=True,
        )
        input_len = inputs["input_ids"].shape[1]
        for key, value in inputs.items():
            if torch.is_tensor(value):
                inputs[key] = value.to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=False,
            )

        new_tokens = generated[:, input_len:]
        batch_preds = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        batch_preds = [str(pred).strip() for pred in batch_preds]
        predictions.extend(batch_preds)
        progress.set_postfix(done=len(predictions), total=len(split_ds))

    wer_default = compute_wer_from_texts(references, predictions, normalization="default")
    wer_standardize = compute_wer_from_texts(references, predictions, normalization="standardize")
    wer_aggressive = compute_wer_from_texts(references, predictions, normalization="aggressive")

    normalized_exact = sum(
        1
        for ref, pred in zip(references, predictions)
        if _normalize_text(ref) == _normalize_text(pred)
    )
    exact_rate = normalized_exact / max(1, len(predictions))

    output_path = (
        Path(args.output).expanduser()
        if args.output
        else PACKAGE_ROOT / "artifacts" / "speech_qa" / "metrics" / "qualitative" / f"{args.split}_transcription_debug_{len(predictions)}.jsonl"
    )
    if not output_path.is_absolute():
        output_path = (PACKAGE_ROOT / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for idx, (sample_id, question, ref, pred) in enumerate(zip(ids, questions, references, predictions)):
            record = {
                "index": idx,
                "id": sample_id,
                "question": question,
                "reference_column": reference_column,
                "reference_text": ref,
                "prediction_text": pred,
                "normalized_exact": float(_normalize_text(ref) == _normalize_text(pred)),
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            if not args.no_print and idx < int(args.print_limit):
                print(f"[{idx}] id={sample_id}")
                print(f"  Q   : {question}")
                print(f"  REF : {ref}")
                print(f"  PRED: {pred}")

    print("=== Speech-QA Transcription Debug ===")
    print(f"split: {args.split}")
    print(f"samples: {len(predictions)}")
    print(f"reference_column: {reference_column}")
    print(f"audio_merge_policy: {'first_sentence' if args.no_concatenate else 'concatenate_sentences'}")
    print(f"adapter: {str(adapter_path) if adapter_path is not None else 'base'}")
    print(f"wer_default: {wer_default:.4f}")
    print(f"wer_standardize: {wer_standardize:.4f}")
    print(f"wer_aggressive: {wer_aggressive:.4f}")
    print(f"normalized_exact_rate: {exact_rate:.4f}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
