#!/usr/bin/env python3
"""Run a small qualitative ST eval and dump predictions vs references."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import sacrebleu
from transformers import IntervalStrategy, TrainingArguments

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core import ensure_dir, load_config, load_model_and_processor, prepare_task_for_evaluation
from core.evaluation.eval_utils import DEFAULT_GENERATION_KWARGS
from core.training.trainer import CustomTrainer
from tasks.st import TASK_NAME as ST_TASK_NAME, get_artifact_directories, get_config_path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _clean_text(text: str) -> str:
    text = text.strip()
    if "assistant\n" in text:
        text = text.split("assistant\n", 1)[1].strip()
    return text


def _decode_predictions(processor: Any, preds: Any) -> List[str]:
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.array(preds)
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id or 0
    preds = np.where((preds < 0) | ~np.isfinite(preds), pad_id, preds).astype(np.int64)
    pred_texts_raw = processor.batch_decode(preds, skip_special_tokens=True)
    return [_clean_text(text) for text in pred_texts_raw]


def _resolve_batch_size(config: Dict[str, Any], batch_size: Optional[int]) -> int:
    if batch_size:
        return int(batch_size)
    eval_cfg = config.get("evaluation", {})
    if "per_device_eval_batch_size" in eval_cfg:
        return int(eval_cfg["per_device_eval_batch_size"])
    training_cfg = config.get("training", {})
    return int(training_cfg.get("per_device_eval_batch_size", 4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qualitative ST eval on a small sample.")
    parser.add_argument("--config", default=None, help="Optional config filename override.")
    parser.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to run.")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before taking the subset.")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device batch size.")
    parser.add_argument("--output", default=None, help="Optional JSONL output path.")
    parser.add_argument("--no-print", action="store_true", help="Disable printing to stdout.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = get_config_path(PACKAGE_ROOT, args.config)
    config = load_config(config_path)
    language = config.get("language", "en_de")
    artifact_dirs = get_artifact_directories(PACKAGE_ROOT, language=language)

    model_path = (PACKAGE_ROOT / config.get("model", {}).get("path", "data/models/Qwen2.5-Omni-3B")).resolve()
    model, processor = load_model_and_processor(model_path, adapter_path=None)

    eval_setup = prepare_task_for_evaluation(
        ST_TASK_NAME,
        processor,
        split=args.split,
        config=config,
    )

    dataset = eval_setup.dataset
    if args.shuffle:
        dataset = dataset.shuffle(seed=args.seed)

    if args.num_samples is not None:
        target_count = min(int(args.num_samples), len(dataset))
        dataset = dataset.select(range(target_count))

    refs = dataset["translation"] if "translation" in dataset.column_names else [None] * len(dataset)
    sources = dataset["text"] if "text" in dataset.column_names else [None] * len(dataset)
    ids = dataset["id"] if "id" in dataset.column_names else [None] * len(dataset)

    generation_kwargs = dict(DEFAULT_GENERATION_KWARGS)
    generation_kwargs.update(config.get("training", {}).get("generation_kwargs", {}))

    resolved_batch_size = _resolve_batch_size(config, args.batch_size)
    output_dir = ensure_dir(artifact_dirs["metrics"] / "qualitative")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=resolved_batch_size,
        dataloader_drop_last=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=[],
        eval_strategy=IntervalStrategy.NO,
        save_strategy=IntervalStrategy.NO,
        logging_strategy=IntervalStrategy.NO,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        eval_dataset=dataset,
        data_collator=eval_setup.data_collator,
        processing_class=processor,
        compute_metrics=None,
        generation_kwargs=generation_kwargs,
    )

    outputs = trainer.predict(dataset)
    pred_texts = _decode_predictions(processor, outputs.predictions)

    output_path: Optional[Path] = None
    if args.output:
        output_path = Path(args.output).expanduser()
        if not output_path.is_absolute():
            output_path = output_dir / output_path
    else:
        output_path = output_dir / f"{args.split}_base_{len(pred_texts)}.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as handle:
        for idx, pred in enumerate(pred_texts):
            ref = refs[idx]
            bleu_score = None
            if ref is not None:
                bleu = sacrebleu.sentence_bleu(
                    pred,
                    [str(ref)],
                    lowercase=False,
                    tokenize="13a",
                    smooth_method="exp",
                    smooth_value=0.01,
                )
                bleu_score = float(bleu.score)
            entry = {
                "index": idx,
                "id": ids[idx],
                "source": sources[idx],
                "reference": ref,
                "prediction": pred,
                "bleu": bleu_score,
            }
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

    if not args.no_print:
        print(f"✅ Wrote {len(pred_texts)} predictions to {output_path}")
        for idx, pred in enumerate(pred_texts):
            ref = refs[idx]
            bleu_score = None
            if ref is not None:
                bleu = sacrebleu.sentence_bleu(
                    pred,
                    [str(ref)],
                    lowercase=False,
                    tokenize="13a",
                    smooth_method="exp",
                    smooth_value=0.01,
                )
                bleu_score = float(bleu.score)
            print(f"[{idx}]")
            print(f"  REF : {ref}")
            print(f"  PRED: {pred}")
            print(f"  BLEU: {bleu_score:.4f}" if bleu_score is not None else "  BLEU: None")


if __name__ == "__main__":
    main()
