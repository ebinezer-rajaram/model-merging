#!/usr/bin/env python3
"""Run qualitative Speech-QA eval on a subset and print predictions vs references."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import evaluate
from transformers import IntervalStrategy, TrainingArguments

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core import ensure_dir, load_config, load_model_and_processor, prepare_task_for_evaluation
from core.evaluation.eval_utils import DEFAULT_GENERATION_KWARGS
from core.training.trainer import CustomTrainer
from tasks.speech_qa.metrics import compute_speech_qa_metrics
from tasks.speech_qa import TASK_NAME as SPEECH_QA_TASK_NAME, get_artifact_directories, get_config_path


def _clean_text(text: str) -> str:
    text = str(text or "").strip()
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
    training_cfg = config.get("training", {})
    return int(training_cfg.get("per_device_eval_batch_size", 4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qualitative Speech-QA eval on a subset.")
    parser.add_argument("--config", default=None, help="Optional config filename override.")
    parser.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to run.")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before taking the subset.")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device batch size.")
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter path to evaluate.")
    parser.add_argument("--output", default=None, help="Optional JSONL output path.")
    parser.add_argument("--no-print", action="store_true", help="Disable per-sample printing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = get_config_path(PACKAGE_ROOT, args.config)
    config = load_config(config_path)
    artifact_dirs = get_artifact_directories(PACKAGE_ROOT)

    model_path = (PACKAGE_ROOT / config.get("model", {}).get("path", "data/models/Qwen2.5-Omni-3B")).resolve()
    adapter_path = Path(args.adapter).expanduser().resolve() if args.adapter else None
    model, processor = load_model_and_processor(model_path, adapter_path=adapter_path)

    eval_setup = prepare_task_for_evaluation(
        SPEECH_QA_TASK_NAME,
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

    questions = dataset["question"] if "question" in dataset.column_names else [""] * len(dataset)
    references = dataset["answers"] if "answers" in dataset.column_names else [[] for _ in range(len(dataset))]
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
        group_by_length=False,
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
    normalized_references: List[List[str]] = []
    for ref in references:
        if isinstance(ref, (list, tuple)):
            cleaned = [str(item).strip() for item in ref if str(item).strip()]
        else:
            cleaned = [str(ref).strip()] if str(ref).strip() else []
        normalized_references.append(cleaned or [""])

    official = compute_speech_qa_metrics(
        (outputs.predictions, outputs.label_ids),
        processor=processor,
        reference_answers=normalized_references,
        reference_ids=[str(item) if item is not None else "" for item in ids],
        reference_questions=[str(item) for item in questions],
        split_name=args.split,
    )
    squad_metric = evaluate.load("squad")

    output_path: Optional[Path]
    if args.output:
        output_path = Path(args.output).expanduser()
        if not output_path.is_absolute():
            output_path = output_dir / output_path
    else:
        tag = "adapter" if adapter_path is not None else "base"
        output_path = output_dir / f"{args.split}_{tag}_{len(pred_texts)}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for idx, pred in enumerate(pred_texts):
            refs = normalized_references[idx]
            single_result = squad_metric.compute(
                predictions=[{"id": str(idx), "prediction_text": pred}],
                references=[
                    {
                        "id": str(idx),
                        "answers": {
                            "text": refs,
                            "answer_start": [0] * max(1, len(refs)),
                        },
                    }
                ],
            )
            em = float(single_result.get("exact_match", 0.0))
            f1 = float(single_result.get("f1", 0.0))

            if em >= 100.0:
                verdict = "exact"
            elif f1 >= 50.0:
                verdict = "good"
            elif f1 >= 20.0:
                verdict = "partial"
            else:
                verdict = "miss"

            entry = {
                "index": idx,
                "id": ids[idx],
                "question": str(questions[idx]),
                "references": refs,
                "prediction": pred,
                "exact_match": em,
                "f1": f1,
                "verdict": verdict,
            }
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

            if not args.no_print:
                print(f"[{idx}] {verdict.upper()}  f1={f1:.1f}  em={em:.1f}")
                print(f"  Q   : {questions[idx]}")
                print(f"  REF : {refs}")
                print(f"  PRED: {pred}")

    print(
        f"✅ Wrote {len(pred_texts)} samples to {output_path} | "
        f"mean_em={float(official.get('exact_match', 0.0)):.4f} "
        f"mean_f1={float(official.get('f1', 0.0)):.4f}"
    )


if __name__ == "__main__":
    main()
