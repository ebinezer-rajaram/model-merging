#!/usr/bin/env python3
"""Sweep Speech-QA prompt phrasings on a representative subset and rank templates."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

from _repo import find_repo_root
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from transformers import IntervalStrategy, TrainingArguments

PACKAGE_ROOT = find_repo_root(__file__)
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core import ensure_dir, load_config, load_model_and_processor, prepare_task_for_evaluation
from core.evaluation.eval_utils import DEFAULT_GENERATION_KWARGS
from core.training.trainer import CustomTrainer
from tasks.speech_qa import TASK_NAME as SPEECH_QA_TASK_NAME, get_artifact_directories, get_config_path
from tasks.speech_qa.dataset import CHOICE_COLUMNS, SpeechQACollator


@dataclass(frozen=True)
class PromptTemplateSpec:
    name: str
    preamble: str
    include_choices: bool = False
    include_transcript: bool = False
    include_context: bool = False


DEFAULT_PROMPT_TEMPLATES: Sequence[PromptTemplateSpec] = (
    PromptTemplateSpec(
        name="audio_mcq_grounded_v3",
        preamble="Choose the most supported option from the audio evidence. Output only the final answer text.",
        include_choices=True,
    ),
    PromptTemplateSpec(
        name="audio_mcq_grounded_v3_short",
        preamble="Choose the option best supported by the audio. Output only answer text.",
        include_choices=True,
    ),
    PromptTemplateSpec(
        name="audio_mcq_grounded_v3_strict",
        preamble="Choose the option most supported by what is explicitly heard in the audio. Output only the answer text.",
        include_choices=True,
    ),
    PromptTemplateSpec(
        name="audio_mcq_grounded_v3_no_extra",
        preamble="Use audio evidence to select one option. Return only the exact option text with no extra words.",
        include_choices=True,
    ),
    PromptTemplateSpec(
        name="audio_mcq_grounded_v1",
        preamble="Select the answer that best matches the audio. Base your answer only on what is heard.",
        include_choices=True,
    ),
    PromptTemplateSpec(
        name="audio_mcq_grounded_v4",
        preamble="Answer strictly from the audio. If choices are provided, pick the one most consistent with the audio and output its text.",
        include_choices=True,
    ),
    PromptTemplateSpec(
        name="audio_mcq_grounded_v3_transcript",
        preamble="Choose the option best supported by the audio (and transcript if available). Output only answer text.",
        include_choices=True,
        include_transcript=True,
    ),
    PromptTemplateSpec(
        name="audio_mcq_elimination",
        preamble="Use the audio to eliminate incorrect options, then output the best answer text.",
        include_choices=True,
    ),
    PromptTemplateSpec(
        name="audio_mcq_evidence_first",
        preamble="Identify key evidence in the audio, then choose the best option. Output only answer text.",
        include_choices=True,
    ),
    PromptTemplateSpec(
        name="audio_mcq_disambiguate",
        preamble="When options are similar, pick the one most directly supported by the audio. Return only answer text.",
        include_choices=True,
    ),
    PromptTemplateSpec(
        name="audio_mcq_no_hallucination",
        preamble="Select an option only if supported by the audio evidence. Output only the option text.",
        include_choices=True,
    ),
    PromptTemplateSpec(
        name="audio_mcq_grounded_transcript",
        preamble="Use the audio (and transcript if available) plus choices to produce the best answer text.",
        include_choices=True,
        include_transcript=True,
    ),
)



def _clean_text(text: str) -> str:
    value = str(text or "").strip()
    if "assistant\n" in value:
        value = value.split("assistant\n", 1)[1].strip()
    return value


def _decode_predictions(processor: Any, preds: Any) -> List[str]:
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.array(preds)
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id or 0
    preds = np.where((preds < 0) | ~np.isfinite(preds), pad_id, preds).astype(np.int64)
    pred_texts_raw = processor.batch_decode(preds, skip_special_tokens=True)
    return [_clean_text(text) for text in pred_texts_raw]


def _question_bucket(question: str) -> str:
    text = str(question or "").strip().lower()
    match = re.match(r"^([a-z]+)", text)
    first = match.group(1) if match else ""
    wh_words = {"what", "which", "who", "whom", "whose", "when", "where", "why", "how"}
    if first in wh_words:
        return first
    if first in {"is", "are", "was", "were", "do", "does", "did", "can", "could", "will", "would", "has", "have"}:
        return "aux_yes_no"
    return "other"


def _allocate_counts(bucket_sizes: Mapping[str, int], total_target: int) -> Dict[str, int]:
    names = [name for name, size in bucket_sizes.items() if size > 0]
    if not names:
        return {}
    total_available = sum(bucket_sizes[name] for name in names)
    total_target = min(int(total_target), total_available)
    if total_target <= 0:
        return {name: 0 for name in names}

    raw = {name: total_target * bucket_sizes[name] / total_available for name in names}
    base = {name: int(math.floor(raw[name])) for name in names}
    remainder = total_target - sum(base.values())
    order = sorted(names, key=lambda name: (raw[name] - base[name], bucket_sizes[name]), reverse=True)
    idx = 0
    while remainder > 0 and order:
        choice = order[idx % len(order)]
        if base[choice] < bucket_sizes[choice]:
            base[choice] += 1
            remainder -= 1
        idx += 1
    return base


def _select_representative_indices(dataset: Any, *, sample_count: int, seed: int) -> tuple[List[int], Dict[str, int], str]:
    buckets: Dict[str, List[int]] = {}
    stratify_key = "task_name"
    if "task_name" in dataset.column_names:
        task_names = dataset["task_name"]
        for idx, task_name in enumerate(task_names):
            bucket = str(task_name or "").strip() or "unknown"
            buckets.setdefault(bucket, []).append(idx)
    else:
        stratify_key = "question_type"
        questions = dataset["question"] if "question" in dataset.column_names else ["" for _ in range(len(dataset))]
        for idx, question in enumerate(questions):
            bucket = _question_bucket(str(question))
            buckets.setdefault(bucket, []).append(idx)

    allocation = _allocate_counts({name: len(idxs) for name, idxs in buckets.items()}, sample_count)
    rng = random.Random(seed)
    selected: List[int] = []
    selected_counts: Dict[str, int] = {}
    for bucket_name, target in allocation.items():
        candidates = list(buckets.get(bucket_name, []))
        rng.shuffle(candidates)
        chosen = sorted(candidates[:target])
        selected.extend(chosen)
        selected_counts[bucket_name] = len(chosen)

    selected = sorted(set(selected))
    if len(selected) < min(sample_count, len(dataset)):
        remaining = [idx for idx in range(len(dataset)) if idx not in set(selected)]
        rng.shuffle(remaining)
        need = min(sample_count, len(dataset)) - len(selected)
        selected.extend(sorted(remaining[:need]))
    return sorted(selected), selected_counts, stratify_key


class PromptTemplateSpeechQACollator(SpeechQACollator):
    """Speech-QA collator that swaps only the instruction wording."""

    def __init__(self, *, template: PromptTemplateSpec, **kwargs):
        super().__init__(**kwargs)
        self.template = template

    def _build_instruction(self, feature: Dict[str, Any]) -> str:
        question = str(feature.get("question", "")).strip()
        transcript = str(feature.get("transcript", "") or "").strip()
        context = str(feature.get("context", "") or "").strip()

        lines: List[str] = []
        if self.template.preamble:
            lines.append(self.template.preamble)
        lines.append(f"Question: {question}" if question else "Question:")

        if self.template.include_choices:
            choice_lines: List[str] = []
            for label, key in zip(("A", "B", "C", "D"), CHOICE_COLUMNS):
                choice_value = str(feature.get(key, "") or "").strip()
                if choice_value:
                    choice_lines.append(f"{label}. {choice_value}")
            if choice_lines:
                lines.append("Choices:\n" + "\n".join(choice_lines))
            elif context:
                lines.append("Choices:\n" + context)

        if self.template.include_transcript and transcript:
            lines.append(f"Transcript: {transcript}")
        if self.template.include_context and context:
            lines.append(f"Context: {context}")
        return "\n".join(lines)


def _resolve_batch_size(config: Dict[str, Any], batch_size: Optional[int]) -> int:
    if batch_size:
        return int(batch_size)
    return int(config.get("training", {}).get("per_device_eval_batch_size", 4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep Speech-QA prompt templates.")
    parser.add_argument("--config", default=None, help="Optional config filename override.")
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--num-samples", type=int, default=250, help="Representative sample size.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device eval batch size.")
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter path to evaluate.")
    parser.add_argument(
        "--templates",
        nargs="*",
        default=None,
        help="Template names to run (default: all built-ins).",
    )
    parser.add_argument("--save-json", default=None, help="Optional output json path.")
    parser.add_argument("--save-jsonl", action="store_true", help="Save per-sample predictions per template.")
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
    selected_indices, selected_counts, stratify_key = _select_representative_indices(
        dataset,
        sample_count=max(1, int(args.num_samples)),
        seed=int(args.seed),
    )
    subset = dataset.select(selected_indices)

    if eval_setup.apply_subset_indices is not None:
        eval_setup.apply_subset_indices(selected_indices)

    generation_kwargs = dict(DEFAULT_GENERATION_KWARGS)
    generation_kwargs.update(config.get("training", {}).get("generation_kwargs", {}))
    resolved_batch_size = _resolve_batch_size(config, args.batch_size)
    output_dir = ensure_dir(artifact_dirs["metrics"] / "prompt_sweep")

    template_specs = [spec for spec in DEFAULT_PROMPT_TEMPLATES]
    if args.templates:
        requested = set(args.templates)
        template_specs = [spec for spec in template_specs if spec.name in requested]
        missing = sorted(requested.difference({spec.name for spec in template_specs}))
        if missing:
            raise ValueError(f"Unknown template names: {missing}")
    if not template_specs:
        raise ValueError("No templates selected.")

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

    results: List[Dict[str, Any]] = []
    questions = subset["question"] if "question" in subset.column_names else ["" for _ in range(len(subset))]
    references = subset["answers"] if "answers" in subset.column_names else [[] for _ in range(len(subset))]
    ids = subset["id"] if "id" in subset.column_names else [None for _ in range(len(subset))]

    print(f"Using subset size={len(subset)} split={args.split} seed={args.seed}")
    print(f"Stratification key: {stratify_key}")
    print(f"Strata distribution: {json.dumps(selected_counts, sort_keys=True)}")

    for spec in template_specs:
        collator = PromptTemplateSpeechQACollator(
            processor=processor,
            sampling_rate=getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000),
            template=spec,
            include_transcript=bool(spec.include_transcript),
            include_context=bool(spec.include_context),
            include_choices_in_prompt=bool(spec.include_choices),
        )
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            eval_dataset=subset,
            data_collator=collator,
            processing_class=processor,
            compute_metrics=None,
            generation_kwargs=generation_kwargs,
        )
        outputs = trainer.predict(subset)
        metric_scores = eval_setup.compute_metrics((outputs.predictions, outputs.label_ids))
        pred_texts = _decode_predictions(processor, outputs.predictions)
        row = {
            "template": spec.name,
            "preamble": spec.preamble,
            "include_choices": bool(spec.include_choices),
            "num_samples": int(len(subset)),
            "exact_match": float(metric_scores.get("exact_match", 0.0)),
            "f1": float(metric_scores.get("f1", 0.0)),
        }
        results.append(row)
        print(
            f"[{spec.name}] EM={row['exact_match']:.3f} F1={row['f1']:.3f} "
            f"(choices={row['include_choices']})"
        )

        if args.save_jsonl:
            pred_path = output_dir / f"{spec.name}_{args.split}_{len(subset)}.jsonl"
            with pred_path.open("w", encoding="utf-8") as handle:
                for idx, pred in enumerate(pred_texts):
                    refs = references[idx] if idx < len(references) else []
                    handle.write(
                        json.dumps(
                            {
                                "index": int(idx),
                                "id": ids[idx] if idx < len(ids) else None,
                                "question": str(questions[idx]) if idx < len(questions) else "",
                                "gold_answers": list(refs) if isinstance(refs, (list, tuple)) else [str(refs)],
                                "prediction": pred,
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )

    ranked = sorted(results, key=lambda item: (item["f1"], item["exact_match"]), reverse=True)
    summary = {
        "config_path": str(config_path),
        "split": args.split,
        "num_samples": int(len(subset)),
        "seed": int(args.seed),
        "adapter_path": str(adapter_path) if adapter_path is not None else None,
        "stratify_key": stratify_key,
        "strata_distribution": selected_counts,
        "results": ranked,
        "best_template": ranked[0]["template"] if ranked else None,
    }

    save_json = Path(args.save_json).expanduser() if args.save_json else (output_dir / "prompt_sweep_summary.json")
    if not save_json.is_absolute():
        save_json = output_dir / save_json
    save_json.parent.mkdir(parents=True, exist_ok=True)
    save_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Ranked templates ===")
    for idx, row in enumerate(ranked, start=1):
        print(f"{idx:>2}. {row['template']:<24} F1={row['f1']:.3f} EM={row['exact_match']:.3f}")
    print(f"\nSaved summary to: {save_json}")


if __name__ == "__main__":
    main()
