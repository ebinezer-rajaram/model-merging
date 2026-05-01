#!/usr/bin/env python3
"""Accurate ASR pairwise comparison under real evaluation conditions.

This script intentionally reuses the same evaluation setup as production ASR eval:
- same config loading (`configs/asr.yaml`)
- same dataset loader/filtering/collator path (`prepare_task_for_evaluation`)
- same model loading path (`load_model_and_processor`)
- same generation path (`CustomTrainer` + generation kwargs from config)

It then adds post-hoc per-sample comparison outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

from _repo import find_repo_root
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from jiwer import (
    Compose,
    ReduceToListOfListOfWords,
    RemoveMultipleSpaces,
    RemovePunctuation,
    Strip,
    ToLowerCase,
    process_words,
    wer_default,
    wer_standardize,
)
from transformers import IntervalStrategy, TrainingArguments

PACKAGE_ROOT = find_repo_root(__file__)
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core import compute_wer_from_texts, load_config, load_model_and_processor, prepare_task_for_evaluation
from core.training.trainer import CustomTrainer
from core.evaluation.evaluate_task import get_model_path, prepare_dataset_cache
from tasks.asr import get_artifact_directories as get_asr_artifact_directories
from tasks.asr import get_config_path as get_asr_config_path
from merging.evaluation.evaluate import _merge_in_memory


DEFAULT_BASELINE_ADAPTER = (
    "artifacts/asr/adapters/qwen2_5_omni_lora_asr_100h/best"
)


@dataclass
class ModelEvalResult:
    label: str
    kind: str
    metrics: Dict[str, Any]
    references: List[str]
    hypotheses: List[str]
    sample_ids: List[str]
    source: str


@dataclass
class EvalModelSpec:
    label: str
    kind: str  # "adapter" | "merged_run"
    path: Path
    source: str


def _sanitize_label(text: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "model"


def _load_default_6task_targets() -> List[EvalModelSpec]:
    script_path = PACKAGE_ROOT / "scripts/merging/build_merge_comparison_6task_merge_itself.py"
    spec = importlib.util.spec_from_file_location("build_merge_comparison_6task_merge_itself", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load default target definitions from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    inputs_by_method = getattr(module, "INPUTS_BY_METHOD", None)
    method_order = getattr(module, "METHOD_ORDER", None)
    if not isinstance(inputs_by_method, dict):
        raise ValueError("INPUTS_BY_METHOD not found or invalid in default target script.")
    if not isinstance(method_order, list):
        method_order = list(inputs_by_method.keys())

    specs: List[EvalModelSpec] = []
    for method in method_order:
        entries = inputs_by_method.get(method, [])
        if not entries:
            continue
        path = Path(entries[0])
        run_dir = (PACKAGE_ROOT / path).resolve().parent
        merge_metadata_path = run_dir / "merge_metadata.json"
        if not merge_metadata_path.exists():
            raise FileNotFoundError(
                f"Default target for '{method}' does not have merge metadata: {merge_metadata_path}"
            )
        specs.append(
            EvalModelSpec(
                label=_sanitize_label(method),
                kind="merged_run",
                path=run_dir,
                source=f"default_6task:{method}",
            )
        )
    return specs


def _resolve_custom_spec(raw: str, label: Optional[str]) -> EvalModelSpec:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (PACKAGE_ROOT / p).resolve()

    if p.is_file() and p.name.startswith("eval_results") and p.suffix == ".json":
        run_dir = p.parent
        if (run_dir / "merge_metadata.json").exists():
            return EvalModelSpec(
                label=_sanitize_label(label or run_dir.name),
                kind="merged_run",
                path=run_dir,
                source=f"custom_eval_json:{p}",
            )

    if p.is_dir():
        if (p / "adapter_model.safetensors").exists() and (p / "adapter_config.json").exists():
            return EvalModelSpec(
                label=_sanitize_label(label or p.name),
                kind="adapter",
                path=p,
                source=f"custom_adapter:{p}",
            )
        if (p / "merge_metadata.json").exists():
            return EvalModelSpec(
                label=_sanitize_label(label or p.name),
                kind="merged_run",
                path=p,
                source=f"custom_merged_run:{p}",
            )

    raise ValueError(
        f"Unsupported custom model path: {raw}. "
        "Expected an adapter dir, merged run dir, or eval_results_*.json under a merged run."
    )


def _get_generation_kwargs(config: Mapping[str, Any]) -> Dict[str, Any]:
    training_cfg = config.get("training", {}) if isinstance(config, Mapping) else {}
    return dict(training_cfg.get("generation_kwargs", {}))


def _resolve_batch_size(config: Mapping[str, Any], override: Optional[int]) -> int:
    if override is not None:
        return int(override)
    eval_cfg = config.get("evaluation", {}) if isinstance(config, Mapping) else {}
    if isinstance(eval_cfg, Mapping) and "per_device_eval_batch_size" in eval_cfg:
        return int(eval_cfg["per_device_eval_batch_size"])
    training_cfg = config.get("training", {}) if isinstance(config, Mapping) else {}
    return int(training_cfg.get("per_device_eval_batch_size", 4))


def _decode_asr_texts(preds: Any, labels: Any, processor: Any) -> Tuple[List[str], List[str]]:
    preds_arr = np.array(preds[0] if isinstance(preds, tuple) else preds)
    labels_arr = np.array(labels[0] if isinstance(labels, tuple) else labels)

    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id or 0
    preds_arr = np.where((preds_arr < 0) | ~np.isfinite(preds_arr), pad_id, preds_arr).astype(np.int64)
    labels_arr = np.where((labels_arr < 0) | ~np.isfinite(labels_arr), pad_id, labels_arr).astype(np.int64)

    pred_texts_raw = processor.batch_decode(preds_arr, skip_special_tokens=True)
    label_texts_raw = processor.batch_decode(labels_arr, skip_special_tokens=True)

    def _clean(text: str) -> str:
        text = str(text).strip()
        if "assistant\n" in text:
            text = text.split("assistant\n", 1)[1].strip()
        return text

    pred_texts = [_clean(x) for x in pred_texts_raw]
    label_texts = [_clean(x) for x in label_texts_raw]
    return label_texts, pred_texts


def _normalize_metric_keys(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in metrics.items():
        if k.startswith("test_"):
            out[k.replace("test_", "", 1)] = v
        elif k.startswith("eval_"):
            out[k.replace("eval_", "", 1)] = v
        else:
            out[k] = v
    return out


def _prepare_replay_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    replay = dict(params)
    optimizer = replay.get("optimizer")
    if isinstance(optimizer, Mapping):
        optimizer_type = str(optimizer.get("type", "none")).strip().lower()
        if optimizer_type != "none":
            replay["optimizer"] = {
                "type": "none",
                "params": {
                    "replay_from_merge_metadata": True,
                    "original_optimizer_type": optimizer_type,
                },
            }
    return replay


def _build_merged_delta_from_run(run_dir: Path) -> Dict[str, torch.Tensor]:
    metadata_path = run_dir / "merge_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing merge metadata: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    method = str(metadata.get("merge_method", "")).strip()
    merge_mode = str(metadata.get("merge_mode", "common"))
    source_adapters = metadata.get("source_adapters", [])
    if not method or not isinstance(source_adapters, list) or not source_adapters:
        raise ValueError(f"Invalid merge metadata in {metadata_path}")

    adapter_paths: List[Path] = []
    for item in source_adapters:
        path = item.get("path")
        if not path:
            raise ValueError(f"Missing source adapter path in {metadata_path}")
        adapter_paths.append(Path(path))

    source_metadata = source_adapters

    raw_params = metadata.get("method_params")
    if not isinstance(raw_params, Mapping):
        raw_params = metadata.get("params", {})
    if not isinstance(raw_params, Mapping):
        raw_params = {}
    replay_params = _prepare_replay_params(raw_params)

    merged_delta, _, _ = _merge_in_memory(
        method=method,
        adapter_paths=adapter_paths,
        source_metadata=source_metadata,
        params=replay_params,
        merge_mode=merge_mode,
    )
    return merged_delta


def _run_eval_with_transcripts(
    *,
    model_path: Path,
    config: Dict[str, Any],
    split: str,
    batch_size: int,
    adapter_path: Optional[Path],
    delta_weights: Optional[Dict[str, torch.Tensor]],
    max_samples: Optional[int],
) -> Tuple[Dict[str, Any], List[str], List[str], List[str]]:
    model, processor = load_model_and_processor(
        model_path,
        adapter_path=adapter_path,
        delta_weights=delta_weights,
    )

    setup = prepare_task_for_evaluation("asr", processor=processor, split=split, config=config)
    if max_samples is not None:
        setup.dataset = setup.dataset.select(range(min(int(max_samples), len(setup.dataset))))

    generation_kwargs = _get_generation_kwargs(config)
    output_dir = PACKAGE_ROOT / "artifacts" / "analysis_tmp" / "asr_pairwise_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=batch_size,
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
        eval_dataset=setup.dataset,
        data_collator=setup.data_collator,
        processing_class=processor,
        compute_metrics=setup.compute_metrics,
        generation_kwargs=generation_kwargs,
    )

    pred_output = trainer.predict(setup.dataset)
    metrics = _normalize_metric_keys(dict(pred_output.metrics))
    refs, hyps = _decode_asr_texts(pred_output.predictions, pred_output.label_ids, processor)
    sample_ids = [str(i) for i in range(len(refs))]

    # Ensure WER recomputation from saved transcripts matches the pipeline metric.
    normalization = str(config.get("metrics", {}).get("wer_normalization", "default"))
    if "wer" in metrics:
        recomputed = compute_wer_from_texts(refs, hyps, normalization=normalization)
        if abs(float(metrics["wer"]) - float(recomputed)) > 1e-9:
            raise RuntimeError(
                f"WER mismatch under normalization='{normalization}': "
                f"metrics={metrics['wer']} recomputed={recomputed}"
            )

    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics, refs, hyps, sample_ids


def _extract_ops(
    ref: str,
    hyp: str,
    normalization: str,
) -> Tuple[int, int, int, List[Dict[str, str]], List[str], List[str]]:
    transform = _resolve_wer_transform(normalization)
    m = process_words(
        [ref],
        [hyp],
        reference_transform=transform,
        hypothesis_transform=transform,
    )
    sub = int(getattr(m, "substitutions", 0))
    ins = int(getattr(m, "insertions", 0))
    dele = int(getattr(m, "deletions", 0))
    sub_details: List[Dict[str, str]] = []
    ins_details: List[str] = []
    del_details: List[str] = []

    alignments = list(getattr(m, "alignments", []) or [])
    references = list(getattr(m, "references", []) or [])
    hypotheses = list(getattr(m, "hypotheses", []) or [])

    for idx, chunks in enumerate(alignments):
        ref_words = references[idx] if idx < len(references) else []
        hyp_words = hypotheses[idx] if idx < len(hypotheses) else []
        for chunk in chunks:
            op_type = str(getattr(chunk, "type", ""))
            rs = int(getattr(chunk, "ref_start_idx", 0))
            re = int(getattr(chunk, "ref_end_idx", 0))
            hs = int(getattr(chunk, "hyp_start_idx", 0))
            he = int(getattr(chunk, "hyp_end_idx", 0))
            ref_span = [str(x) for x in ref_words[rs:re]]
            hyp_span = [str(x) for x in hyp_words[hs:he]]
            if op_type == "substitute":
                sub_details.append({"ref": " ".join(ref_span), "hyp": " ".join(hyp_span)})
            elif op_type == "insert":
                ins_details.append(" ".join(hyp_span))
            elif op_type == "delete":
                del_details.append(" ".join(ref_span))
    return sub, ins, dele, sub_details, ins_details, del_details


def _resolve_wer_transform(normalization: str):
    key = str(normalization).strip().lower()
    if key == "aggressive":
        return Compose(
            [
                RemovePunctuation(),
                ToLowerCase(),
                RemoveMultipleSpaces(),
                Strip(),
                ReduceToListOfListOfWords(),
            ]
        )
    if key == "standardize":
        return wer_standardize
    return wer_default


def _normalized_text(text: str, normalization: str) -> str:
    transform = _resolve_wer_transform(normalization)
    transformed = transform([str(text)])
    if (
        isinstance(transformed, list)
        and transformed
        and isinstance(transformed[0], list)
    ):
        return " ".join(str(tok) for tok in transformed[0])
    if isinstance(transformed, list) and transformed:
        return str(transformed[0])
    return str(text)


def _compare_pair(
    *,
    name_a: str,
    name_b: str,
    refs: Sequence[str],
    hyps_a: Sequence[str],
    hyps_b: Sequence[str],
    normalization: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if len(refs) != len(hyps_a) or len(refs) != len(hyps_b):
        raise ValueError("Reference/hypothesis lengths mismatch in pairwise comparison.")

    per_sample: List[Dict[str, Any]] = []
    improved = 0
    worsened = 0
    unchanged = 0
    changed_hyp = 0
    sub_pair_delta: Counter[Tuple[str, str]] = Counter()
    ins_token_delta: Counter[str] = Counter()
    del_token_delta: Counter[str] = Counter()

    for idx, (ref, ha, hb) in enumerate(zip(refs, hyps_a, hyps_b)):
        wer_a = float(compute_wer_from_texts([ref], [ha], normalization=normalization))
        wer_b = float(compute_wer_from_texts([ref], [hb], normalization=normalization))
        delta = wer_b - wer_a

        if delta < 0:
            improved += 1
        elif delta > 0:
            worsened += 1
        else:
            unchanged += 1
        if ha != hb:
            changed_hyp += 1
        norm_ref = _normalized_text(ref, normalization)
        norm_ha = _normalized_text(ha, normalization)
        norm_hb = _normalized_text(hb, normalization)
        norm_changed = int(norm_ha != norm_hb)

        sub_a, ins_a, del_a, subd_a, insd_a, deld_a = _extract_ops(ref, ha, normalization)
        sub_b, ins_b, del_b, subd_b, insd_b, deld_b = _extract_ops(ref, hb, normalization)

        c_a = Counter((d["ref"], d["hyp"]) for d in subd_a)
        c_b = Counter((d["ref"], d["hyp"]) for d in subd_b)
        sub_pair_delta.update(c_b)
        sub_pair_delta.subtract(c_a)

        c_ins_a = Counter(insd_a)
        c_ins_b = Counter(insd_b)
        ins_token_delta.update(c_ins_b)
        ins_token_delta.subtract(c_ins_a)

        c_del_a = Counter(deld_a)
        c_del_b = Counter(deld_b)
        del_token_delta.update(c_del_b)
        del_token_delta.subtract(c_del_a)

        per_sample.append(
            {
                "sample_id": str(idx),
                "reference": ref,
                "reference_normalized": norm_ref,
                f"hyp_{name_a}": ha,
                f"hyp_{name_b}": hb,
                f"hyp_{name_a}_normalized": norm_ha,
                f"hyp_{name_b}_normalized": norm_hb,
                f"wer_{name_a}": wer_a,
                f"wer_{name_b}": wer_b,
                "wer_delta_b_minus_a": delta,
                f"subs_{name_a}": sub_a,
                f"ins_{name_a}": ins_a,
                f"dels_{name_a}": del_a,
                f"subs_{name_b}": sub_b,
                f"ins_{name_b}": ins_b,
                f"dels_{name_b}": del_b,
                "hyp_changed": int(ha != hb),
                "hyp_changed_after_wer_normalization": norm_changed,
            }
        )

    summary = {
        "model_a": name_a,
        "model_b": name_b,
        "num_samples": len(per_sample),
        "improved_samples": improved,
        "worsened_samples": worsened,
        "unchanged_samples": unchanged,
        "changed_hypothesis_samples": changed_hyp,
        "normalization_used_for_alignment": normalization,
        "changed_hypothesis_samples_after_wer_normalization": int(
            sum(int(row["hyp_changed_after_wer_normalization"]) for row in per_sample)
        ),
        "top_substitution_delta": [
            {"ref": k[0], "hyp": k[1], "delta_count_b_minus_a": v}
            for k, v in sub_pair_delta.most_common(20)
            if v != 0
        ],
        "top_insertion_delta": [
            {"token_span": k, "delta_count_b_minus_a": v}
            for k, v in ins_token_delta.most_common(20)
            if v != 0
        ],
        "top_deletion_delta": [
            {"token_span": k, "delta_count_b_minus_a": v}
            for k, v in del_token_delta.most_common(20)
            if v != 0
        ],
    }
    return per_sample, summary


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["empty"])
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _digest_texts(items: Sequence[str]) -> str:
    joined = "\n".join(items).encode("utf-8")
    return hashlib.md5(joined).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ASR comparisons under real eval conditions and export per-sample diffs."
    )
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default="analysis/results/asr_pairwise",
        help="Output directory for comparison artifacts.",
    )
    parser.add_argument("--model-a", default=None, help="Optional custom model A path.")
    parser.add_argument("--model-b", default=None, help="Optional custom model B path.")
    parser.add_argument("--label-a", default=None, help="Optional custom label for model A.")
    parser.add_argument("--label-b", default=None, help="Optional custom label for model B.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = (PACKAGE_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    asr_config_path = get_asr_config_path(PACKAGE_ROOT, None)
    config = load_config(asr_config_path)
    artifact_dirs = get_asr_artifact_directories(PACKAGE_ROOT)
    config = prepare_dataset_cache(config, artifact_dirs)
    model_path = get_model_path(config, "asr")
    batch_size = _resolve_batch_size(config, args.batch_size)
    normalization = str(config.get("metrics", {}).get("wer_normalization", "default"))

    if bool(args.model_a) ^ bool(args.model_b):
        raise ValueError("Both --model-a and --model-b must be provided together.")

    if args.model_a and args.model_b:
        baseline_spec = _resolve_custom_spec(args.model_a, args.label_a or "model_a")
        target_specs = [_resolve_custom_spec(args.model_b, args.label_b or "model_b")]
    else:
        baseline_spec = EvalModelSpec(
            label="asr_best",
            kind="adapter",
            path=(PACKAGE_ROOT / DEFAULT_BASELINE_ADAPTER).resolve(),
            source="default_baseline",
        )
        target_specs = _load_default_6task_targets()
        if not target_specs:
            raise RuntimeError("No default 6-task targets resolved.")

    all_specs = [baseline_spec] + target_specs
    results: Dict[str, ModelEvalResult] = {}
    ref_signature: Optional[str] = None
    ref_texts: Optional[List[str]] = None

    for spec in all_specs:
        print(f"\n=== Evaluating {spec.label} ({spec.kind}) ===")
        if spec.kind == "adapter":
            metrics, refs, hyps, sample_ids = _run_eval_with_transcripts(
                model_path=model_path,
                config=config,
                split=args.split,
                batch_size=batch_size,
                adapter_path=spec.path,
                delta_weights=None,
                max_samples=args.max_samples,
            )
        elif spec.kind == "merged_run":
            delta = _build_merged_delta_from_run(spec.path)
            metrics, refs, hyps, sample_ids = _run_eval_with_transcripts(
                model_path=model_path,
                config=config,
                split=args.split,
                batch_size=batch_size,
                adapter_path=None,
                delta_weights=delta,
                max_samples=args.max_samples,
            )
        else:
            raise ValueError(f"Unknown spec kind: {spec.kind}")

        current_sig = _digest_texts(refs)
        if ref_signature is None:
            ref_signature = current_sig
            ref_texts = list(refs)
        else:
            if ref_signature != current_sig:
                raise RuntimeError(
                    f"Reference mismatch for model '{spec.label}'. "
                    "This violates same-sample comparison guarantees."
                )

        model_out = out_dir / f"transcripts_{_sanitize_label(spec.label)}.csv"
        rows = []
        for sid, ref, hyp in zip(sample_ids, refs, hyps):
            rows.append({"sample_id": sid, "reference": ref, "hypothesis": hyp})
        _write_csv(model_out, rows)

        results[spec.label] = ModelEvalResult(
            label=spec.label,
            kind=spec.kind,
            metrics=metrics,
            references=list(refs),
            hypotheses=list(hyps),
            sample_ids=list(sample_ids),
            source=spec.source,
        )
        print(
            f"WER={metrics.get('wer')} runtime={metrics.get('runtime')} "
            f"samples_per_second={metrics.get('samples_per_second')}"
        )

    baseline = results[baseline_spec.label]
    summary_rows: List[Dict[str, Any]] = []

    for target_spec in target_specs:
        target = results[target_spec.label]
        pair_name = f"{_sanitize_label(baseline.label)}__vs__{_sanitize_label(target.label)}"
        per_sample, op_summary = _compare_pair(
            name_a=_sanitize_label(baseline.label),
            name_b=_sanitize_label(target.label),
            refs=baseline.references,
            hyps_a=baseline.hypotheses,
            hyps_b=target.hypotheses,
            normalization=normalization,
        )

        per_sample_sorted = sorted(per_sample, key=lambda r: float(r["wer_delta_b_minus_a"]))
        best_examples = per_sample_sorted[:20]
        worst_examples = per_sample_sorted[-20:]

        per_sample_path = out_dir / f"per_sample_{pair_name}.csv"
        op_json_path = out_dir / f"op_breakdown_{pair_name}.json"
        examples_json_path = out_dir / f"examples_{pair_name}.json"

        _write_csv(per_sample_path, per_sample)
        with op_json_path.open("w", encoding="utf-8") as f:
            json.dump(op_summary, f, indent=2)
        with examples_json_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "pair": pair_name,
                    "normalization": normalization,
                    "best_improvements": best_examples,
                    "worst_regressions": worst_examples,
                },
                f,
                indent=2,
            )

        summary_rows.append(
            {
                "pair": pair_name,
                "split": args.split,
                "normalization": normalization,
                "num_samples": len(per_sample),
                "wer_baseline": baseline.metrics.get("wer"),
                "wer_target": target.metrics.get("wer"),
                "wer_delta_target_minus_baseline": (
                    float(target.metrics.get("wer")) - float(baseline.metrics.get("wer"))
                    if "wer" in baseline.metrics and "wer" in target.metrics
                    else None
                ),
                "runtime_baseline": baseline.metrics.get("runtime"),
                "runtime_target": target.metrics.get("runtime"),
                "samples_per_second_baseline": baseline.metrics.get("samples_per_second"),
                "samples_per_second_target": target.metrics.get("samples_per_second"),
                "improved_samples": op_summary["improved_samples"],
                "worsened_samples": op_summary["worsened_samples"],
                "unchanged_samples": op_summary["unchanged_samples"],
                "changed_hypothesis_samples": op_summary["changed_hypothesis_samples"],
                "baseline_source": baseline.source,
                "target_source": target.source,
            }
        )

    summary_path = out_dir / "summary.csv"
    _write_csv(summary_path, summary_rows)

    meta_path = out_dir / "run_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "split": args.split,
                "batch_size": batch_size,
                "max_samples": args.max_samples,
                "normalization": normalization,
                "asr_config_path": str(asr_config_path),
                "model_path": str(model_path),
                "reference_hash": ref_signature,
                "models": [
                    {
                        "label": spec.label,
                        "kind": spec.kind,
                        "path": str(spec.path),
                        "source": spec.source,
                    }
                    for spec in all_specs
                ],
            },
            f,
            indent=2,
        )

    print(f"\nSaved comparison summary to: {summary_path}")
    print(f"Saved run metadata to: {meta_path}")


if __name__ == "__main__":
    main()
