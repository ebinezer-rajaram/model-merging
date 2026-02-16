"""Shared interference-metric helpers used by evaluation and optimizers."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Tuple

from core import compute_eval_subset_tag
from merging.runtime.utils import PACKAGE_ROOT

EPS = 1e-8
TASK_METRICS: Dict[str, Tuple[str, bool]] = {
    "asr": ("wer", False),
    "emotion": ("macro_f1", True),
    "intent": ("accuracy", True),
    "speech_qa": ("f1", True),
    "kws": ("macro_f1", True),
    "langid": ("accuracy", True),
    "speaker_id": ("accuracy", True),
    "speaker_ver": ("accuracy", True),
}


def with_eval_tag(filename: str, eval_tag: Optional[str]) -> str:
    if not eval_tag:
        return filename
    stem, dot, ext = filename.rpartition(".")
    if not dot:
        stem, ext = filename, "json"
    return f"{stem}__{eval_tag}.{ext}"


def load_eval_metric(
    task: str,
    split: str,
    filename: str,
    metric_key: str,
    *,
    eval_tag: Optional[str] = None,
) -> Optional[float]:
    # In subset-tagged mode, only read subset-tagged metrics to avoid mixing
    # full-split baselines with subset evaluations.
    candidates = [with_eval_tag(filename, eval_tag)] if eval_tag else [filename]
    metrics_path = None
    for name in candidates:
        path = PACKAGE_ROOT / "artifacts" / task / "metrics" / "eval" / split / name
        if path.exists():
            metrics_path = path
            break
    if metrics_path is None or not metrics_path.exists():
        return None
    try:
        with metrics_path.open("r") as handle:
            data = json.load(handle)
    except Exception:
        return None
    value = data.get(metric_key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def load_eval_metrics_json(
    task: str,
    split: str,
    filename: str,
    *,
    eval_tag: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    # In subset-tagged mode, only read subset-tagged metrics to avoid mixing
    # full-split baselines with subset evaluations.
    candidates = [with_eval_tag(filename, eval_tag)] if eval_tag else [filename]
    metrics_path = None
    for name in candidates:
        path = PACKAGE_ROOT / "artifacts" / task / "metrics" / "eval" / split / name
        if path.exists():
            metrics_path = path
            break
    if metrics_path is None or not metrics_path.exists():
        return None
    try:
        with metrics_path.open("r") as handle:
            data = json.load(handle)
    except Exception:
        return None
    if isinstance(data, dict):
        return data
    return None


def oriented_score(value: float, higher_is_better: bool) -> float:
    return value if higher_is_better else -value


def compute_eval_tag_from_subset(eval_subset: Optional[Mapping[str, Any]]) -> Optional[str]:
    if eval_subset and bool(eval_subset.get("enabled", True)):
        return compute_eval_subset_tag(eval_subset)
    return None


def maybe_compute_interference_baselines(
    *,
    tasks: List[str],
    split: str,
    enable_cache: bool,
    batch_size: Optional[int],
    show_summary: bool,
    eval_subset: Optional[Dict[str, Any]] = None,
) -> None:
    """Ensure baseline metrics exist for computing interference_delta."""
    # Import lazily to avoid pulling in heavyweight deps for call sites
    # that only want to read or post-process metrics.
    from core.evaluation.evaluate_task import evaluate

    eval_tag = compute_eval_tag_from_subset(eval_subset)
    for task in tasks:
        task_key = task.lower()
        if task_key not in TASK_METRICS:
            continue

        metric_key, _ = TASK_METRICS[task_key]
        base_value = load_eval_metric(task_key, split, "base_model.json", metric_key, eval_tag=eval_tag)
        task_value = load_eval_metric(
            task_key,
            split,
            f"best_{task_key}_adapter.json",
            metric_key,
            eval_tag=eval_tag,
        )
        if base_value is not None and task_value is not None:
            continue

        if show_summary:
            missing = []
            if base_value is None:
                missing.append("base_model.json")
            if task_value is None:
                missing.append(f"best_{task_key}_adapter.json")
            print(f"🧮 Computing missing interference baselines for {task_key}/{split}: {', '.join(missing)}")

        try:
            if base_value is None:
                evaluate(
                    task=task_key,
                    adapter=None,
                    split=split,
                    batch_size=batch_size,
                    enable_cache=enable_cache,
                    show_summary=show_summary,
                    generate_confusion_matrix=False,
                    eval_subset=eval_subset,
                )
        except Exception as exc:
            if show_summary:
                print(f"⚠️  Failed to compute base_model metrics for {task_key}/{split}: {exc}")

        try:
            if task_value is None:
                # Passing adapter=<task> resolves that task's "best" adapter run.
                evaluate(
                    task=task_key,
                    adapter=task_key,
                    split=split,
                    batch_size=batch_size,
                    enable_cache=enable_cache,
                    show_summary=show_summary,
                    generate_confusion_matrix=False,
                    eval_subset=eval_subset,
                )
        except Exception as exc:
            if show_summary:
                print(f"⚠️  Failed to compute best adapter metrics for {task_key}/{split}: {exc}")


def maybe_add_interference_delta(
    task: str,
    metrics: Dict[str, Any],
    split: str,
    show_summary: bool,
    *,
    eval_tag: Optional[str],
) -> None:
    task_key = task.lower()
    if task_key not in TASK_METRICS:
        return

    metric_key, higher_is_better = TASK_METRICS[task_key]
    merged_value = metrics.get(metric_key)
    if not isinstance(merged_value, (int, float)):
        return

    base_value = load_eval_metric(task_key, split, "base_model.json", metric_key, eval_tag=eval_tag)
    task_value = load_eval_metric(
        task_key,
        split,
        f"best_{task_key}_adapter.json",
        metric_key,
        eval_tag=eval_tag,
    )
    if base_value is None or task_value is None:
        if show_summary:
            missing = []
            if base_value is None:
                missing.append(with_eval_tag("base_model.json", eval_tag))
            if task_value is None:
                missing.append(with_eval_tag(f"best_{task_key}_adapter.json", eval_tag))
            missing_str = ", ".join(missing)
            print(
                f"⚠️  Skipping interference_delta for {task_key}/{split}: "
                f"missing {missing_str} under artifacts/{task_key}/metrics/eval/{split}"
            )
        return

    merged_score = oriented_score(float(merged_value), higher_is_better)
    base_score = oriented_score(base_value, higher_is_better)
    task_score = oriented_score(task_value, higher_is_better)
    denom = task_score - base_score
    if abs(denom) < EPS:
        return

    metrics["interference_delta"] = (merged_score - base_score) / denom
    metrics["interference_delta_meta"] = {
        "metric": metric_key,
        "base": base_value,
        "task_adapter": task_value,
        "merged": float(merged_value),
        "split": split,
    }
    if show_summary:
        print(f"   interference_delta: {metrics['interference_delta']:.4f}")


__all__ = [
    "EPS",
    "TASK_METRICS",
    "with_eval_tag",
    "load_eval_metric",
    "load_eval_metrics_json",
    "oriented_score",
    "compute_eval_tag_from_subset",
    "maybe_compute_interference_baselines",
    "maybe_add_interference_delta",
]
