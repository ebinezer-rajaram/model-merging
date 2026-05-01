"""Writes a standardised experiment_summary.json alongside every run's artifacts.

All three experiment types (single_task, mtl, merge) produce the same schema.
Existing output files are never modified — this is purely additive.

Schema version: 2
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# Canonical primary metric per task — imported from the single source of truth.
try:
    from merging.evaluation.interference import TASK_METRICS  # type: ignore
except Exception:
    TASK_METRICS: Dict[str, tuple] = {
        "asr": ("wer", False),
        "emotion": ("macro_f1", True),
        "intent": ("accuracy", True),
        "speech_qa": ("accuracy", True),
        "kws": ("macro_f1", True),
        "langid": ("accuracy", True),
        "speaker_id": ("accuracy", True),
        "speaker_ver": ("accuracy", True),
        "vocalsound": ("accuracy", True),
        "st": ("BLEU", False),
    }

# Metric keys that belong to the named set (not written as "extra").
_NAMED_METRICS = frozenset({
    "loss", "accuracy", "macro_f1", "weighted_f1", "wer",
    "recognized_rate", "num_samples",
    "interference_delta", "interference_delta_info",
})

# Keys to strip when reading raw metric dicts from source files.
_RUNTIME_KEYS = frozenset({
    "runtime", "model_preparation_time",
    "samples_per_second", "steps_per_second",
    "_eval_subset", "_predictions", "_labels",
    "interference_delta_meta",  # raw form — extracted into interference_delta_info
})

_EVAL_PREFIX = "eval_"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_experiment_summary(
    output_path: Path,
    experiment_type: str,
    run_id: Optional[str],
    timestamp: Optional[str],
    config_name: Optional[str],
    source_tasks: List[str],
    method: str,
    results: Dict[str, Dict[str, Dict[str, Any]]],
    adapter_path: Optional[str] = None,
    is_best: Optional[bool] = None,
    is_latest: Optional[bool] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    merge_info: Optional[Dict[str, Any]] = None,
    mtl_aggregate: Optional[Dict[str, Any]] = None,
    selection: Optional[Dict[str, Any]] = None,
    source_files: Optional[List[str]] = None,
) -> None:
    """Write experiment_summary.json to output_path.

    This function is silent on error — it logs a warning and returns without
    raising, following the same pattern as MTL's _refresh_plots_best_effort().

    Args:
        output_path: Destination path (e.g. run_dir / "experiment_summary.json").
        experiment_type: "single_task" | "mtl" | "merge".
        run_id: Run directory name (e.g. "run_20260114_093047").
        timestamp: ISO 8601 timestamp string or None.
        config_name: Config hash (single/MTL) or merge_tag (merge).
        source_tasks: Tasks that were trained or merged. Always sorted.
        method: Task name (single_task) | "mtl" | merge method name.
        results: {task: {split: {metric_key: value}}}. Metric keys may have
                 "eval_" prefix — it will be stripped automatically.
        adapter_path: Absolute or relative path to the run directory containing
                      the adapter weights.
        is_best: Whether this run is the best in its registry.
        is_latest: Whether this run is the latest in its registry.
        hyperparameters: Dict of training hyperparameters (null for merge).
        merge_info: {"lambda": float, "merge_tag": str, "params": dict} or None.
        mtl_aggregate: MTL summary stats (null for single_task/merge).
        selection: {"policy", "metric_name", "metric_value", "ranked_by_split"}.
        source_files: Filenames read to build this summary.
    """
    try:
        _write(
            output_path=output_path,
            experiment_type=experiment_type,
            run_id=run_id,
            timestamp=timestamp,
            config_name=config_name,
            source_tasks=source_tasks,
            method=method,
            results=results,
            adapter_path=adapter_path,
            is_best=is_best,
            is_latest=is_latest,
            hyperparameters=hyperparameters,
            merge_info=merge_info,
            mtl_aggregate=mtl_aggregate,
            selection=selection,
            source_files=source_files,
        )
    except Exception as exc:
        log.warning("Could not write experiment_summary.json to %s: %s", output_path, exc)


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

def _write(
    output_path: Path,
    experiment_type: str,
    run_id: Optional[str],
    timestamp: Optional[str],
    config_name: Optional[str],
    source_tasks: List[str],
    method: str,
    results: Dict[str, Dict[str, Dict[str, Any]]],
    adapter_path: Optional[str],
    is_best: Optional[bool],
    is_latest: Optional[bool],
    hyperparameters: Optional[Dict[str, Any]],
    merge_info: Optional[Dict[str, Any]],
    mtl_aggregate: Optional[Dict[str, Any]],
    selection: Optional[Dict[str, Any]],
    source_files: Optional[List[str]],
) -> None:
    source_tasks = sorted(source_tasks) if source_tasks else []
    source_tasks_key = "+".join(source_tasks)

    # Build the annotated results dict.
    annotated_results = _annotate_results(results, source_tasks, experiment_type)

    doc: Dict[str, Any] = {
        "schema_version": "2",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment_type": experiment_type,
        "run_id": run_id,
        "timestamp": timestamp,
        "config_name": config_name,
        "adapter_path": str(adapter_path) if adapter_path is not None else None,
        "is_best": is_best,
        "is_latest": is_latest,
        "source_tasks": source_tasks,
        "source_tasks_key": source_tasks_key,
        "method": method,
        "merge": merge_info,
        "hyperparameters": hyperparameters,
        "results": annotated_results,
        "mtl_aggregate": mtl_aggregate,
        "selection": selection,
        "source_files": source_files or [],
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: write to .tmp then rename.
    tmp_path = output_path.with_suffix(".json.tmp")
    with tmp_path.open("w") as fh:
        json.dump(doc, fh, indent=2, default=_json_default)
    os.replace(tmp_path, output_path)

    log.info("Wrote experiment_summary.json → %s", output_path)


def _annotate_results(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    source_tasks: List[str],
    experiment_type: str,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Strip eval_ prefix, annotate primary_metric and eval_context per split."""
    from core.results.utils import derive_eval_context  # type: ignore

    annotated: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for task, splits in results.items():
        task_entry: Dict[str, Dict[str, Any]] = {}
        primary_metric = _primary_metric_name(task)

        # Order splits: test first, then validation, then others.
        ordered_splits = _order_splits(splits.keys())

        for split in ordered_splits:
            raw_metrics = splits[split]
            if not isinstance(raw_metrics, dict):
                continue

            cleaned = _clean_metrics(raw_metrics)
            if not cleaned:
                continue

            # eval_context: for single_task always "full"; infer for MTL/merge.
            if experiment_type == "single_task":
                eval_ctx = "full"
            else:
                eval_tag = raw_metrics.get("eval_tag") or cleaned.get("eval_tag")
                eval_ctx = derive_eval_context(task, source_tasks, eval_tag)

            entry: Dict[str, Any] = {
                "eval_context": eval_ctx,
                "primary_metric": primary_metric,
            }

            # Named metrics first (in a consistent order), then extras.
            for key in ("loss", "accuracy", "macro_f1", "weighted_f1", "wer", "recognized_rate", "num_samples"):
                val = cleaned.get(key)
                if val is not None:
                    entry[key] = val

            # Interference delta.
            delta = cleaned.get("interference_delta")
            if delta is not None:
                entry["interference_delta"] = delta

            delta_info = _extract_delta_info(raw_metrics, cleaned)
            if delta_info is not None:
                entry["interference_delta_info"] = delta_info
            elif experiment_type != "single_task":
                # Explicitly null for MTL/merge when not available.
                entry["interference_delta_info"] = None

            # Extra metrics (anything not in the named set).
            for k, v in cleaned.items():
                if k not in _NAMED_METRICS and k not in entry and v is not None:
                    entry[k] = v

            task_entry[split] = entry

        if task_entry:
            annotated[task] = task_entry

    return annotated


def _clean_metrics(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Strip eval_ prefix and remove runtime/internal keys."""
    out: Dict[str, Any] = {}
    for k, v in raw.items():
        if k in _RUNTIME_KEYS:
            continue
        stripped = k.removeprefix(_EVAL_PREFIX)
        if stripped in _RUNTIME_KEYS:
            continue
        out[stripped] = v
    return out


def _extract_delta_info(
    raw_metrics: Dict[str, Any],
    cleaned: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Extract interference_delta_info from raw metrics dict."""
    # Merge experiments store this as "interference_delta_meta".
    meta = raw_metrics.get("interference_delta_meta")
    if isinstance(meta, dict):
        return {
            "metric": meta.get("metric"),
            "base": meta.get("base"),
            "task_adapter": meta.get("task_adapter"),
            "merged": meta.get("merged"),
        }
    # MTL JSONL does not carry full provenance inline — return None.
    return None


def _primary_metric_name(task: str) -> Optional[str]:
    entry = TASK_METRICS.get(task)
    return entry[0] if entry else None


def _order_splits(splits) -> List[str]:
    """Return splits ordered: test, validation, then others alphabetically."""
    priority = {"test": 0, "validation": 1}
    return sorted(splits, key=lambda s: (priority.get(s, 2), s))


def _json_default(obj: Any) -> Any:
    """Fallback JSON serializer for non-standard types."""
    if hasattr(obj, "__fspath__"):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Convenience builders — used by generate_outputs.py and training integrations
# ---------------------------------------------------------------------------

def build_selection(
    policy: str,
    metric_name: Optional[str],
    metric_value: Optional[float],
    ranked_by_split: Optional[str],
) -> Dict[str, Any]:
    return {
        "policy": policy,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "ranked_by_split": ranked_by_split,
    }


def build_hyperparameters(
    learning_rate: Optional[float] = None,
    lora_r: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    num_train_epochs: Optional[int] = None,
    per_device_train_batch_size: Optional[int] = None,
    sampling_temperature: Optional[float] = None,
    selection_criterion: Optional[str] = None,
    num_tasks: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "learning_rate": learning_rate,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
    }
    if sampling_temperature is not None:
        d["sampling_temperature"] = sampling_temperature
    if selection_criterion is not None:
        d["selection_criterion"] = selection_criterion
    if num_tasks is not None:
        d["num_tasks"] = num_tasks
    if max_steps is not None:
        d["max_steps"] = max_steps
    return d
