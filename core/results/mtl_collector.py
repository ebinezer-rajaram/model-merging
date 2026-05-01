"""Collector for multi-task learning (MTL) experiment results."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .base import ResultsCollector
from .schema import (
    ExperimentMetadata,
    ExperimentResult,
    InterferenceDeltaInfo,
    TaskEvalResult,
    TrainingHyperparameters,
)
from .utils import (
    KNOWN_TASKS,
    derive_eval_context,
    is_primary_metric,
    make_experiment_id,
    make_source_tasks_key,
    safe_load_json,
)

log = logging.getLogger(__name__)

_DEFAULT_SELECTION_CRITERION = "geometric_mean_interference_delta"
_EVAL_PREFIX = "eval_"
_MTL_PREFIX = "eval_mtl_"


class MTLCollector(ResultsCollector):
    """Reads MTL experiment results from mtl_eval_history.jsonl files.

    Discovery:
        artifacts/mtl/**/mtl_eval_history.jsonl   (rglob — handles both
        flat and nested artifact layouts produced by train_multitask.py)

    Best checkpoint selection:
        Reads the sibling mtl_config_resolved.json to find the
        training.selection_criterion used for this run, then picks the
        JSONL row with the highest value for eval_mtl_{criterion}.
        Falls back to geometric_mean_interference_delta if config is missing.
    """

    def __init__(
        self,
        artifacts_root: Path,
    ):
        self.artifacts_root = Path(artifacts_root)

    def collect(self) -> Iterator[ExperimentResult]:
        mtl_root = self.artifacts_root / "mtl"
        if not mtl_root.is_dir():
            return

        for jsonl_path in sorted(mtl_root.rglob("mtl_eval_history.jsonl")):
            try:
                yield from self._collect_from_jsonl(jsonl_path)
            except Exception as exc:
                log.warning(
                    "[MTLCollector] Failed to process %s: %s", jsonl_path, exc
                )

    def _collect_from_jsonl(self, jsonl_path: Path) -> Iterator[ExperimentResult]:
        metrics_dir = jsonl_path.parent

        # Load config for selection criterion and task list.
        config = safe_load_json(metrics_dir / "mtl_config_resolved.json")
        selection_criterion, task_list, seed, hp = _parse_mtl_config(config)

        # Load all JSONL rows.
        rows = _load_jsonl_rows(jsonl_path)
        if not rows:
            return

        # Select the best checkpoint row.
        sel_key = f"{_EVAL_PREFIX}mtl_{selection_criterion}"
        best_row, best_value = _select_best_row(rows, sel_key)
        if best_row is None:
            log.warning("[MTLCollector] No valid rows found in %s", jsonl_path)
            return

        # Resolve task list from config or by parsing the row keys.
        if not task_list:
            task_list = _infer_tasks_from_row(best_row)
        if not task_list:
            log.warning("[MTLCollector] Could not determine task list from %s", jsonl_path)
            return

        source_tasks = sorted(task_list)
        source_tasks_key = make_source_tasks_key(source_tasks)
        experiment_id = make_experiment_id("mtl", "mtl", source_tasks)

        # MTL aggregate metrics from the best row.
        mtl_geom = _row_float(best_row, f"{_EVAL_PREFIX}mtl_geometric_mean_interference_delta")
        mtl_arith = _row_float(best_row, f"{_EVAL_PREFIX}mtl_arithmetic_mean_interference_delta")
        mtl_num = _row_int(best_row, f"{_EVAL_PREFIX}mtl_num_tasks_with_delta")
        global_step = best_row.get("step")

        # Determine run_id: check if jsonl lives inside a runs/ directory.
        run_id = _infer_run_id(jsonl_path)

        metadata_base = ExperimentMetadata(
            experiment_type="mtl",
            experiment_id=experiment_id,
            method="mtl",
            selection_policy="best_delta",
            selection_metric_name=sel_key,
            selection_metric_value=best_value,
            seed=seed,
            run_id=run_id,
            source_tasks=source_tasks,
            source_tasks_key=source_tasks_key,
            source_path=str(jsonl_path),
        )

        for task in task_list:
            try:
                result = self._build_task_result(
                    task=task,
                    row=best_row,
                    metadata_base=metadata_base,
                    hyperparameters=hp,
                    mtl_geom=mtl_geom,
                    mtl_arith=mtl_arith,
                    mtl_num=mtl_num,
                )
                if result is not None:
                    yield result
            except Exception as exc:
                log.warning(
                    "[MTLCollector] Failed to build result for task %s from %s: %s",
                    task,
                    jsonl_path,
                    exc,
                )

    def _build_task_result(
        self,
        task: str,
        row: Dict[str, Any],
        metadata_base: ExperimentMetadata,
        hyperparameters: Optional[TrainingHyperparameters],
        mtl_geom: Optional[float],
        mtl_arith: Optional[float],
        mtl_num: Optional[int],
    ) -> Optional[ExperimentResult]:
        prefix = f"{_EVAL_PREFIX}{task}_"

        # Collect all keys for this task.
        task_vals: Dict[str, Optional[float]] = {}
        for k, v in row.items():
            if k.startswith(prefix):
                subkey = k[len(prefix):]
                task_vals[subkey] = _safe_float(v)

        if not task_vals:
            return None  # task not present in this row

        # Interference delta info
        interference_delta = task_vals.get("interference_delta")
        interference_delta_info = None
        # The MTL JSONL does not store interference_delta_meta inline;
        # the provenance fields are not written to the JSONL (only to per-eval
        # JSON artifacts). We store what we have.

        # Known metric fields
        extra: Dict[str, Any] = {}
        known_keys = {
            "loss", "accuracy", "macro_f1", "weighted_f1", "wer",
            "recognized_rate", "interference_delta", "num_samples",
            "runtime", "model_preparation_time", "samples_per_second",
            "steps_per_second",
        }
        for k, v in task_vals.items():
            if k not in known_keys and v is not None:
                extra[k] = v

        # eval_context — MTL trains on a set of tasks; speech_qa/st may be
        # evaluated as OOD even if not in source_tasks.
        eval_ctx = derive_eval_context(
            task=task,
            source_tasks=metadata_base.source_tasks,
            eval_tag=None,
        )

        task_result = TaskEvalResult(
            task=task,
            split="validation",  # MTL history is always validation-split evals
            eval_context=eval_ctx,
            loss=task_vals.get("loss"),
            accuracy=task_vals.get("accuracy"),
            macro_f1=task_vals.get("macro_f1"),
            weighted_f1=task_vals.get("weighted_f1"),
            wer=task_vals.get("wer"),
            recognized_rate=task_vals.get("recognized_rate"),
            interference_delta=interference_delta,
            interference_delta_info=interference_delta_info,
            num_samples=_to_int(task_vals.get("num_samples")),
            runtime_seconds=task_vals.get("runtime"),
            extra_metrics=extra,
        )

        return ExperimentResult(
            metadata=metadata_base,
            task_result=task_result,
            hyperparameters=hyperparameters,
            mtl_geometric_mean_interference_delta=mtl_geom,
            mtl_arithmetic_mean_interference_delta=mtl_arith,
            mtl_num_tasks_with_delta=mtl_num,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_mtl_config(
    config: Optional[Dict[str, Any]],
) -> tuple[str, List[str], Optional[int], Optional[TrainingHyperparameters]]:
    """Extract selection_criterion, task list, seed, and hyperparameters from config."""
    if config is None:
        return _DEFAULT_SELECTION_CRITERION, [], None, None

    training_cfg = config.get("training", {}) or {}
    criterion = training_cfg.get("selection_criterion", _DEFAULT_SELECTION_CRITERION) or _DEFAULT_SELECTION_CRITERION
    criterion = str(criterion).strip().lower()

    tasks_raw = config.get("tasks", []) or []
    task_list: List[str] = []
    for entry in tasks_raw:
        if isinstance(entry, dict):
            name = entry.get("name")
            if name:
                task_list.append(str(name))
        elif isinstance(entry, str):
            task_list.append(entry)

    seed_raw = config.get("seed")
    seed = int(seed_raw) if seed_raw is not None else None

    lora_cfg = (config.get("model", {}) or {}).get("lora", {}) or {}
    sampling_cfg = training_cfg.get("sampling", {}) or {}
    hp = TrainingHyperparameters(
        learning_rate=_safe_float(training_cfg.get("learning_rate")),
        lora_r=_to_int(lora_cfg.get("r")),
        lora_alpha=_to_int(lora_cfg.get("alpha")),
        num_train_epochs=_to_int(training_cfg.get("num_train_epochs")),
        per_device_train_batch_size=_to_int(training_cfg.get("per_device_train_batch_size")),
        sampling_temperature=_safe_float(sampling_cfg.get("temperature")),
        selection_criterion=criterion,
        num_tasks=len(task_list) if task_list else None,
    )

    return criterion, task_list, seed, hp


def _load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    try:
        with path.open("r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        log.warning("[MTLCollector] Could not read %s: %s", path, exc)
    return rows


def _select_best_row(
    rows: List[Dict[str, Any]],
    sel_key: str,
) -> tuple[Optional[Dict[str, Any]], Optional[float]]:
    """Return the row with the highest finite value for sel_key."""
    best_row = None
    best_value: Optional[float] = None
    for row in rows:
        raw = row.get(sel_key)
        if raw is None:
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v):
            continue
        if best_value is None or v > best_value:
            best_value = v
            best_row = row
    # Fallback: last row if nothing had a finite selection metric.
    if best_row is None and rows:
        best_row = rows[-1]
    return best_row, best_value


def _infer_tasks_from_row(row: Dict[str, Any]) -> List[str]:
    """Infer task names from eval_{task}_{metric} keys present in the row.

    Task names are matched against KNOWN_TASKS (longest-match first) to handle
    multi-word task names like 'speaker_ver' correctly.
    """
    # Sort known tasks by descending length so 'speaker_ver' matches before 'speaker'.
    sorted_tasks = sorted(KNOWN_TASKS, key=len, reverse=True)

    tasks = set()
    for k in row.keys():
        if not k.startswith(_EVAL_PREFIX):
            continue
        without_prefix = k[len(_EVAL_PREFIX):]
        if without_prefix.startswith("mtl_"):
            continue
        for task in sorted_tasks:
            if without_prefix.startswith(task + "_") or without_prefix == task:
                tasks.add(task)
                break
    return sorted(tasks)


def _infer_run_id(jsonl_path: Path) -> Optional[str]:
    """Try to extract run_id from the path (looks for a 'runs/run_...' component)."""
    for part in jsonl_path.parts:
        if part.startswith("run_"):
            return part
    return None


def _row_float(row: Dict[str, Any], key: str) -> Optional[float]:
    return _safe_float(row.get(key))


def _row_int(row: Dict[str, Any], key: str) -> Optional[int]:
    return _to_int(row.get(key))


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return None if not math.isfinite(f) else f
    except (TypeError, ValueError):
        return None


def _to_int(v) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
