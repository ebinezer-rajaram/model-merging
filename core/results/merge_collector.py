"""Collector for adapter merging experiment results."""

from __future__ import annotations

import logging
import math
import statistics
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
    derive_eval_context,
    make_experiment_id,
    make_source_tasks_key,
    safe_load_json,
)

log = logging.getLogger(__name__)

# Directories under artifacts/merged/ that are not method directories.
_SKIP_DIRS = frozenset({"comparisons", "eval", ".git"})


class MergeCollector(ResultsCollector):
    """Reads adapter merging results from eval_results_*.json files.

    Discovery path:
        artifacts/merged/{method}/{task_combo}/eval/{split}/eval_results_*.json

    Method directories are discovered dynamically — no method names are
    hardcoded. Any subdirectory under artifacts/merged/ is treated as a method.

    By default:
      - All lambda sweep values are included (use best_lambda_only=True to filter).
      - Files with '__' in their stem (subset evaluations) are excluded.
    """

    def __init__(
        self,
        artifacts_root: Path,
        methods: Optional[List[str]] = None,
        splits: Optional[List[str]] = None,
        include_subset_evals: bool = False,
        best_lambda_only: bool = False,
    ):
        """
        Args:
            artifacts_root: Root of the artifacts directory.
            methods: Restrict to these merge method names. None = all found.
            splits: Splits to collect. Default: ["test"].
            include_subset_evals: If True, include files with '__' in their stem.
            best_lambda_only: If True, keep only the best lambda per (method, task_combo).
        """
        self.artifacts_root = Path(artifacts_root)
        self.methods = set(methods) if methods else None
        self.splits = list(splits) if splits else ["test"]
        self.include_subset_evals = include_subset_evals
        self.best_lambda_only = best_lambda_only

    def collect(self) -> Iterator[ExperimentResult]:
        merged_root = self.artifacts_root / "merged"
        if not merged_root.is_dir():
            return

        # Gather all candidate result files first (needed for best-lambda filtering).
        all_files: List[tuple[str, Path, Path]] = []  # (method_name, task_combo_dir, result_file)
        for method_dir in sorted(merged_root.iterdir()):
            if not method_dir.is_dir():
                continue
            if method_dir.name in _SKIP_DIRS:
                continue
            if self.methods and method_dir.name not in self.methods:
                continue

            for task_combo_dir in sorted(method_dir.iterdir()):
                if not task_combo_dir.is_dir():
                    continue
                eval_dir = task_combo_dir / "eval"
                if not eval_dir.is_dir():
                    continue

                for split in self.splits:
                    split_dir = eval_dir / split
                    if not split_dir.is_dir():
                        continue
                    for result_file in sorted(split_dir.glob("eval_results_*.json")):
                        if not self.include_subset_evals and "__" in result_file.stem:
                            continue
                        all_files.append((method_dir.name, task_combo_dir, result_file))

        if self.best_lambda_only:
            all_files = self._filter_best_lambda(all_files)

        for method_name, task_combo_dir, result_file in all_files:
            try:
                yield from self._collect_from_file(method_name, task_combo_dir, result_file)
            except Exception as exc:
                log.warning(
                    "[MergeCollector] Failed to parse %s: %s", result_file, exc
                )

    def _filter_best_lambda(
        self,
        all_files: List[tuple],
    ) -> List[tuple]:
        """Keep only the result file with the highest mean interference_delta per
        (method_name, task_combo_dir, split)."""
        from collections import defaultdict

        grouped: Dict[tuple, List[tuple]] = defaultdict(list)
        for entry in all_files:
            method_name, task_combo_dir, result_file = entry
            group_key = (method_name, str(task_combo_dir), result_file.parent.name)
            grouped[group_key].append(entry)

        kept = []
        for group_key, entries in grouped.items():
            best_entry = None
            best_score = float("-inf")
            for entry in entries:
                _, _, result_file = entry
                score = _compute_mean_delta_from_file(result_file)
                if score is not None and score > best_score:
                    best_score = score
                    best_entry = entry
            if best_entry is not None:
                kept.append(best_entry)
            else:
                # No scores available — keep all entries for this group.
                kept.extend(entries)
        return kept

    def _collect_from_file(
        self,
        method_name: str,
        task_combo_dir: Path,
        result_file: Path,
    ) -> Iterator[ExperimentResult]:
        payload = safe_load_json(result_file)
        if payload is None:
            return

        split = payload.get("split", result_file.parent.name)
        timestamp = payload.get("timestamp")
        merge_tag = payload.get("merge_tag")
        source_tasks: List[str] = list(payload.get("source_tasks") or [])
        merge_method = payload.get("merge_method") or method_name
        merge_lambda = _safe_float(payload.get("lambda"))
        eval_tag = payload.get("eval_tag") or None

        # Detect subset eval from filename if eval_tag not in payload.
        if eval_tag is None and "__" in result_file.stem:
            eval_tag = result_file.stem.split("__", 1)[1]

        results: Dict[str, Any] = dict(payload.get("results") or {})

        # Supplement with per_task/ sidecar files when the top-level results
        # dict is partial (e.g. weighted_delta_n only evaluated a subset of tasks).
        # per_task/{task}/*.json files contain flat metric dicts for each task.
        per_task_dir = result_file.parent / "per_task"
        if per_task_dir.is_dir():
            for task_dir in sorted(per_task_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                task_name = task_dir.name
                if task_name in results:
                    continue  # already present from top-level results
                task_files = sorted(task_dir.glob("*.json"))
                if not task_files:
                    continue
                try:
                    task_metrics = safe_load_json(task_files[-1])
                    if task_metrics:
                        results[task_name] = task_metrics
                except Exception as exc:
                    log.warning(
                        "[MergeCollector] Failed to read per_task file %s: %s",
                        task_files[-1], exc,
                    )

        if not results:
            return

        source_tasks_key = make_source_tasks_key(source_tasks)
        experiment_id = make_experiment_id("merge", merge_method, source_tasks)

        # Try to load merge_metadata for additional provenance.
        run_id = _infer_run_id_from_path(result_file)
        merge_metadata = _try_load_merge_metadata(task_combo_dir, run_id)
        if merge_lambda is None and merge_metadata:
            merge_lambda = _safe_float(merge_metadata.get("lambda"))

        # Selection provenance — each file is one sweep point.
        # If best_lambda_only was applied upstream, we'd label this differently,
        # but that info is not available here; the aggregator handles it at a
        # higher level. We always use "sweep_point" per file.
        selection_policy = "sweep_point"

        metadata = ExperimentMetadata(
            experiment_type="merge",
            experiment_id=experiment_id,
            method=merge_method,
            selection_policy=selection_policy,
            selection_metric_name=None,
            selection_metric_value=None,
            timestamp=timestamp,
            run_id=run_id,
            source_tasks=source_tasks,
            source_tasks_key=source_tasks_key,
            merge_lambda=merge_lambda,
            merge_tag=merge_tag,
            source_path=str(result_file),
        )

        for task, task_metrics in results.items():
            try:
                result = self._build_task_result(
                    task=task,
                    task_metrics=task_metrics,
                    split=split,
                    eval_tag=eval_tag,
                    source_tasks=source_tasks,
                    metadata=metadata,
                )
                if result is not None:
                    yield result
            except Exception as exc:
                log.warning(
                    "[MergeCollector] Failed to build result for task %s in %s: %s",
                    task,
                    result_file,
                    exc,
                )

    def _build_task_result(
        self,
        task: str,
        task_metrics: Dict[str, Any],
        split: str,
        eval_tag: Optional[str],
        source_tasks: List[str],
        metadata: ExperimentMetadata,
    ) -> Optional[ExperimentResult]:
        if not isinstance(task_metrics, dict):
            return None

        # Interference delta
        interference_delta = _safe_float(task_metrics.get("interference_delta"))
        interference_delta_info: Optional[InterferenceDeltaInfo] = None
        meta_raw = task_metrics.get("interference_delta_meta")
        if isinstance(meta_raw, dict):
            try:
                interference_delta_info = InterferenceDeltaInfo(
                    value=float(meta_raw["merged"] - meta_raw.get("base", 0)),  # not the delta formula — use stored value
                    metric_name=str(meta_raw.get("metric", "")),
                    base_score=float(meta_raw.get("base", 0.0)),
                    task_adapter_score=float(meta_raw.get("task_adapter", 0.0)),
                    merged_score=float(meta_raw.get("merged", 0.0)),
                )
                # Use the stored interference_delta value, not a recomputation.
                if interference_delta is not None and interference_delta_info is not None:
                    interference_delta_info = InterferenceDeltaInfo(
                        value=interference_delta,
                        metric_name=interference_delta_info.metric_name,
                        base_score=interference_delta_info.base_score,
                        task_adapter_score=interference_delta_info.task_adapter_score,
                        merged_score=interference_delta_info.merged_score,
                    )
            except Exception:
                interference_delta_info = None

        eval_ctx = derive_eval_context(task, source_tasks, eval_tag)

        # Extra metrics — anything not in the named set.
        known_keys = {
            "loss", "accuracy", "macro_f1", "weighted_f1", "wer",
            "recognized_rate", "interference_delta", "interference_delta_meta",
            "num_samples", "runtime", "model_preparation_time",
            "samples_per_second", "steps_per_second", "_eval_subset",
        }
        extra: Dict[str, Any] = {
            k: v for k, v in task_metrics.items()
            if k not in known_keys and isinstance(v, (int, float))
        }

        task_result = TaskEvalResult(
            task=task,
            split=split,
            eval_context=eval_ctx,
            eval_tag=eval_tag,
            loss=_safe_float(task_metrics.get("loss")),
            accuracy=_safe_float(task_metrics.get("accuracy")),
            macro_f1=_safe_float(task_metrics.get("macro_f1")),
            weighted_f1=_safe_float(task_metrics.get("weighted_f1")),
            wer=_safe_float(task_metrics.get("wer")),
            recognized_rate=_safe_float(task_metrics.get("recognized_rate")),
            interference_delta=interference_delta,
            interference_delta_info=interference_delta_info,
            num_samples=_to_int(task_metrics.get("num_samples")),
            runtime_seconds=_safe_float(task_metrics.get("runtime")),
            extra_metrics=extra,
        )

        return ExperimentResult(
            metadata=metadata,
            task_result=task_result,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_run_id_from_path(result_file: Path) -> Optional[str]:
    """Try to extract run_id from the result file path."""
    for part in result_file.parts:
        if part.startswith("run_"):
            return part
    return None


def _try_load_merge_metadata(
    task_combo_dir: Path,
    run_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Attempt to load merge_metadata.json from the run directory."""
    if run_id:
        path = task_combo_dir / "runs" / run_id / "merge_metadata.json"
        if path.exists():
            return safe_load_json(path)
    # Fallback: search all run dirs for a merge_metadata.json.
    runs_dir = task_combo_dir / "runs"
    if runs_dir.is_dir():
        candidates = sorted(runs_dir.glob("*/merge_metadata.json"), reverse=True)
        if candidates:
            return safe_load_json(candidates[0])
    return None


def _compute_mean_delta_from_file(result_file: Path) -> Optional[float]:
    """Compute mean interference_delta across tasks in a result file."""
    data = safe_load_json(result_file)
    if data is None:
        return None
    results = data.get("results") or {}
    deltas = []
    for task_metrics in results.values():
        if not isinstance(task_metrics, dict):
            continue
        delta = _safe_float(task_metrics.get("interference_delta"))
        if delta is not None:
            deltas.append(delta)
    if not deltas:
        return None
    try:
        return statistics.mean(deltas)
    except Exception:
        return None


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
