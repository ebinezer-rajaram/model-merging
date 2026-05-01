"""ResultsAggregator: runs all collectors and produces the unified DataFrame."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import ResultsCollector
from .schema import ExperimentResult
from .utils import NAMED_METRIC_FIELDS, is_primary_metric, make_source_tasks_key

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column order for the output DataFrame.
# New columns should be appended here to maintain backward compatibility.
# ---------------------------------------------------------------------------
DATAFRAME_COLUMNS: List[str] = [
    # Identity
    "experiment_id",
    "experiment_type",
    "method",
    "task",
    "split",
    "eval_context",
    # Metric
    "metric_name",
    "metric_value",
    "is_primary_metric",
    # Selection provenance
    "selection_policy",
    "selection_metric_name",
    "selection_metric_value",
    # Run identity
    "seed",
    "config_name",
    "run_id",
    "timestamp",
    # Task provenance
    "source_tasks",
    "source_tasks_key",
    "source_tasks_count",
    # Interference delta
    "interference_delta",
    "interference_metric",
    "interference_base",
    "interference_task_adapter",
    "interference_merged",
    # Merge-specific
    "merge_lambda",
    "merge_tag",
    "eval_tag",
    # Hyperparameters
    "learning_rate",
    "lora_r",
    "lora_alpha",
    "num_train_epochs",
    "per_device_train_batch_size",
    # MTL aggregates
    "mtl_geometric_mean_interference_delta",
    "mtl_arithmetic_mean_interference_delta",
    "mtl_num_tasks",
    # Filesystem provenance
    "adapter_subdir",
    "model_path",
    "source_path",
]

# Columns useful for analysis (excluding low-level provenance).
ANALYSIS_COLUMNS: List[str] = [
    "experiment_id",
    "experiment_type",
    "method",
    "task",
    "split",
    "eval_context",
    "metric_name",
    "metric_value",
    "is_primary_metric",
    "selection_policy",
    "seed",
    "source_tasks_key",
    "source_tasks_count",
    "interference_delta",
    "interference_metric",
    "interference_base",
    "interference_task_adapter",
    "interference_merged",
    "merge_lambda",
    "learning_rate",
    "lora_r",
    "lora_alpha",
    "num_train_epochs",
    "per_device_train_batch_size",
    "mtl_geometric_mean_interference_delta",
    "mtl_arithmetic_mean_interference_delta",
    "mtl_num_tasks",
]


class ResultsAggregator:
    """Runs registered collectors and produces a unified long-format DataFrame.

    Usage:
        aggregator = ResultsAggregator.from_artifacts_root(Path("artifacts"))
        df = aggregator.to_dataframe()
        df.to_parquet("analysis/results/all_experiments.parquet", index=False)

    Adding a new experiment type:
        1. Create a new collector module.
        2. Pass it via collector_overrides={"new_type": MyCollector(...)}.
        3. Add any new columns to DATAFRAME_COLUMNS with None defaults.
    """

    def __init__(self, collectors: List[ResultsCollector]):
        self.collectors = collectors

    @classmethod
    def from_artifacts_root(
        cls,
        artifacts_root: Path,
        collector_overrides: Optional[Dict[str, ResultsCollector]] = None,
        *,
        # SingleTaskCollector options
        single_task_only_best: bool = True,
        single_task_tasks: Optional[List[str]] = None,
        # MergeCollector options
        merge_splits: Optional[List[str]] = None,
        merge_methods: Optional[List[str]] = None,
        merge_include_subset_evals: bool = False,
        merge_best_lambda_only: bool = False,
    ) -> "ResultsAggregator":
        """Build aggregator with default collectors for all three experiment types.

        Args:
            artifacts_root: Root of the artifacts directory.
            collector_overrides: Dict of {key: collector} to replace defaults.
                                 Keys: "single_task", "mtl", "merge".
        """
        from .single_task_collector import SingleTaskCollector
        from .mtl_collector import MTLCollector
        from .merge_collector import MergeCollector

        artifacts_root = Path(artifacts_root)
        defaults: Dict[str, ResultsCollector] = {
            "single_task": SingleTaskCollector(
                artifacts_root,
                tasks=single_task_tasks,
                only_best=single_task_only_best,
            ),
            "mtl": MTLCollector(artifacts_root),
            "merge": MergeCollector(
                artifacts_root,
                methods=merge_methods,
                splits=merge_splits,
                include_subset_evals=merge_include_subset_evals,
                best_lambda_only=merge_best_lambda_only,
            ),
        }
        if collector_overrides:
            defaults.update(collector_overrides)
        return cls(list(defaults.values()))

    def collect_all(self) -> List[ExperimentResult]:
        """Run all collectors, returning a flat list of ExperimentResult objects."""
        results: List[ExperimentResult] = []
        for collector in self.collectors:
            name = type(collector).__name__
            try:
                batch = collector.collect_all()
                results.extend(batch)
                log.info("[%s] Collected %d results.", name, len(batch))
            except Exception as exc:
                log.warning("[%s] Collection failed: %s", name, exc)
        return results

    def to_dataframe(self) -> "pd.DataFrame":
        """Return a long-format pandas DataFrame with one row per metric."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install it with: pip install pandas"
            )

        rows: List[Dict[str, Any]] = []
        for result in self.collect_all():
            rows.extend(_result_to_rows(result))

        if not rows:
            return pd.DataFrame(columns=DATAFRAME_COLUMNS)

        df = pd.DataFrame(rows)
        # Ensure all expected columns are present with None/NaN for missing ones.
        for col in DATAFRAME_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[DATAFRAME_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Row serialisation
# ---------------------------------------------------------------------------

def _result_to_rows(result: ExperimentResult) -> List[Dict[str, Any]]:
    """Convert one ExperimentResult into a list of row dicts (one per metric)."""
    m = result.metadata
    tr = result.task_result
    hp = result.hyperparameters

    # Shared fields for all rows from this result.
    base: Dict[str, Any] = {
        "experiment_id": m.experiment_id,
        "experiment_type": m.experiment_type,
        "method": m.method,
        "task": tr.task,
        "split": tr.split,
        "eval_context": tr.eval_context,
        # Selection provenance
        "selection_policy": m.selection_policy,
        "selection_metric_name": m.selection_metric_name,
        "selection_metric_value": m.selection_metric_value,
        # Run identity
        "seed": m.seed,
        "config_name": m.config_name,
        "run_id": m.run_id,
        "timestamp": m.timestamp,
        # Task provenance
        "source_tasks": json.dumps(sorted(m.source_tasks)) if m.source_tasks else None,
        "source_tasks_key": m.source_tasks_key,
        "source_tasks_count": len(m.source_tasks) if m.source_tasks else None,
        # Interference delta
        "interference_delta": tr.interference_delta,
        "interference_metric": tr.interference_delta_info.metric_name if tr.interference_delta_info else None,
        "interference_base": tr.interference_delta_info.base_score if tr.interference_delta_info else None,
        "interference_task_adapter": tr.interference_delta_info.task_adapter_score if tr.interference_delta_info else None,
        "interference_merged": tr.interference_delta_info.merged_score if tr.interference_delta_info else None,
        # Merge-specific
        "merge_lambda": m.merge_lambda,
        "merge_tag": m.merge_tag,
        "eval_tag": tr.eval_tag,
        # Hyperparameters
        "learning_rate": hp.learning_rate if hp else None,
        "lora_r": hp.lora_r if hp else None,
        "lora_alpha": hp.lora_alpha if hp else None,
        "num_train_epochs": hp.num_train_epochs if hp else None,
        "per_device_train_batch_size": hp.per_device_train_batch_size if hp else None,
        # MTL aggregates
        "mtl_geometric_mean_interference_delta": result.mtl_geometric_mean_interference_delta,
        "mtl_arithmetic_mean_interference_delta": result.mtl_arithmetic_mean_interference_delta,
        "mtl_num_tasks": result.mtl_num_tasks_with_delta,
        # Filesystem provenance
        "adapter_subdir": m.adapter_subdir,
        "model_path": m.model_path,
        "source_path": m.source_path,
    }

    rows: List[Dict[str, Any]] = []

    # One row per named metric field.
    named_metrics: List[tuple[str, Any]] = [
        ("accuracy", tr.accuracy),
        ("macro_f1", tr.macro_f1),
        ("weighted_f1", tr.weighted_f1),
        ("wer", tr.wer),
        ("loss", tr.loss),
        ("recognized_rate", tr.recognized_rate),
    ]
    for metric_name, metric_value in named_metrics:
        if metric_value is None:
            continue
        rows.append({
            **base,
            "metric_name": metric_name,
            "metric_value": float(metric_value),
            "is_primary_metric": is_primary_metric(tr.task, metric_name),
        })

    # One row per extra_metrics key (ensures no metrics are silently discarded).
    for metric_name, metric_value in (tr.extra_metrics or {}).items():
        if metric_value is None:
            continue
        try:
            rows.append({
                **base,
                "metric_name": str(metric_name),
                "metric_value": float(metric_value),
                "is_primary_metric": is_primary_metric(tr.task, metric_name),
            })
        except (TypeError, ValueError):
            pass

    return rows
