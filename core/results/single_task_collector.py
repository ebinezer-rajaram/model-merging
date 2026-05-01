"""Collector for single-task fine-tuning experiment results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, List, Optional

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

# Metric keys stored in the registry with the "eval_" prefix that should be stripped.
_EVAL_PREFIX = "eval_"


class SingleTaskCollector(ResultsCollector):
    """Reads single-task fine-tuning results from runs_registry.json files.

    Discovery path:
        artifacts/{task}/adapters/{adapter_subdir}/runs_registry.json

    All (task, adapter_subdir) combinations are scanned dynamically —
    no task name or adapter subdir is hardcoded.
    """

    def __init__(
        self,
        artifacts_root: Path,
        tasks: Optional[List[str]] = None,
        only_best: bool = True,
    ):
        """
        Args:
            artifacts_root: Root of the artifacts directory.
            tasks: Restrict collection to these tasks. None = all known tasks.
            only_best: If True (default), only emit the is_best=True run entry
                       per (task, adapter_subdir). If False, emit all entries.
        """
        self.artifacts_root = Path(artifacts_root)
        self.tasks = list(tasks) if tasks else sorted(KNOWN_TASKS)
        self.only_best = only_best

    def collect(self) -> Iterator[ExperimentResult]:
        for task in self.tasks:
            task_dir = self.artifacts_root / task
            adapters_dir = task_dir / "adapters"
            if not adapters_dir.is_dir():
                continue

            for adapter_subdir in sorted(adapters_dir.iterdir()):
                if not adapter_subdir.is_dir():
                    continue
                registry_path = adapter_subdir / "runs_registry.json"
                if not registry_path.exists():
                    continue

                try:
                    yield from self._collect_from_registry(
                        task, adapter_subdir, registry_path
                    )
                except Exception as exc:
                    log.warning(
                        "[SingleTaskCollector] Failed to parse %s: %s",
                        registry_path,
                        exc,
                    )

    def _collect_from_registry(
        self,
        task: str,
        adapter_subdir: Path,
        registry_path: Path,
    ) -> Iterator[ExperimentResult]:
        registry = safe_load_json(registry_path)
        if registry is None:
            return

        metric_for_ranking = registry.get("metric_for_ranking", "")
        runs = registry.get("runs", [])
        if not runs:
            return

        for run_entry in runs:
            if self.only_best and not run_entry.get("is_best", False):
                continue
            try:
                result = self._build_result(
                    task=task,
                    adapter_subdir=adapter_subdir,
                    run_entry=run_entry,
                    metric_for_ranking=metric_for_ranking,
                    registry_path=registry_path,
                )
                if result is not None:
                    yield result
            except Exception as exc:
                log.warning(
                    "[SingleTaskCollector] Failed to build result from run %s in %s: %s",
                    run_entry.get("run_id", "?"),
                    registry_path,
                    exc,
                )

    def _build_result(
        self,
        task: str,
        adapter_subdir: Path,
        run_entry: dict,
        metric_for_ranking: str,
        registry_path: Path,
    ) -> Optional[ExperimentResult]:
        run_id = run_entry.get("run_id")
        timestamp = run_entry.get("timestamp")
        config_hash = run_entry.get("config_hash")

        # Hyperparameters
        hp_raw = run_entry.get("hyperparameters_summary", {}) or {}
        hyperparameters = TrainingHyperparameters(
            learning_rate=_to_float(hp_raw.get("learning_rate")),
            lora_r=_to_int(hp_raw.get("lora_r")),
            lora_alpha=_to_int(hp_raw.get("lora_alpha")),
            num_train_epochs=_to_int(hp_raw.get("num_train_epochs")),
            per_device_train_batch_size=_to_int(hp_raw.get("per_device_train_batch_size")),
        )

        # Metrics — strip "eval_" prefix from keys
        raw_metrics = run_entry.get("metrics", {}) or {}
        metrics = _strip_eval_prefix(raw_metrics)

        # Determine which split these metrics come from.
        # The registry stores validation metrics (the best-checkpoint eval).
        split = "validation"

        # Try to infer seed from a config.yaml in the run directory if available.
        seed = _try_read_seed(adapter_subdir / "runs" / run_id if run_id else None)

        source_tasks = [task]
        source_tasks_key = make_source_tasks_key(source_tasks)
        experiment_id = make_experiment_id(
            "single_task", task, source_tasks, adapter_subdir.name
        )

        # Selection provenance
        selection_metric_value = _to_float(raw_metrics.get(metric_for_ranking))
        if selection_metric_value is None and metric_for_ranking:
            # Try stripped key
            stripped = metric_for_ranking.removeprefix(_EVAL_PREFIX)
            selection_metric_value = _to_float(metrics.get(stripped))

        metadata = ExperimentMetadata(
            experiment_type="single_task",
            experiment_id=experiment_id,
            method=task,
            selection_policy="best_run_metric",
            selection_metric_name=metric_for_ranking or None,
            selection_metric_value=selection_metric_value,
            config_name=config_hash,
            seed=seed,
            run_id=run_id,
            timestamp=timestamp,
            source_tasks=source_tasks,
            source_tasks_key=source_tasks_key,
            adapter_subdir=adapter_subdir.name,
            source_path=str(registry_path),
        )

        task_result = TaskEvalResult(
            task=task,
            split=split,
            eval_context=derive_eval_context(task, source_tasks, None),
            loss=_to_float(metrics.get("loss")),
            accuracy=_to_float(metrics.get("accuracy")),
            macro_f1=_to_float(metrics.get("macro_f1")),
            weighted_f1=_to_float(metrics.get("weighted_f1")),
            wer=_to_float(metrics.get("wer")),
            recognized_rate=_to_float(metrics.get("recognized_rate")),
            num_samples=_to_int(metrics.get("num_samples")),
            runtime_seconds=_to_float(metrics.get("runtime")),
        )

        return ExperimentResult(
            metadata=metadata,
            task_result=task_result,
            hyperparameters=hyperparameters,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_eval_prefix(metrics: dict) -> dict:
    """Return a copy of metrics with 'eval_' prefix stripped from all keys."""
    out = {}
    for k, v in metrics.items():
        stripped = k.removeprefix(_EVAL_PREFIX)
        out[stripped] = v
    return out


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
        return None if (f != f) else f  # filter NaN
    except (TypeError, ValueError):
        return None


def _to_int(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _try_read_seed(run_dir: Optional[Path]) -> Optional[int]:
    """Attempt to read seed from a config.yaml in the run directory."""
    if run_dir is None or not run_dir.is_dir():
        return None
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return None
    try:
        import yaml  # type: ignore
        with config_path.open() as fh:
            cfg = yaml.safe_load(fh) or {}
        seed = cfg.get("seed")
        return int(seed) if seed is not None else None
    except Exception:
        return None
