"""Periodic held-out selection with interference-aware Pareto early stopping."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from core import load_config, prepare_task_for_evaluation, run_evaluation
from experiments.evaluate_task import _prepare_dataset_cache
from merging.evaluation.interference import (
    compute_eval_tag_from_subset,
    maybe_add_interference_delta,
    maybe_compute_interference_baselines,
)
from merging.optimizers.core.common import _maybe_subset_dataset
from merging.runtime.utils import PACKAGE_ROOT, get_task_module


@dataclass(frozen=True)
class HeldoutParetoConfig:
    patience_evals: int
    min_evals_before_stop: int
    hypervolume_min_delta: float
    selection_min_delta: float
    dominance_epsilon: float
    reference_point: List[float]
    selection_criterion: str


@dataclass(frozen=True)
class HeldoutEvalConfig:
    enabled: bool
    split: str
    frequency_updates: int
    batch_size: Optional[int]
    subset: Optional[Dict[str, Any]]
    compute_missing_interference_baselines: bool
    restore_best_checkpoint: bool
    pareto: HeldoutParetoConfig


def resolve_heldout_eval_config(
    *,
    params: Mapping[str, Any],
    tasks: Sequence[str],
    optimization_split: str,
    default_batch_size: int,
    default_patience: int,
    default_threshold: float,
    default_restore_best_checkpoint: bool,
) -> Optional[HeldoutEvalConfig]:
    raw = params.get("heldout_eval")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("optimizer.params.heldout_eval must be a mapping when provided.")

    enabled = bool(raw.get("enabled", False))
    if not enabled:
        return HeldoutEvalConfig(
            enabled=False,
            split=str(raw.get("split", "test")).strip().lower(),
            frequency_updates=int(raw.get("frequency_updates", 100)),
            batch_size=None,
            subset=None,
            compute_missing_interference_baselines=bool(raw.get("compute_missing_interference_baselines", True)),
            restore_best_checkpoint=default_restore_best_checkpoint,
            pareto=HeldoutParetoConfig(
                patience_evals=max(0, int(default_patience)),
                min_evals_before_stop=1,
                hypervolume_min_delta=max(0.0, float(default_threshold)),
                selection_min_delta=max(0.0, float(default_threshold)),
                dominance_epsilon=0.0,
                reference_point=[0.0 for _ in tasks],
                selection_criterion="pareto_hypervolume",
            ),
        )

    split = str(raw.get("split", "test")).strip().lower()
    if not split:
        raise ValueError("optimizer.params.heldout_eval.split must be non-empty.")
    if split == str(optimization_split).strip().lower():
        raise ValueError(
            "optimizer.params.heldout_eval.split must differ from optimizer.params.split "
            "to create a separate held-out set."
        )

    frequency_updates = int(raw.get("frequency_updates", 100))
    if frequency_updates <= 0:
        raise ValueError("optimizer.params.heldout_eval.frequency_updates must be > 0.")

    batch_size_raw = raw.get("batch_size")
    batch_size: Optional[int]
    if batch_size_raw is None:
        batch_size = int(default_batch_size)
    else:
        batch_size = int(batch_size_raw)
        if batch_size <= 0:
            raise ValueError("optimizer.params.heldout_eval.batch_size must be > 0 when provided.")

    subset = raw.get("subset")
    if subset is not None and not isinstance(subset, Mapping):
        raise ValueError("optimizer.params.heldout_eval.subset must be a mapping when provided.")
    subset_map = dict(subset) if isinstance(subset, Mapping) else None

    compute_missing = bool(raw.get("compute_missing_interference_baselines", True))
    restore_best = bool(raw.get("restore_best_checkpoint", default_restore_best_checkpoint))

    pareto_raw = raw.get("pareto", {})
    if pareto_raw is None:
        pareto_raw = {}
    if not isinstance(pareto_raw, Mapping):
        raise ValueError("optimizer.params.heldout_eval.pareto must be a mapping when provided.")

    patience_evals = int(pareto_raw.get("patience_evals", default_patience))
    if patience_evals < 0:
        raise ValueError("optimizer.params.heldout_eval.pareto.patience_evals must be >= 0.")
    min_evals_before_stop = int(pareto_raw.get("min_evals_before_stop", 1))
    if min_evals_before_stop < 1:
        raise ValueError("optimizer.params.heldout_eval.pareto.min_evals_before_stop must be >= 1.")
    hypervolume_min_delta = float(pareto_raw.get("hypervolume_min_delta", default_threshold))
    if hypervolume_min_delta < 0.0:
        raise ValueError("optimizer.params.heldout_eval.pareto.hypervolume_min_delta must be >= 0.")
    selection_min_delta = float(pareto_raw.get("selection_min_delta", hypervolume_min_delta))
    if selection_min_delta < 0.0:
        raise ValueError("optimizer.params.heldout_eval.pareto.selection_min_delta must be >= 0.")
    dominance_epsilon = float(pareto_raw.get("dominance_epsilon", 0.0))
    if dominance_epsilon < 0.0:
        raise ValueError("optimizer.params.heldout_eval.pareto.dominance_epsilon must be >= 0.")
    selection_criterion = str(pareto_raw.get("selection_criterion", "pareto_hypervolume")).strip().lower()
    criterion_aliases = {
        "hypervolume": "pareto_hypervolume",
        "pareto_hypervolume": "pareto_hypervolume",
        "best_single_metric": "best_single_metric",
        "best_single": "best_single_metric",
        "max_interference": "best_single_metric",
        "l2_shortfall": "l2_shortfall",
        "rms_shortfall": "l2_shortfall",
        "l2_rms_shortfall": "l2_shortfall",
        "min_interference_delta": "min_interference_delta",
        "min_interference": "min_interference_delta",
        "min_delta": "min_interference_delta",
        "arithmetic_mean_interference_delta": "arithmetic_mean_interference_delta",
        "mean_interference_delta": "arithmetic_mean_interference_delta",
        "arithmetic_mean_interference": "arithmetic_mean_interference_delta",
        "mean_interference": "arithmetic_mean_interference_delta",
        "mean_delta": "arithmetic_mean_interference_delta",
        "geometric_mean_interference_delta": "geometric_mean_interference_delta",
        "geomean_interference_delta": "geometric_mean_interference_delta",
        "geometric_mean_interference": "geometric_mean_interference_delta",
        "geomean_interference": "geometric_mean_interference_delta",
        "gmean_interference": "geometric_mean_interference_delta",
    }
    if selection_criterion not in criterion_aliases:
        raise ValueError(
            "optimizer.params.heldout_eval.pareto.selection_criterion must be one of: "
            "pareto_hypervolume|best_single_metric|l2_shortfall|min_interference_delta|"
            "arithmetic_mean_interference_delta|geometric_mean_interference_delta."
        )
    selection_criterion = criterion_aliases[selection_criterion]

    reference_raw = pareto_raw.get("reference_point")
    if reference_raw is None:
        reference_point = [0.0 for _ in tasks]
    else:
        if not isinstance(reference_raw, list):
            raise ValueError("optimizer.params.heldout_eval.pareto.reference_point must be a list when provided.")
        if len(reference_raw) != len(tasks):
            raise ValueError(
                "optimizer.params.heldout_eval.pareto.reference_point length must match tasks length. "
                f"Got len(reference_point)={len(reference_raw)}, len(tasks)={len(tasks)}."
            )
        reference_point = [float(x) for x in reference_raw]

    return HeldoutEvalConfig(
        enabled=True,
        split=split,
        frequency_updates=frequency_updates,
        batch_size=batch_size,
        subset=subset_map,
        compute_missing_interference_baselines=compute_missing,
        restore_best_checkpoint=restore_best,
        pareto=HeldoutParetoConfig(
            patience_evals=patience_evals,
            min_evals_before_stop=min_evals_before_stop,
            hypervolume_min_delta=hypervolume_min_delta,
            selection_min_delta=selection_min_delta,
            dominance_epsilon=dominance_epsilon,
            reference_point=reference_point,
            selection_criterion=selection_criterion,
        ),
    )


def _dominates(a: Sequence[float], b: Sequence[float], eps: float) -> bool:
    ge_all = True
    gt_any = False
    for av, bv in zip(a, b):
        if av < (bv - eps):
            ge_all = False
            break
        if av > (bv + eps):
            gt_any = True
    return ge_all and gt_any


def _equal_with_eps(a: Sequence[float], b: Sequence[float], eps: float) -> bool:
    return all(abs(float(x) - float(y)) <= eps for x, y in zip(a, b))


def _compute_hypervolume(frontier: Sequence[Sequence[float]], reference_point: Sequence[float]) -> float:
    if not frontier:
        return 0.0
    dims = len(reference_point)
    axis_coords: List[List[float]] = []
    for d in range(dims):
        coords = [float(reference_point[d])]
        for point in frontier:
            value = max(float(reference_point[d]), float(point[d]))
            coords.append(value)
        unique_sorted = sorted(set(coords))
        if len(unique_sorted) <= 1:
            return 0.0
        axis_coords.append(unique_sorted)

    total = 0.0
    for cell_idx in itertools.product(*[range(len(c) - 1) for c in axis_coords]):
        low = [axis_coords[d][cell_idx[d]] for d in range(dims)]
        high = [axis_coords[d][cell_idx[d] + 1] for d in range(dims)]
        if any(h <= l for l, h in zip(low, high)):
            continue
        covered = False
        for point in frontier:
            if all(float(point[d]) >= high[d] for d in range(dims)):
                covered = True
                break
        if not covered:
            continue
        cell_volume = 1.0
        for l, h in zip(low, high):
            cell_volume *= (h - l)
        total += cell_volume
    return float(total)


class PeriodicHeldoutEvaluator:
    """Runs periodic held-out eval, tracks Pareto frontier, and suggests early stopping."""

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        tasks: List[str],
        config: HeldoutEvalConfig,
        show_summary: bool = True,
    ) -> None:
        self.model = model
        self.processor = processor
        self.tasks = list(tasks)
        self.config = config
        self.show_summary = bool(show_summary)
        self.eval_tag = compute_eval_tag_from_subset(self.config.subset)
        self.setups = self._build_setups()
        if self.config.compute_missing_interference_baselines:
            maybe_compute_interference_baselines(
                tasks=self.tasks,
                split=self.config.split,
                enable_cache=False,
                batch_size=self.config.batch_size,
                show_summary=self.show_summary,
                eval_subset=self.config.subset,
            )

        self.frontier: List[List[float]] = []
        self.eval_history: List[Dict[str, Any]] = []
        self.selection_criterion = self.config.pareto.selection_criterion
        self.best_score: float = float("-inf")
        self.best_hypervolume: float = float("-inf")
        self.best_update_step: Optional[int] = None
        self.no_improve_evals: int = 0
        self.early_stopped: bool = False

    def _compute_selection_score(self, vector: Sequence[float], hypervolume: float) -> float:
        criterion = self.selection_criterion
        if criterion == "pareto_hypervolume":
            return float(hypervolume)
        if criterion == "best_single_metric":
            return float(max(vector)) if vector else float("-inf")
        if criterion == "l2_shortfall":
            if not vector:
                return float("-inf")
            # S(Δ) = -sqrt(mean_t max(0, 1 - Δ_t)^2); maximize S <=> minimize RMS shortfall.
            sq = [(max(0.0, 1.0 - float(v))) ** 2 for v in vector]
            return float(-(sum(sq) / float(len(sq))) ** 0.5)
        if criterion == "min_interference_delta":
            return float(min(vector)) if vector else float("-inf")
        if criterion == "arithmetic_mean_interference_delta":
            return float(sum(float(v) for v in vector) / float(len(vector))) if vector else float("-inf")
        if criterion == "geometric_mean_interference_delta":
            if not vector:
                return float("-inf")
            vals = [float(v) for v in vector]
            if any(v < 0.0 for v in vals):
                return float("-inf")
            if any(v == 0.0 for v in vals):
                return 0.0
            return float(math.exp(sum(math.log(v) for v in vals) / float(len(vals))))
        raise ValueError(f"Unsupported held-out selection criterion '{criterion}'.")

    def _build_setups(self) -> Dict[str, Any]:
        setups: Dict[str, Any] = {}
        for task in self.tasks:
            task_module = get_task_module(task)
            config_path = task_module.get_config_path(PACKAGE_ROOT, None)
            config = load_config(config_path)
            artifact_dirs = task_module.get_artifact_directories(PACKAGE_ROOT)
            config = _prepare_dataset_cache(config, artifact_dirs)
            setup = prepare_task_for_evaluation(task, self.processor, split=self.config.split, config=config)
            setup.dataset = _maybe_subset_dataset(
                setup.dataset,
                task=task,
                eval_subset=self.config.subset if isinstance(self.config.subset, Mapping) else None,
            )
            setups[task] = setup
        return setups

    @staticmethod
    def _sanitize_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, bool, str)):
                sanitized[str(key)] = value
        return sanitized

    def _insert_frontier_point(self, vector: List[float]) -> bool:
        eps = float(self.config.pareto.dominance_epsilon)
        reference = self.config.pareto.reference_point
        clipped = [max(float(reference[i]), float(vector[i])) for i in range(len(vector))]
        for existing in self.frontier:
            if _equal_with_eps(existing, clipped, eps):
                return False
            if _dominates(existing, clipped, eps):
                return False
        retained: List[List[float]] = []
        for existing in self.frontier:
            if _dominates(clipped, existing, eps):
                continue
            retained.append(existing)
        retained.append(clipped)
        self.frontier = retained
        return True

    def maybe_evaluate(
        self,
        *,
        update_step: int,
        apply_coefficients_fn: Callable[[], None],
    ) -> Optional[Dict[str, Any]]:
        if not self.config.enabled:
            return None
        if update_step <= 0 or (update_step % int(self.config.frequency_updates)) != 0:
            return None

        apply_coefficients_fn()
        was_training = bool(getattr(self.model, "training", False))
        self.model.eval()
        try:
            per_task_metrics: Dict[str, Dict[str, Any]] = {}
            for idx, task in enumerate(self.tasks, start=1):
                if self.show_summary:
                    print(
                        "[heldout] "
                        f"evaluating task {idx}/{len(self.tasks)}: {task} "
                        f"(split={self.config.split}, update_step={update_step})"
                    )
                metrics = run_evaluation(
                    self.model,
                    self.setups[task],
                    batch_size=int(self.config.batch_size or 1),
                    store_predictions=False,
                    processor=self.processor,
                )
                maybe_add_interference_delta(
                    task,
                    metrics,
                    self.config.split,
                    False,
                    eval_tag=self.eval_tag,
                )
                safe_metrics = self._sanitize_metrics(metrics)
                per_task_metrics[task] = safe_metrics
                if self.show_summary:
                    numeric_items = [
                        f"{k}={float(v):.4f}"
                        for k, v in safe_metrics.items()
                        if isinstance(v, (int, float))
                    ]
                    if numeric_items:
                        print(f"[heldout] {task}: {', '.join(numeric_items)}")
        finally:
            if was_training:
                self.model.train()

        vector: List[float] = []
        for task in self.tasks:
            value = per_task_metrics[task].get("interference_delta")
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Held-out interference_delta is missing for task '{task}' "
                    f"at split='{self.config.split}'."
                )
            vector.append(float(value))

        min_interference_delta = float(min(vector)) if vector else None
        arithmetic_mean_interference_delta = (
            float(sum(vector) / float(len(vector))) if vector else None
        )
        if not vector:
            geometric_mean_interference_delta = None
        elif any(v < 0.0 for v in vector):
            geometric_mean_interference_delta = None
        elif any(v == 0.0 for v in vector):
            geometric_mean_interference_delta = 0.0
        else:
            geometric_mean_interference_delta = float(
                math.exp(sum(math.log(float(v)) for v in vector) / float(len(vector)))
            )

        is_nondominated = self._insert_frontier_point(vector)
        hv = _compute_hypervolume(self.frontier, self.config.pareto.reference_point)
        hv_delta = hv - self.best_hypervolume if self.best_hypervolume != float("-inf") else hv
        selection_score = self._compute_selection_score(vector, hv)
        score_delta = selection_score - self.best_score if self.best_score != float("-inf") else selection_score
        best_single_metric = float(max(vector)) if vector else None
        l2_shortfall_score = (
            float(-(sum((max(0.0, 1.0 - float(v))) ** 2 for v in vector) / float(len(vector))) ** 0.5)
            if vector
            else None
        )
        improved = (self.best_score == float("-inf")) or (
            score_delta > float(self.config.pareto.selection_min_delta)
        )
        if improved:
            self.best_score = selection_score
            self.best_hypervolume = hv
            self.best_update_step = int(update_step)
            self.no_improve_evals = 0
        else:
            self.no_improve_evals += 1

        eval_entry = {
            "update_step": int(update_step),
            "interference_by_task": {task: float(vector[i]) for i, task in enumerate(self.tasks)},
            "per_task_metrics": per_task_metrics,
            "is_nondominated": bool(is_nondominated),
            "frontier_size": int(len(self.frontier)),
            "hypervolume": float(hv),
            "delta_hypervolume": float(hv_delta),
            "best_hypervolume": float(self.best_hypervolume),
            "selection_criterion": self.selection_criterion,
            "selection_score": float(selection_score),
            "delta_selection_score": float(score_delta),
            "best_selection_score": float(self.best_score),
            "best_single_metric": best_single_metric,
            "min_interference_delta": min_interference_delta,
            "arithmetic_mean_interference_delta": arithmetic_mean_interference_delta,
            "geometric_mean_interference_delta": geometric_mean_interference_delta,
            "l2_shortfall_score": l2_shortfall_score,
        }
        self.eval_history.append(eval_entry)
        eval_count = len(self.eval_history)
        should_stop = False
        if (
            self.config.pareto.patience_evals > 0
            and eval_count >= int(self.config.pareto.min_evals_before_stop)
            and self.no_improve_evals >= int(self.config.pareto.patience_evals)
        ):
            should_stop = True
            self.early_stopped = True

        eval_entry["eval_count"] = int(eval_count)
        eval_entry["no_improve_evals"] = int(self.no_improve_evals)
        eval_entry["should_stop"] = bool(should_stop)
        return eval_entry


__all__ = [
    "HeldoutParetoConfig",
    "HeldoutEvalConfig",
    "resolve_heldout_eval_config",
    "PeriodicHeldoutEvaluator",
]
