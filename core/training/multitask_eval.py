"""Evaluation helpers for joint multi-task training."""

from __future__ import annotations

import csv
import json
import math
import random
import time
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from core.evaluation.eval_utils import run_evaluation
from core.data.io_utils import ensure_dir
from merging.evaluation.interference import (
    TASK_METRICS,
    compute_eval_tag_from_subset,
    maybe_add_interference_delta,
    maybe_compute_interference_baselines,
    oriented_score,
)

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - backend availability is environment-dependent
    plt = None


def aggregate_interference_delta_geometric(values: Sequence[float]) -> float:
    """Aggregate interference deltas with repo-consistent geometric-mean semantics."""
    vals = [float(v) for v in values]
    if not vals:
        return float("-inf")
    if any(v < 0.0 for v in vals):
        return float("-inf")
    if any(v == 0.0 for v in vals):
        return 0.0
    return float(math.exp(sum(math.log(v) for v in vals) / float(len(vals))))


def aggregate_interference_delta_arithmetic(values: Sequence[float]) -> float:
    """Aggregate interference deltas with arithmetic mean."""
    vals = [float(v) for v in values]
    if not vals:
        return float("-inf")
    return float(sum(vals) / float(len(vals)))


@dataclass
class MultiTaskEvalResult:
    """Structured per-evaluation-cycle summary."""

    metrics: Dict[str, float]
    per_task_metrics: Dict[str, Dict[str, Any]]
    aggregate_delta: float


class MultiTaskEvaluator:
    """Run per-task evals and derive multi-task selection metrics."""

    def __init__(
        self,
        *,
        tasks: Sequence[str],
        eval_setups: Mapping[str, Any],
        task_generation_kwargs: Mapping[str, Mapping[str, Any]],
        split: str,
        batch_size: int,
        compute_missing_interference_baselines: bool,
        metrics_dir: Path,
        selection_criterion: str = "geometric_mean_interference_delta",
        selection_mode: str = "mtl_interference",
        selected_primary_tasks: Optional[Sequence[str]] = None,
        use_cache: bool = True,
        eval_subset: Optional[Dict[str, Any]] = None,
        wandb_project: Optional[str] = None,
        auto_plot: bool = True,
        dataloader_num_workers: int = 0,
        dataloader_prefetch_factor: Optional[int] = None,
    ):
        self.tasks = [str(task) for task in tasks]
        self.task_generation_kwargs = {
            str(task): dict(kwargs) for task, kwargs in task_generation_kwargs.items()
        }
        self.split = str(split)
        self.batch_size = int(batch_size)
        self.compute_missing_interference_baselines = bool(compute_missing_interference_baselines)
        self.metrics_dir = ensure_dir(Path(metrics_dir))
        self.selection_criterion = str(selection_criterion).strip().lower()
        if self.selection_criterion not in {
            "geometric_mean_interference_delta",
            "arithmetic_mean_interference_delta",
        }:
            raise ValueError(
                "selection_criterion must be one of "
                "{geometric_mean_interference_delta, arithmetic_mean_interference_delta}."
            )
        self.selection_mode = str(selection_mode).strip().lower()
        if self.selection_mode not in {"mtl_interference", "added_task_metric"}:
            raise ValueError("selection_mode must be one of {mtl_interference, added_task_metric}.")
        self.selected_primary_tasks = [
            str(task).strip().lower()
            for task in (selected_primary_tasks or [])
            if str(task).strip()
        ]
        self.use_cache = bool(use_cache)
        self.eval_subset = dict(eval_subset) if isinstance(eval_subset, Mapping) else None
        self.eval_tag = compute_eval_tag_from_subset(self.eval_subset)
        self.wandb_project = str(wandb_project) if wandb_project else None
        self.auto_plot = bool(auto_plot)
        self.dataloader_num_workers = int(dataloader_num_workers)
        self.dataloader_prefetch_factor = (
            int(dataloader_prefetch_factor)
            if dataloader_prefetch_factor is not None
            else None
        )

        self.history_jsonl_path = self.metrics_dir / "mtl_eval_history.jsonl"
        self.history_csv_path = self.metrics_dir / "mtl_eval_history.csv"
        self.selection_plot_path = self.metrics_dir / "mtl_selection_delta_trends.png"
        self.per_task_delta_plot_path = self.metrics_dir / "mtl_per_task_interference_delta.png"
        self.per_task_primary_plot_path = self.metrics_dir / "mtl_per_task_primary_metric.png"
        self.eval_setups = self._prepare_eval_setups(dict(eval_setups))
        print(
            "🧾 MTL evaluator configured: "
            f"split={self.split}, batch_size={self.batch_size}, "
            f"selection_criterion={self.selection_criterion}, "
            f"selection_mode={self.selection_mode}, "
            f"eval_subset_tag={self.eval_tag or 'none'}, "
            f"auto_plot={self.auto_plot}"
        )

    def _prepare_eval_setups(self, setups: Dict[str, Any]) -> Dict[str, Any]:
        if not self.eval_subset or not bool(self.eval_subset.get("enabled", True)):
            return setups
        print("✂️ Applying selection eval subset to per-task validation datasets...")
        for task, setup in setups.items():
            before = len(setup.dataset)
            setup.dataset = self._subset_dataset(task=task, setup=setup)
            after = len(setup.dataset)
            print(f"   • {task}: {before} -> {after} samples")
        return setups

    def _subset_dataset(self, *, task: str, setup: Any):
        dataset = setup.dataset
        max_samples = self.eval_subset.get("max_samples") if isinstance(self.eval_subset, Mapping) else None
        shuffle = bool(self.eval_subset.get("shuffle", False)) if isinstance(self.eval_subset, Mapping) else False
        seed = int(self.eval_subset.get("seed", 0) or 0) if isinstance(self.eval_subset, Mapping) else 0
        stratified = bool(self.eval_subset.get("stratified", False)) if isinstance(self.eval_subset, Mapping) else False
        label_column = (
            self.eval_subset.get("label_column")
            if isinstance(self.eval_subset, Mapping)
            else None
        )

        per_task = self.eval_subset.get("per_task") if isinstance(self.eval_subset, Mapping) else None
        if isinstance(per_task, Mapping) and task in per_task:
            override = per_task[task]
            if isinstance(override, Mapping):
                if override.get("max_samples") is not None:
                    max_samples = int(override.get("max_samples"))
                if "shuffle" in override and override.get("shuffle") is not None:
                    shuffle = bool(override.get("shuffle"))
                if "seed" in override and override.get("seed") is not None:
                    seed = int(override.get("seed"))
                if "stratified" in override and override.get("stratified") is not None:
                    stratified = bool(override.get("stratified"))
                if "label_column" in override and override.get("label_column") is not None:
                    label_column = str(override.get("label_column"))
                if "stratify_by" in override and override.get("stratify_by") is not None:
                    label_column = str(override.get("stratify_by"))
            elif isinstance(override, (int, float)):
                max_samples = int(override)

        if str(task).strip().lower() == "asr":
            stratified = False
            label_column = None

        if max_samples is None:
            return dataset
        max_samples = int(max_samples)
        if max_samples <= 0:
            raise ValueError("selection_eval_subset.max_samples must be > 0.")
        if max_samples >= len(dataset):
            return dataset

        if stratified:
            resolved_label_column = str(label_column or "label")
            if resolved_label_column not in set(getattr(dataset, "column_names", []) or []):
                print(
                    f"   ⚠️ {task}: stratified subset requested but label column "
                    f"'{resolved_label_column}' not found; falling back to non-stratified subset."
                )
                stratified = False
            else:
                rng = random.Random(seed)
                labels_arr = np.array(dataset[resolved_label_column])
                unique_labels = np.unique(labels_arr)
                by_label = {lbl: np.where(labels_arr == lbl)[0].tolist() for lbl in unique_labels}

                total = len(labels_arr)
                targets = []
                for key, idxs in by_label.items():
                    frac = (len(idxs) / float(total)) * float(max_samples)
                    targets.append((key, frac))

                alloc = {key: int(frac) for key, frac in targets}
                remainder = max_samples - sum(alloc.values())
                for key, frac in sorted(targets, key=lambda x: x[1] - int(x[1]), reverse=True):
                    if remainder <= 0:
                        break
                    alloc[key] += 1
                    remainder -= 1

                selected = []
                leftovers = []
                for key, idxs in by_label.items():
                    k = min(alloc.get(key, 0), len(idxs))
                    if shuffle:
                        picks = rng.sample(idxs, k=k) if k > 0 else []
                    else:
                        picks = list(idxs[:k])
                    selected.extend(picks)
                    pick_set = set(picks)
                    leftovers.extend(i for i in idxs if i not in pick_set)

                if len(selected) < max_samples and leftovers:
                    missing = max_samples - len(selected)
                    extra = rng.sample(leftovers, k=min(missing, len(leftovers))) if shuffle else list(leftovers[:missing])
                    selected.extend(extra)
                indices = sorted(selected)

        if not stratified:
            if shuffle:
                rng = random.Random(seed)
                indices = sorted(rng.sample(range(len(dataset)), k=max_samples))
            else:
                indices = list(range(max_samples))
        else:
            print(f"   📊 {task}: using stratified subset via label_column='{resolved_label_column}'.")

        if getattr(setup, "apply_subset_indices", None) is not None:
            setup.apply_subset_indices(indices)
        return dataset.select(indices)

    def evaluate(self, *, model: Any, processor: Any, global_step: int) -> MultiTaskEvalResult:
        print(
            "🧮 MTL selection evaluation: "
            f"step={global_step}, split={self.split}, tasks={len(self.tasks)}"
        )
        if self.compute_missing_interference_baselines:
            print(
                "🧱 Checking interference baselines "
                f"(cache-first, subset_tag={self.eval_tag or 'none'})..."
            )
            maybe_compute_interference_baselines(
                tasks=self.tasks,
                split=self.split,
                enable_cache=self.use_cache,
                batch_size=self.batch_size,
                show_summary=True,
                eval_subset=self.eval_subset,
            )

        per_task_metrics: Dict[str, Dict[str, Any]] = {}
        deltas = []
        flat_metrics: Dict[str, float] = {}

        for task in self.tasks:
            setup = self.eval_setups[task]
            task_start = time.time()
            print(f"   ▶ Evaluating task '{task}' on {len(setup.dataset)} samples...")
            metrics = run_evaluation(
                model,
                setup,
                batch_size=self.batch_size,
                generation_kwargs=self.task_generation_kwargs.get(task, {}),
                output_dir=self.metrics_dir / "tmp_eval" / task,
                processor=processor,
                dataloader_num_workers=self.dataloader_num_workers,
                dataloader_prefetch_factor=self.dataloader_prefetch_factor,
            )
            maybe_add_interference_delta(
                task=task,
                metrics=metrics,
                split=self.split,
                show_summary=False,
                eval_subset=self.eval_subset,
            )
            per_task_metrics[task] = dict(metrics)

            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    flat_metrics[f"eval_{task}_{key}"] = float(value)

            value = metrics.get("interference_delta")
            if isinstance(value, (int, float)):
                deltas.append(float(value))
                delta_str = f"{float(value):.4f}"
            else:
                delta_str = "n/a"
            elapsed = time.time() - task_start
            print(f"   ✅ {task}: interference_delta={delta_str}, elapsed={elapsed:.1f}s")

        geometric = aggregate_interference_delta_geometric(deltas)
        arithmetic = aggregate_interference_delta_arithmetic(deltas)
        aggregate = geometric if self.selection_criterion == "geometric_mean_interference_delta" else arithmetic

        flat_metrics["eval_mtl_geometric_mean_interference_delta"] = float(geometric)
        flat_metrics["eval_mtl_arithmetic_mean_interference_delta"] = float(arithmetic)
        flat_metrics["eval_mtl_num_tasks_with_delta"] = float(len(deltas))

        if self.selected_primary_tasks:
            primary_values = []
            for task in self.selected_primary_tasks:
                metric_spec = TASK_METRICS.get(task)
                if metric_spec is None:
                    continue
                metric_name, higher_is_better = metric_spec
                task_metrics = per_task_metrics.get(task, {})
                raw_value = task_metrics.get(metric_name)
                if not isinstance(raw_value, (int, float)):
                    continue
                primary_values.append(oriented_score(float(raw_value), higher_is_better))
            if primary_values:
                flat_metrics["eval_added_tasks_primary_oriented_mean"] = float(
                    sum(primary_values) / float(len(primary_values))
                )
                flat_metrics["eval_added_tasks_primary_num_tasks"] = float(len(primary_values))

        selected_value = aggregate
        if self.selection_mode == "added_task_metric":
            selected_value = float(flat_metrics.get("eval_added_tasks_primary_oriented_mean", float("-inf")))

        self._append_history(global_step=global_step, metrics=flat_metrics)
        self._log_to_wandb(global_step=global_step, flat_metrics=flat_metrics)
        print(
            "📌 MTL aggregate: "
            f"arithmetic={arithmetic:.4f}, geometric={geometric}, "
            f"selected({self.selection_mode})={selected_value}"
        )

        return MultiTaskEvalResult(
            metrics=flat_metrics,
            per_task_metrics=per_task_metrics,
            aggregate_delta=float(selected_value),
        )

    def _append_history(self, *, global_step: int, metrics: Mapping[str, float]) -> None:
        record = {"step": int(global_step), **{k: float(v) for k, v in metrics.items()}}

        with self.history_jsonl_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

        csv_exists = self.history_csv_path.exists()
        with self.history_csv_path.open("a", newline="") as handle:
            fieldnames = ["step"] + sorted(k for k in record.keys() if k != "step")
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not csv_exists:
                writer.writeheader()
            writer.writerow({field: record.get(field) for field in fieldnames})
        self._refresh_plots_best_effort()

    def _refresh_plots_best_effort(self) -> None:
        if not self.auto_plot:
            return
        if plt is None:
            print("⚠️ Matplotlib not available; skipping MTL plot refresh.")
            return
        try:
            steps, series = self._load_eval_history_series()
            if not steps or not series:
                return

            self._plot_selection_trends(steps=steps, series=series)
            self._plot_per_task_interference_deltas(steps=steps, series=series)
            self._plot_per_task_primary_metrics(steps=steps, series=series)
            print(
                "🖼️ Refreshed MTL plots: "
                f"{self.selection_plot_path.name}, "
                f"{self.per_task_delta_plot_path.name}, "
                f"{self.per_task_primary_plot_path.name}"
            )
        except Exception as exc:
            print(f"⚠️ Failed to refresh MTL plots: {exc}")

    def _load_eval_history_series(self) -> tuple[list[int], dict[str, list[Optional[float]]]]:
        if not self.history_csv_path.exists():
            return [], {}
        steps: list[int] = []
        series: dict[str, list[Optional[float]]] = {}

        with self.history_csv_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            if not fieldnames:
                return [], {}

            for field in fieldnames:
                if field != "step":
                    series[field] = []

            for row in reader:
                raw_step = row.get("step")
                if raw_step is None or str(raw_step).strip() == "":
                    continue
                try:
                    step = int(float(raw_step))
                except Exception:
                    continue

                steps.append(step)
                for field in series:
                    raw_value = row.get(field)
                    if raw_value is None or str(raw_value).strip() == "":
                        series[field].append(None)
                        continue
                    try:
                        series[field].append(float(raw_value))
                    except Exception:
                        series[field].append(None)
        return steps, series

    def _plot_selection_trends(self, *, steps: Sequence[int], series: Mapping[str, Sequence[Optional[float]]]) -> None:
        selection_series = {}
        for metric_name in (
            "eval_mtl_arithmetic_mean_interference_delta",
            "eval_mtl_geometric_mean_interference_delta",
        ):
            values = series.get(metric_name)
            if values:
                selection_series[metric_name] = values
        self._plot_metric_lines(
            steps=steps,
            series_by_name=selection_series,
            plot_path=self.selection_plot_path,
            title="MTL Selection Delta Trends",
            ylabel="Interference Delta",
        )

    def _plot_per_task_interference_deltas(
        self,
        *,
        steps: Sequence[int],
        series: Mapping[str, Sequence[Optional[float]]],
    ) -> None:
        task_series = {}
        for task in self.tasks:
            metric_name = f"eval_{task}_interference_delta"
            values = series.get(metric_name)
            if values:
                task_series[metric_name] = values
        self._plot_metric_lines(
            steps=steps,
            series_by_name=task_series,
            plot_path=self.per_task_delta_plot_path,
            title="Per-Task Interference Delta",
            ylabel="Interference Delta",
        )

    def _plot_per_task_primary_metrics(
        self,
        *,
        steps: Sequence[int],
        series: Mapping[str, Sequence[Optional[float]]],
    ) -> None:
        task_series = {}
        for task in self.tasks:
            metric_key = self._primary_metric_name_for_task(task)
            metric_name = f"eval_{task}_{metric_key}"
            values = series.get(metric_name)
            if not values:
                fallback_name = f"eval_{task}_interference_delta"
                values = series.get(fallback_name)
                metric_name = fallback_name
            if values:
                task_series[metric_name] = values
        self._plot_metric_lines(
            steps=steps,
            series_by_name=task_series,
            plot_path=self.per_task_primary_plot_path,
            title="Per-Task Primary Metric Trend",
            ylabel="Metric Value",
        )

    @staticmethod
    def _primary_metric_name_for_task(task: str) -> str:
        if task == "asr":
            return "wer"
        if task in {"emotion", "intent", "kws", "langid", "speaker_id", "speaker_ver", "vocalsound"}:
            return "accuracy"
        return "interference_delta"

    def _plot_metric_lines(
        self,
        *,
        steps: Sequence[int],
        series_by_name: Mapping[str, Sequence[Optional[float]]],
        plot_path: Path,
        title: str,
        ylabel: str,
    ) -> None:
        if not series_by_name:
            return
        fig, ax = plt.subplots(1, 1, figsize=(11, 5))
        colors = plt.cm.tab10.colors
        plotted = 0

        for metric_name, values in sorted(series_by_name.items()):
            points = [
                (step, value)
                for step, value in zip(steps, values)
                if isinstance(value, (int, float)) and not math.isnan(float(value))
            ]
            if not points:
                continue

            xs = [item[0] for item in points]
            ys = [float(item[1]) for item in points]
            ax.plot(xs, ys, linewidth=2, alpha=0.9, color=colors[plotted % len(colors)], label=metric_name)
            plotted += 1

        if plotted == 0:
            plt.close(fig)
            return

        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    def _log_to_wandb(self, *, global_step: int, flat_metrics: Mapping[str, float]) -> None:
        try:
            import wandb  # type: ignore
        except Exception:
            return

        if wandb.run is None:
            return

        payload: Dict[str, float] = {}
        for key, value in flat_metrics.items():
            if not isinstance(value, (int, float)):
                continue
            if key.startswith("eval_mtl_"):
                payload[f"mtl/{key.replace('eval_mtl_', '', 1)}"] = float(value)
            elif key.startswith("eval_"):
                parts = key.split("_", 2)
                if len(parts) >= 3:
                    _, task, metric = parts[0], parts[1], parts[2]
                    payload[f"task/{task}/{metric}"] = float(value)
                else:
                    payload[key] = float(value)

        if payload:
            wandb.log(payload, step=int(global_step))


__all__ = [
    "aggregate_interference_delta_geometric",
    "MultiTaskEvalResult",
    "MultiTaskEvaluator",
]
