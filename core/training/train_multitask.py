"""Joint multi-task LoRA training entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping
from transformers import EarlyStoppingCallback

from core.config.multitask_schema import MultiTaskConfig, parse_multitask_config
from core.data.io_utils import ensure_dir, load_config
from core.evaluation.eval_utils import prepare_task_for_evaluation
from core.evaluation.evaluate_task import prepare_dataset_cache
from core.models.models import create_lora_config_from_dict, load_qwen_model
from core.training.trainer import CustomTrainer, save_artifacts, save_history_to_csv
from core.training.training_config import (
    build_early_stopping_kwargs,
    build_training_arguments,
    parse_training_config,
)
from core.utils.seed_utils import set_global_seed
from core.training.multitask_eval import MultiTaskEvaluator
from core.training.multitask_sampler import (
    MultiTaskDataset,
    TemperatureMultiTaskBatchSampler,
    estimate_batches_per_epoch,
)
from core.training.run_manager import RunManager
from merging.runtime.utils import PACKAGE_ROOT, get_task_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train shared LoRA with joint multi-task sampling.")
    parser.add_argument("--config", required=True, help="Path to MTL YAML config.")
    return parser.parse_args()


def _setup_wandb(training_cfg: Mapping[str, Any], logging_cfg: Mapping[str, Any]) -> None:
    os.environ["WANDB_DIR"] = str(PACKAGE_ROOT / "logs" / "wandb")
    os.environ["TENSORBOARD_LOG_DIR"] = str(PACKAGE_ROOT / "logs" / "runs")

    report_to = list(training_cfg.get("report_to", []))
    if "wandb" not in report_to:
        return

    project = str(logging_cfg.get("wandb_project", "speech-merging-mtl"))
    os.environ["WANDB_PROJECT"] = project


@dataclass
class BootstrappedTask:
    name: str
    config: Dict[str, Any]
    train_setup: Any
    eval_setup: Any
    generation_kwargs: Dict[str, Any]


class MultiTaskCollator:
    """Route homogeneous task batches to their task-specific collator."""

    def __init__(self, collators_by_task: Mapping[str, Any]):
        self.collators_by_task = dict(collators_by_task)

    def __call__(self, features):
        if not features:
            raise ValueError("Received empty feature batch.")

        task_names = [str(feature.get("__task_name", "")) for feature in features]
        unique = sorted(set(task_names))
        if len(unique) != 1:
            raise ValueError(f"Expected homogeneous task batch; got tasks={unique}")

        task = unique[0]
        if task not in self.collators_by_task:
            raise KeyError(f"No collator configured for task '{task}'.")

        cleaned = []
        for feature in features:
            row = dict(feature)
            row.pop("__task_name", None)
            cleaned.append(row)

        return self.collators_by_task[task](cleaned)


class MultiTaskTrainer(CustomTrainer):
    """Trainer that augments eval with per-task interference-aware metrics."""

    def __init__(self, *args, multitask_evaluator: MultiTaskEvaluator, **kwargs):
        self.multitask_evaluator = multitask_evaluator
        super().__init__(*args, **kwargs)

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        # HF Trainer runs callback_handler.on_evaluate() inside super().evaluate().
        # Temporarily remove EarlyStoppingCallback so it can see the augmented
        # MTL metrics after we inject them below.
        callback_list = list(self.callback_handler.callbacks)
        early_stoppers = [cb for cb in callback_list if isinstance(cb, EarlyStoppingCallback)]
        if early_stoppers:
            self.callback_handler.callbacks = [cb for cb in callback_list if not isinstance(cb, EarlyStoppingCallback)]

        try:
            metrics = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            if early_stoppers:
                self.callback_handler.callbacks = callback_list

        extra = self.multitask_evaluator.evaluate(
            model=self.model,
            processor=self.processing_class,
            global_step=int(self.state.global_step),
        )
        metrics.update(extra.metrics)

        self.log(extra.metrics)

        # Re-run early stopping with the complete metric dict including
        # eval_mtl_geometric_mean_interference_delta.
        for stopper in early_stoppers:
            maybe_control = stopper.on_evaluate(
                self.args,
                self.state,
                self.control,
                metrics=metrics,
            )
            if maybe_control is not None:
                self.control = maybe_control
        return metrics


def _load_bootstrapped_tasks(
    *,
    cfg: MultiTaskConfig,
    processor: Any,
) -> Dict[str, BootstrappedTask]:
    tasks: Dict[str, BootstrappedTask] = {}

    for entry in cfg.tasks:
        task = entry.name
        print(f"🔧 Bootstrapping task '{task}' (config={entry.config or 'default'})...")
        task_module = get_task_module(task)
        config_path = task_module.get_config_path(PACKAGE_ROOT, entry.config)
        task_cfg = load_config(config_path)

        artifact_dirs = task_module.get_artifact_directories(PACKAGE_ROOT)
        task_cfg = prepare_dataset_cache(task_cfg, artifact_dirs)

        train_setup = prepare_task_for_evaluation(task, processor, split="train", config=task_cfg)
        eval_setup = prepare_task_for_evaluation(
            task,
            processor,
            split=cfg.training.selection_split,
            config=task_cfg,
        )

        generation_kwargs = dict(task_cfg.get("training", {}).get("generation_kwargs", {}))

        tasks[task] = BootstrappedTask(
            name=task,
            config=task_cfg,
            train_setup=train_setup,
            eval_setup=eval_setup,
            generation_kwargs=generation_kwargs,
        )
        try:
            train_size = len(train_setup.dataset)
        except Exception:
            train_size = -1
        try:
            eval_size = len(eval_setup.dataset)
        except Exception:
            eval_size = -1
        print(
            f"   ✅ {task}: train_samples={train_size}, "
            f"{cfg.training.selection_split}_samples={eval_size}"
        )

    return tasks


def _slugify_task_name(task: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(task).strip().lower()).strip("_")
    return value or "task"


def _build_task_set_slug(task_names: list[str], *, mode: str = "sorted_names") -> str:
    if mode != "sorted_names":
        raise ValueError(f"Unsupported task_set_slug_mode '{mode}'. Expected 'sorted_names'.")
    names = sorted(_slugify_task_name(task) for task in task_names)
    return "_".join(names) if names else "empty_task_set"


def _resolve_mtl_paths(
    *,
    adapter_subdir: str,
    task_names: list[str],
    layout: str = "task_set",
    task_set_slug_mode: str = "sorted_names",
    artifacts_root: Path | None = None,
) -> Dict[str, Path]:
    base_dir = artifacts_root if artifacts_root is not None else (PACKAGE_ROOT / "artifacts" / "mtl")
    base = ensure_dir(base_dir)
    if layout != "task_set":
        raise ValueError(f"Unsupported MTL artifacts.layout '{layout}'. Expected 'task_set'.")

    task_set_slug = _build_task_set_slug(task_names, mode=task_set_slug_mode)
    task_set_root = ensure_dir(base / f"{len(task_names)}_task" / task_set_slug)
    adapters = ensure_dir(task_set_root / "adapters")
    metrics = ensure_dir(task_set_root / "metrics")

    output_dir = ensure_dir(adapters / adapter_subdir)
    return {
        "base": task_set_root,
        "global_base": base,
        "task_set_slug": Path(task_set_slug),
        "adapters": adapters,
        "metrics": metrics,
        "output_dir": output_dir,
        "history_csv": metrics / "mtl_training_history.csv",
        "config_dump": metrics / "mtl_config_resolved.json",
        "final_adapter": output_dir / "final",
    }


def _copy_mtl_metrics_to_subdirs(run_manager: RunManager, metrics_dir: Path) -> None:
    """Copy MTL metrics from best/latest runs into metrics/best and metrics/latest."""
    mtl_files = [
        "mtl_eval_history.jsonl",
        "mtl_eval_history.csv",
        "mtl_selection_delta_trends.png",
        "mtl_per_task_interference_delta.png",
        "mtl_per_task_primary_metric.png",
        "validation_metrics.json",
        "metrics.json",
        "training_history.csv",
    ]
    for label, get_path in [
        ("best", run_manager.get_best_run_path),
        ("latest", run_manager.get_latest_run_path),
    ]:
        run_path = get_path()
        if run_path is None:
            continue
        dest_dir = ensure_dir(metrics_dir / label)
        for fname in mtl_files:
            src = run_path / fname
            if src.exists():
                shutil.copy(src, dest_dir / fname)


def _load_mtl_config(config_path: Path) -> tuple[MultiTaskConfig, Dict[str, Any]]:
    raw = load_config(config_path, base_config_path=None)
    parsed = parse_multitask_config(raw)
    return parsed, raw


def run_multitask_training(config_path: Path) -> None:
    cfg, raw_cfg = _load_mtl_config(config_path)
    set_global_seed(int(cfg.seed))

    training_cfg_raw = dict(raw_cfg.get("training", {}))
    logging_cfg_raw = dict(raw_cfg.get("logging", {}))
    _setup_wandb(training_cfg_raw, logging_cfg_raw)

    model_path = PACKAGE_ROOT / cfg.model.path
    lora_config = None
    if cfg.model.lora is not None:
        lora_config = create_lora_config_from_dict(cfg.model.lora.model_dump())

    model, processor = load_qwen_model(model_path, lora_config=lora_config)

    tasks = _load_bootstrapped_tasks(cfg=cfg, processor=processor)
    task_names = list(tasks.keys())

    train_datasets = {task: tasks[task].train_setup.dataset for task in task_names}
    eval_setups = {task: tasks[task].eval_setup for task in task_names}
    collators = {task: tasks[task].train_setup.data_collator for task in task_names}
    generation_kwargs = {task: tasks[task].generation_kwargs for task in task_names}
    train_weights = {entry.name: float(entry.train_weight) for entry in cfg.tasks}

    multitask_train_dataset = MultiTaskDataset(train_datasets)
    anchor_task = task_names[0]
    anchor_eval_dataset = MultiTaskDataset({anchor_task: tasks[anchor_task].eval_setup.dataset})

    paths = _resolve_mtl_paths(
        adapter_subdir=cfg.artifacts.adapter_subdir,
        task_names=task_names,
        layout=cfg.artifacts.layout,
        task_set_slug_mode=cfg.artifacts.task_set_slug_mode,
    )
    print(f"📁 MTL artifact root: {paths['base']}")

    task_defaults = {
        "metric_for_best_model": "mtl_geometric_mean_interference_delta",
        "greater_is_better": True,
        "length_column_name": "duration",
        "load_best_model_at_end": True,
        "eval_strategy": "steps",
        "save_strategy": "steps",
    }
    train_config = parse_training_config(
        training_cfg_raw,
        num_train_examples=len(multitask_train_dataset),
        task_defaults=task_defaults,
    )

    training_args = build_training_arguments(train_config, output_dir=str(paths["output_dir"]))
    early_stopping_kwargs = build_early_stopping_kwargs(train_config)

    batches_per_epoch = estimate_batches_per_epoch(
        total_examples=len(multitask_train_dataset),
        batch_size=int(training_args.per_device_train_batch_size),
        drop_last=bool(training_args.dataloader_drop_last),
    )

    sampler = TemperatureMultiTaskBatchSampler(
        task_to_indices=multitask_train_dataset.task_to_global_indices,
        task_weights=train_weights,
        batch_size=int(training_args.per_device_train_batch_size),
        temperature=float(cfg.training.sampling.temperature),
        num_batches=batches_per_epoch,
        drop_last=bool(training_args.dataloader_drop_last),
        seed=int(cfg.seed),
    )
    sampling_summary = ", ".join(
        f"{task}={sampler.task_probabilities.get(task, 0.0):.3f}" for task in task_names
    )
    print(f"🧪 Task sampling probabilities (temperature={cfg.training.sampling.temperature}): {sampling_summary}")

    evaluator = MultiTaskEvaluator(
        tasks=task_names,
        eval_setups=eval_setups,
        task_generation_kwargs=generation_kwargs,
        split=cfg.training.selection_split,
        batch_size=int(training_args.per_device_eval_batch_size),
        compute_missing_interference_baselines=bool(cfg.training.compute_missing_interference_baselines),
        metrics_dir=paths["metrics"],
        selection_criterion=cfg.training.selection_criterion,
        use_cache=True,
        eval_subset=(dict(cfg.training.selection_eval_subset) if cfg.training.selection_eval_subset else None),
        wandb_project=(cfg.logging.wandb_project if cfg.logging is not None else None),
        auto_plot=(bool(cfg.metrics.auto_plot) if cfg.metrics is not None else True),
    )

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=multitask_train_dataset,
        eval_dataset=anchor_eval_dataset,
        data_collator=MultiTaskCollator(collators),
        processing_class=processor,
        compute_metrics=None,
        callbacks=[EarlyStoppingCallback(**early_stopping_kwargs)],
        generation_kwargs=train_config.generation_kwargs,
        enable_generation=False,
        custom_sampler=sampler,
        multitask_evaluator=evaluator,
    )

    print(
        "🚀 Starting multi-task LoRA fine-tuning: "
        f"tasks={task_names}, temperature={cfg.training.sampling.temperature}, "
        f"selection={cfg.training.selection_criterion}/{cfg.training.selection_split}"
    )
    trainer.train(resume_from_checkpoint=train_config.resume_from_checkpoint)

    print("🎯 Running final multi-task evaluation...")
    final_metrics = evaluator.evaluate(
        model=trainer.model,
        processor=processor,
        global_step=int(trainer.state.global_step),
    )
    trainer.log(final_metrics.metrics)

    run_manager = RunManager(
        adapter_dir=paths["output_dir"],
        metric_for_ranking=f"eval_mtl_{cfg.training.selection_criterion}",
        greater_is_better=True,
    )

    run_dir = run_manager.create_run_directory()
    print(f"📁 Created MTL run directory: {run_dir.name}")

    run_history_csv = run_dir / "training_history.csv"
    save_history_to_csv(trainer, run_history_csv)
    save_artifacts(trainer, processor, run_dir)

    with (run_dir / "mtl_config_resolved.json").open("w") as handle:
        json.dump(raw_cfg, handle, indent=2, sort_keys=True)

    for fname in [
        "mtl_eval_history.jsonl",
        "mtl_eval_history.csv",
        "mtl_selection_delta_trends.png",
        "mtl_per_task_interference_delta.png",
        "mtl_per_task_primary_metric.png",
    ]:
        src = paths["metrics"] / fname
        if src.exists():
            shutil.copy(src, run_dir / fname)

    final_metrics_flat = final_metrics.metrics
    with (run_dir / "validation_metrics.json").open("w") as handle:
        json.dump(final_metrics_flat, handle, indent=2)

    run_manager.register_run(
        run_dir=run_dir,
        metrics=final_metrics_flat,
        config=raw_cfg,
        config_path=config_path,
    )

    _copy_mtl_metrics_to_subdirs(run_manager, paths["metrics"])

    print(f"✅ Saved MTL run: {run_dir.name} (registered and ranked)")
    print(f"📈 Saved MTL history: {run_history_csv}")


def main() -> None:
    args = parse_args()
    run_multitask_training(Path(args.config))


if __name__ == "__main__":
    main()
