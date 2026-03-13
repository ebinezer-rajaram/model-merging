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
    test_setup: Any = None


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

        test_setup = None
        if cfg.training.selection_split != "test":
            try:
                test_setup = prepare_task_for_evaluation(task, processor, split="test", config=task_cfg)
            except Exception as exc:
                print(f"   ⚠️  {task}: no test split available ({exc})")

        tasks[task] = BootstrappedTask(
            name=task,
            config=task_cfg,
            train_setup=train_setup,
            eval_setup=eval_setup,
            generation_kwargs=generation_kwargs,
            test_setup=test_setup,
        )
        try:
            train_size = len(train_setup.dataset)
        except Exception:
            train_size = -1
        try:
            eval_size = len(eval_setup.dataset)
        except Exception:
            eval_size = -1
        try:
            test_size = len(test_setup.dataset) if test_setup is not None else "n/a"
        except Exception:
            test_size = "n/a"
        print(
            f"   ✅ {task}: train_samples={train_size}, "
            f"{cfg.training.selection_split}_samples={eval_size}, "
            f"test_samples={test_size}"
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
        "test_metrics.json",
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
        dest_dir = metrics_dir / label
        if dest_dir.is_symlink() or dest_dir.exists():
            if dest_dir.is_symlink():
                dest_dir.unlink(missing_ok=True)
            elif dest_dir.is_dir():
                shutil.rmtree(dest_dir)
            else:
                dest_dir.unlink(missing_ok=True)
        dest_dir = ensure_dir(dest_dir)
        for fname in mtl_files:
            src = run_path / fname
            if src.exists():
                shutil.copy(src, dest_dir / fname)


def _publish_latest_mtl_metrics_snapshot(run_manager: RunManager, metrics_dir: Path) -> None:
    """Keep top-level metrics files as a latest-run snapshot for backward compatibility."""
    latest_run = run_manager.get_latest_run_path()
    if latest_run is None:
        return

    snapshot_files = [
        "mtl_eval_history.jsonl",
        "mtl_eval_history.csv",
        "mtl_selection_delta_trends.png",
        "mtl_per_task_interference_delta.png",
        "mtl_per_task_primary_metric.png",
    ]
    for fname in snapshot_files:
        src = latest_run / fname
        if src.exists():
            shutil.copy(src, metrics_dir / fname)


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
    test_setups = {task: tasks[task].test_setup for task in task_names if tasks[task].test_setup is not None}
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

    run_manager = RunManager(
        adapter_dir=paths["output_dir"],
        metric_for_ranking=f"eval_mtl_{cfg.training.selection_criterion}",
        greater_is_better=True,
    )
    run_dir = run_manager.create_run_directory()
    print(f"📁 Created MTL run directory: {run_dir.name}")
    metrics_run_dir = ensure_dir(paths["metrics"] / "runs" / run_dir.name)
    print(f"📊 MTL run metrics directory: {metrics_run_dir}")

    evaluator = MultiTaskEvaluator(
        tasks=task_names,
        eval_setups=eval_setups,
        task_generation_kwargs=generation_kwargs,
        split=cfg.training.selection_split,
        batch_size=int(training_args.per_device_eval_batch_size),
        compute_missing_interference_baselines=bool(cfg.training.compute_missing_interference_baselines),
        metrics_dir=metrics_run_dir,
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
    from core.training.training_loop import resolve_checkpoint_path
    checkpoint_path = resolve_checkpoint_path(
        train_config.resume_from_checkpoint,
        paths["output_dir"],
    )
    trainer.train(resume_from_checkpoint=checkpoint_path)

    print("🎯 Running final multi-task evaluation...")
    final_metrics = evaluator.evaluate(
        model=trainer.model,
        processor=processor,
        global_step=int(trainer.state.global_step),
    )
    trainer.log(final_metrics.metrics)

    test_metrics_flat: Dict[str, Any] = {}
    if test_setups:
        print(f"🧪 Running final test-split evaluation on {len(test_setups)}/{len(task_names)} tasks...")
        test_evaluator = MultiTaskEvaluator(
            tasks=list(test_setups.keys()),
            eval_setups=test_setups,
            task_generation_kwargs=generation_kwargs,
            split="test",
            batch_size=int(training_args.per_device_eval_batch_size),
            compute_missing_interference_baselines=bool(cfg.training.compute_missing_interference_baselines),
            metrics_dir=metrics_run_dir,
            selection_criterion=cfg.training.selection_criterion,
            use_cache=False,
            eval_subset=None,
            wandb_project=None,
            auto_plot=False,
        )
        test_result = test_evaluator.evaluate(
            model=trainer.model,
            processor=processor,
            global_step=int(trainer.state.global_step),
        )
        test_metrics_flat = test_result.metrics
        trainer.log(test_metrics_flat)
    else:
        print("⚠️  No test splits available for any task — skipping test evaluation.")

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
        src = metrics_run_dir / fname
        if src.exists():
            shutil.copy(src, run_dir / fname)

    final_metrics_flat = final_metrics.metrics
    with (run_dir / "validation_metrics.json").open("w") as handle:
        json.dump(final_metrics_flat, handle, indent=2)

    if test_metrics_flat:
        with (run_dir / "test_metrics.json").open("w") as handle:
            json.dump(test_metrics_flat, handle, indent=2)

    run_manager.register_run(
        run_dir=run_dir,
        metrics=final_metrics_flat,
        config=raw_cfg,
        config_path=config_path,
    )

    _copy_mtl_metrics_to_subdirs(run_manager, paths["metrics"])
    _publish_latest_mtl_metrics_snapshot(run_manager, paths["metrics"])

    # Write standardised experiment_summary.json for this MTL run.
    try:
        from core.output.summary_writer import (
            write_experiment_summary,
            build_hyperparameters,
            build_selection,
        )
        import math as _math

        _source_tasks = sorted(task_names)
        _selection_criterion = str(cfg.training.selection_criterion).lower()
        _sel_key = f"eval_mtl_{_selection_criterion}"

        # Re-read the JSONL to find the best step (mirrors generate_outputs.py logic).
        _jsonl_path = run_dir / "mtl_eval_history.jsonl"
        _best_step = None
        _best_val = None
        _mtl_geom = None
        _mtl_arith = None
        _mtl_num = None
        if _jsonl_path.exists():
            _rows = []
            with _jsonl_path.open() as _fh:
                for _line in _fh:
                    _line = _line.strip()
                    if _line:
                        try:
                            _rows.append(json.loads(_line))
                        except json.JSONDecodeError:
                            pass
            _best_row = None
            for _row in _rows:
                _rv = _row.get(_sel_key)
                if _rv is None:
                    continue
                try:
                    _rv = float(_rv)
                except (TypeError, ValueError):
                    continue
                if not _math.isfinite(_rv):
                    continue
                if _best_val is None or _rv > _best_val:
                    _best_val = _rv
                    _best_row = _row
            if _best_row is None and _rows:
                _best_row = _rows[-1]
            if _best_row is not None:
                _best_step = _best_row.get("step") or _best_row.get("global_step")
                def _rf(k):
                    v = _best_row.get(k)
                    try:
                        f = float(v)
                        return f if _math.isfinite(f) else None
                    except (TypeError, ValueError):
                        return None
                _mtl_geom = _rf("eval_mtl_geometric_mean_interference_delta")
                _mtl_arith = _rf("eval_mtl_arithmetic_mean_interference_delta")
                _n = _best_row.get("eval_mtl_num_tasks_with_delta")
                try:
                    _mtl_num = int(_n) if _n is not None else None
                except (TypeError, ValueError):
                    _mtl_num = None

        # Build per-task results dict from final_metrics_flat (validation) + test.
        _EVAL_PFX = "eval_"
        _results: Dict[str, Any] = {}
        for _task in _source_tasks:
            _task_entry: Dict[str, Any] = {}
            _task_pfx = f"{_EVAL_PFX}{_task}_"
            # Validation metrics from final_metrics_flat.
            _val_metrics = {k[len(_task_pfx):]: v for k, v in final_metrics_flat.items()
                            if k.startswith(_task_pfx)}
            if _val_metrics:
                _task_entry["validation"] = _val_metrics
            # Test metrics.
            if test_metrics_flat:
                _tst_metrics = {k[len(_task_pfx):]: v for k, v in test_metrics_flat.items()
                                if k.startswith(_task_pfx)}
                if _tst_metrics:
                    _task_entry["test"] = _tst_metrics
            if _task_entry:
                _results[_task] = _task_entry

        _lora_cfg = raw_cfg.get("model", {}).get("lora", {}) or {}
        _train_cfg = raw_cfg.get("training", {}) or {}
        _samp_cfg = _train_cfg.get("sampling", {}) or {}
        _hp = build_hyperparameters(
            learning_rate=_train_cfg.get("learning_rate"),
            lora_r=_lora_cfg.get("r"),
            lora_alpha=_lora_cfg.get("alpha"),
            num_train_epochs=_train_cfg.get("num_train_epochs"),
            per_device_train_batch_size=_train_cfg.get("per_device_train_batch_size"),
            sampling_temperature=_samp_cfg.get("temperature"),
            selection_criterion=_selection_criterion,
            num_tasks=len(_source_tasks),
            max_steps=_train_cfg.get("max_steps"),
        )

        _is_best = (run_manager.get_best_run_path() == run_dir)
        _is_latest = (run_manager.get_latest_run_path() == run_dir)

        _summary_kwargs = dict(
            experiment_type="mtl",
            run_id=run_dir.name,
            timestamp=None,
            config_name=None,
            source_tasks=_source_tasks,
            method="mtl",
            results=_results,
            adapter_path=str(run_dir),
            is_best=_is_best,
            is_latest=_is_latest,
            hyperparameters=_hp,
            merge_info=None,
            mtl_aggregate={
                "best_step": int(_best_step) if _best_step is not None else None,
                "geometric_mean_interference_delta": _mtl_geom,
                "arithmetic_mean_interference_delta": _mtl_arith,
                "num_tasks_with_delta": _mtl_num,
            },
            selection=build_selection(
                policy="best_delta",
                metric_name=_sel_key,
                metric_value=_best_val,
                ranked_by_split="validation",
            ),
            source_files=(
                ["validation_metrics.json", "test_metrics.json"]
                if test_metrics_flat
                else ["validation_metrics.json"]
            ),
        )
        # Run-level summary (per-run, inside adapter subdir).
        write_experiment_summary(output_path=run_dir / "experiment_summary.json", **_summary_kwargs)
        # Combo-root summary (top-level, visible) — only for the best run.
        if _is_best:
            write_experiment_summary(output_path=paths["base"] / "experiment_summary.json", **_summary_kwargs)
    except Exception as _summary_exc:
        print(f"⚠️ Could not write experiment_summary.json: {_summary_exc}")

    print(f"✅ Saved MTL run: {run_dir.name} (registered and ranked)")
    print(f"📈 Saved MTL history: {run_history_csv}")


def main() -> None:
    args = parse_args()
    run_multitask_training(Path(args.config))


if __name__ == "__main__":
    main()
