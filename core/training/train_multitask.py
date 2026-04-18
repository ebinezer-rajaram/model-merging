"""Joint multi-task LoRA training entrypoint."""

from __future__ import annotations

import argparse
import json
import hashlib
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from peft import PeftModel
from transformers import EarlyStoppingCallback

from core.config.multitask_schema import MultiTaskConfig, parse_multitask_config
from core.data.io_utils import ensure_dir, load_config
from core.evaluation.eval_utils import load_task_dataset_bundle, prepare_task_for_evaluation
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
from merging.evaluation.interference import TASK_METRICS
from merging.runtime.utils import PACKAGE_ROOT, get_task_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train shared LoRA with joint multi-task sampling.")
    parser.add_argument("--config", required=True, help="Path to MTL YAML config.")
    parser.add_argument("--continual-enabled", action="store_true", help="Enable continual stage-2 MTL fine-tuning mode.")
    parser.add_argument("--base-adapter", default=None, help="Base MTL adapter path (run dir, best/latest, or adapter root).")
    parser.add_argument(
        "--base-adapter-run-id",
        default=None,
        help="Optional base adapter alias resolved under --base-adapter when it points to an adapter root (best/latest/run_*).",
    )
    parser.add_argument("--added-tasks", nargs="+", default=None, help="Tasks to add/train in continual mode.")
    parser.add_argument("--base-tasks-override", nargs="+", default=None, help="Explicit base task list override.")
    parser.add_argument(
        "--selection-mode",
        default=None,
        choices=("mtl_interference", "added_task_metric"),
        help="Continual selection mode override.",
    )
    parser.add_argument(
        "--selection-task-set",
        default=None,
        choices=("base_plus_added",),
        help="Continual selection task-set override.",
    )
    parser.add_argument(
        "--final-eval-extra-tasks",
        nargs="+",
        default=None,
        help="Extra tasks to include in final eval/test only (non-continual mode).",
    )
    parser.add_argument(
        "--final-eval-include-speech-qa",
        action="store_true",
        help="Include speech_qa in final evaluation task set.",
    )
    parser.add_argument(
        "--no-final-eval-include-speech-qa",
        action="store_false",
        dest="final_eval_include_speech_qa",
        help="Exclude speech_qa from final evaluation task set.",
    )
    parser.set_defaults(final_eval_include_speech_qa=None)
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


@dataclass
class ContinualTaskPlan:
    enabled: bool
    base_adapter_path: Optional[Path]
    train_tasks: List[str]
    selection_eval_tasks: List[str]
    final_eval_tasks: List[str]
    constituent_tasks: List[str]
    selection_mode: str
    base_tasks: List[str]
    added_tasks: List[str]


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in values:
        value = str(raw).strip().lower()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _resolve_base_adapter_path(base_adapter: str, base_adapter_run_id: Optional[str]) -> Path:
    base_path = Path(base_adapter).expanduser()
    if not base_path.is_absolute():
        base_path = (Path.cwd() / base_path).resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"continual.base_adapter not found: {base_path}")

    if base_adapter_run_id:
        candidate = base_path / str(base_adapter_run_id)
        if candidate.exists():
            return candidate.resolve()
        runs_candidate = base_path / "runs" / str(base_adapter_run_id)
        if runs_candidate.exists():
            return runs_candidate.resolve()
        raise FileNotFoundError(
            "base_adapter_run_id was provided but could not be resolved under "
            f"{base_path}: '{base_adapter_run_id}'"
        )
    direct_config = base_path / "adapter_config.json"
    if direct_config.exists():
        return base_path.resolve()
    default_best = base_path / "best"
    if (default_best / "adapter_config.json").exists():
        return default_best.resolve()
    return base_path.resolve()


def _discover_base_tasks_from_adapter(base_adapter_path: Path) -> List[str]:
    candidates = [
        base_adapter_path / "mtl_config_resolved.json",
        base_adapter_path / "config.yaml",
        base_adapter_path.parent / "mtl_config_resolved.json",
        base_adapter_path.parent / "config.yaml",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = load_config(candidate, base_config_path=None)
        except Exception:
            continue
        tasks = payload.get("tasks", [])
        if not isinstance(tasks, list):
            continue
        names = []
        for entry in tasks:
            if isinstance(entry, dict) and entry.get("name"):
                names.append(str(entry["name"]).strip().lower())
        names = _dedupe_preserve_order(names)
        if names:
            return names
    raise ValueError(
        "Could not discover base tasks from adapter metadata. "
        "Provide continual.base_tasks_override explicitly."
    )


def _resolve_continual_plan(cfg: MultiTaskConfig, raw_cfg: Dict[str, Any]) -> ContinualTaskPlan:
    continual = cfg.continual
    if continual is None or not bool(continual.enabled):
        task_names = [entry.name for entry in cfg.tasks]
        return ContinualTaskPlan(
            enabled=False,
            base_adapter_path=None,
            train_tasks=task_names,
            selection_eval_tasks=task_names,
            final_eval_tasks=task_names,
            constituent_tasks=task_names,
            selection_mode="mtl_interference",
            base_tasks=[],
            added_tasks=[],
        )

    if not continual.base_adapter:
        raise ValueError("continual.enabled=true requires continual.base_adapter.")
    if not continual.added_tasks:
        raise ValueError("continual.enabled=true requires continual.added_tasks.")

    base_adapter_path = _resolve_base_adapter_path(
        str(continual.base_adapter),
        continual.base_adapter_run_id,
    )
    base_tasks = (
        list(continual.base_tasks_override)
        if continual.base_tasks_override is not None
        else _discover_base_tasks_from_adapter(base_adapter_path)
    )
    if "speech_qa" in base_tasks:
        print("ℹ️ Continual mode: removing speech_qa from base task set (held-out OOD test-only task).")
    if "speech_qa" in continual.added_tasks:
        print("ℹ️ Continual mode: removing speech_qa from added_tasks (held-out OOD test-only task).")
    base_tasks = [task for task in base_tasks if task != "speech_qa"]
    added_tasks = [task for task in continual.added_tasks if task != "speech_qa"]
    constituent = _dedupe_preserve_order(base_tasks + added_tasks)
    if not constituent:
        raise ValueError("Unable to derive constituent task set for continual MTL run.")

    final_eval_tasks = list(constituent)
    if bool(continual.final_eval_include_speech_qa) and "speech_qa" not in final_eval_tasks:
        final_eval_tasks.append("speech_qa")

    selection_eval_tasks = list(constituent)
    if continual.selection_mode == "added_task_metric":
        selection_eval_tasks = list(added_tasks)

    raw_cfg.setdefault("continual_runtime", {})
    raw_cfg["continual_runtime"]["base_adapter_path"] = str(base_adapter_path)
    raw_cfg["continual_runtime"]["base_tasks"] = list(base_tasks)
    raw_cfg["continual_runtime"]["added_tasks"] = list(added_tasks)
    raw_cfg["continual_runtime"]["constituent_tasks"] = list(constituent)
    raw_cfg["continual_runtime"]["selection_eval_tasks"] = list(selection_eval_tasks)
    raw_cfg["continual_runtime"]["final_eval_tasks"] = list(final_eval_tasks)

    return ContinualTaskPlan(
        enabled=True,
        base_adapter_path=base_adapter_path,
        train_tasks=list(added_tasks),
        selection_eval_tasks=selection_eval_tasks,
        final_eval_tasks=final_eval_tasks,
        constituent_tasks=constituent,
        selection_mode=continual.selection_mode,
        base_tasks=base_tasks,
        added_tasks=added_tasks,
    )


def _extract_base_adapter_lora_config(base_adapter_path: Path) -> Dict[str, Any]:
    adapter_config_path = base_adapter_path / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"Expected adapter_config.json under base adapter path: {base_adapter_path}")
    with adapter_config_path.open("r") as handle:
        payload = json.load(handle)
    target_modules = payload.get("target_modules")
    if not isinstance(target_modules, list):
        target_modules = []
    r_value = payload.get("r")
    alpha_value = payload.get("lora_alpha")
    if r_value is None or alpha_value is None:
        raise ValueError(
            f"Invalid adapter_config.json under {base_adapter_path}: missing r/lora_alpha."
        )
    mapped = {
        "r": int(r_value),
        "alpha": int(alpha_value),
        "dropout": float(payload.get("lora_dropout", 0.0)),
        "bias": str(payload.get("bias", "none")),
        "target_modules": [str(item) for item in target_modules],
        "task_type": str(payload.get("task_type", "CAUSAL_LM")),
    }
    return mapped


def _override_lora_config_from_base_adapter(raw_cfg: Dict[str, Any], base_adapter_path: Path) -> Dict[str, Any]:
    base_lora = _extract_base_adapter_lora_config(base_adapter_path)
    model_cfg = raw_cfg.setdefault("model", {})
    requested = model_cfg.get("lora") if isinstance(model_cfg, dict) else None
    if isinstance(requested, dict):
        req_compact = {
            "r": requested.get("r"),
            "alpha": requested.get("alpha"),
            "dropout": requested.get("dropout"),
            "bias": requested.get("bias"),
            "target_modules": requested.get("target_modules"),
        }
        base_compact = {
            "r": base_lora.get("r"),
            "alpha": base_lora.get("alpha"),
            "dropout": base_lora.get("dropout"),
            "bias": base_lora.get("bias"),
            "target_modules": base_lora.get("target_modules"),
        }
        if req_compact != base_compact:
            print(
                "ℹ️ Continual mode: overriding requested model.lora with base adapter LoRA config "
                "(rank/alpha/dropout/bias/targets)."
            )
    if isinstance(model_cfg, dict):
        model_cfg["lora"] = dict(base_lora)
    return base_lora


def _extract_float_metric(metrics: Mapping[str, Any], key: str) -> Optional[float]:
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _compute_continual_headline_metrics(
    *,
    validation_metrics: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
    constituent_tasks: List[str],
    added_tasks: List[str],
) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    source = test_metrics if test_metrics else validation_metrics

    deltas = []
    for task in constituent_tasks:
        value = _extract_float_metric(source, f"eval_{task}_interference_delta")
        if value is not None:
            deltas.append(value)
    if deltas:
        report["constituent_interference_delta_mean"] = float(sum(deltas) / len(deltas))
        report["constituent_interference_delta_worst"] = float(min(deltas))

    added_delta = []
    for task in added_tasks:
        value = _extract_float_metric(source, f"eval_{task}_interference_delta")
        if value is not None:
            added_delta.append(value)
    if added_delta:
        report["added_tasks_interference_delta_mean"] = float(sum(added_delta) / len(added_delta))

    gains: Dict[str, Any] = {}
    for task in added_tasks:
        metric_spec = TASK_METRICS.get(task)
        if metric_spec is None:
            continue
        metric_name, higher_is_better = metric_spec
        raw_value = _extract_float_metric(source, f"eval_{task}_{metric_name}")
        if raw_value is None:
            continue
        gains[task] = {
            "metric": metric_name,
            "value": raw_value,
            "oriented_value": (raw_value if higher_is_better else -raw_value),
        }
    if gains:
        report["added_tasks_primary_metrics"] = gains

    speech_qa_acc = _extract_float_metric(source, "eval_speech_qa_accuracy")
    if speech_qa_acc is not None:
        report["speech_qa_transfer_accuracy"] = speech_qa_acc
    return report


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
        # Intentionally skip HF's default eval loop (anchor eval_dataset).
        # We only run the MTL evaluator so checkpoint selection is driven by
        # interference-aware metrics on the configured task subset/full set.
        metrics: Dict[str, float] = {}
        extra = self.multitask_evaluator.evaluate(
            model=self.model,
            processor=self.processing_class,
            global_step=int(self.state.global_step),
        )
        metrics.update(extra.metrics)
        self.log(extra.metrics)
        # Trigger callbacks (including early stopping) with the complete metric dict.
        self.control = self.callback_handler.on_evaluate(
            self.args,
            self.state,
            self.control,
            metrics=metrics,
        )
        return metrics


def _load_bootstrapped_tasks(
    *,
    task_specs: List[Dict[str, Any]],
    selection_split: str,
    eval_split_overrides: Optional[Dict[str, str]],
    processor: Any,
) -> Dict[str, BootstrappedTask]:
    tasks: Dict[str, BootstrappedTask] = {}

    for spec in task_specs:
        task = str(spec["name"]).strip().lower()
        task_config_override = spec.get("config")
        print(f"🔧 Bootstrapping task '{task}' (config={task_config_override or 'default'})...")
        task_module = get_task_module(task)
        config_path = task_module.get_config_path(PACKAGE_ROOT, task_config_override)
        task_cfg = load_config(config_path)

        artifact_dirs = task_module.get_artifact_directories(PACKAGE_ROOT)
        task_cfg = prepare_dataset_cache(task_cfg, artifact_dirs)

        bundle = load_task_dataset_bundle(task, config=task_cfg)
        train_setup = prepare_task_for_evaluation(task, processor, split="train", config=task_cfg, bundle=bundle)
        eval_split = str((eval_split_overrides or {}).get(task, selection_split))
        eval_setup = prepare_task_for_evaluation(
            task,
            processor,
            split=eval_split,
            config=task_cfg,
            bundle=bundle,
        )

        generation_kwargs = dict(task_cfg.get("training", {}).get("generation_kwargs", {}))

        test_setup = None
        if selection_split != "test":
            try:
                test_setup = prepare_task_for_evaluation(task, processor, split="test", config=task_cfg, bundle=bundle)
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
            f"{eval_split}_samples={eval_size}, "
            f"test_samples={test_size}"
        )

    return tasks


def _slugify_task_name(task: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(task).strip().lower()).strip("_")
    return value or "task"


def _build_task_set_slug(
    task_names: list[str],
    *,
    mode: str = "sorted_names",
    base_task_names: Optional[list[str]] = None,
    added_task_names: Optional[list[str]] = None,
) -> str:
    if mode == "sorted_names":
        names = sorted(_slugify_task_name(task) for task in task_names)
        return "_".join(names) if names else "empty_task_set"
    if mode == "base_then_added":
        base_names = [_slugify_task_name(task) for task in (base_task_names or [])]
        added_names = [_slugify_task_name(task) for task in (added_task_names or [])]
        base_part = "_".join(base_names) if base_names else "none"
        added_part = "_".join(added_names) if added_names else "none"
        return f"base_{base_part}__added_{added_part}"
    raise ValueError(
        f"Unsupported task_set_slug_mode '{mode}'. Expected one of ['sorted_names', 'base_then_added']."
    )


def _resolve_mtl_paths(
    *,
    adapter_subdir: str,
    task_names: list[str],
    layout: str = "task_set",
    task_set_slug_mode: str = "sorted_names",
    base_task_names: Optional[list[str]] = None,
    added_task_names: Optional[list[str]] = None,
    artifacts_root: Path | None = None,
) -> Dict[str, Path]:
    base_dir = artifacts_root if artifacts_root is not None else (PACKAGE_ROOT / "artifacts" / "mtl")
    base = ensure_dir(base_dir)
    if layout != "task_set":
        raise ValueError(f"Unsupported MTL artifacts.layout '{layout}'. Expected 'task_set'.")

    task_set_slug = _build_task_set_slug(
        task_names,
        mode=task_set_slug_mode,
        base_task_names=base_task_names,
        added_task_names=added_task_names,
    )
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


def _compute_config_hash_short(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(payload), sort_keys=True, default=str)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def _assert_mode_compatible_task_set_root(
    *,
    task_set_root: Path,
    mode: str,
    allow_mixed_output: bool,
) -> None:
    marker_path = task_set_root / "mtl_mode.json"
    if not marker_path.exists():
        return
    try:
        with marker_path.open("r") as handle:
            marker = json.load(handle)
    except Exception:
        return
    existing_mode = str(marker.get("mode", "")).strip().lower()
    current_mode = str(mode).strip().lower()
    if existing_mode and existing_mode != current_mode and not allow_mixed_output:
        raise ValueError(
            "Output collision detected: task-set root already contains a different MTL mode "
            f"(existing={existing_mode}, current={current_mode}) at {task_set_root}. "
            "Set artifacts.root to a separate namespace or set artifacts.allow_mixed_output=true."
        )


def _write_mode_marker(
    *,
    task_set_root: Path,
    mode: str,
    config_hash: str,
    allow_mixed_output: bool,
) -> None:
    _assert_mode_compatible_task_set_root(
        task_set_root=task_set_root,
        mode=mode,
        allow_mixed_output=allow_mixed_output,
    )
    marker_path = task_set_root / "mtl_mode.json"
    payload = {
        "mode": str(mode),
        "config_hash": str(config_hash),
    }
    with marker_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


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


def _normalize_task_name_list(values: Optional[List[str]]) -> Optional[List[str]]:
    if values is None:
        return None
    out: List[str] = []
    for value in values:
        task = str(value).strip().lower()
        if task:
            out.append(task)
    return out


def _apply_cli_overrides(raw_cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    continual_updates: Dict[str, Any] = {}
    if getattr(args, "final_eval_extra_tasks", None):
        training_cfg = dict(raw_cfg.get("training", {}))
        training_cfg["final_eval_extra_tasks"] = _normalize_task_name_list(list(args.final_eval_extra_tasks))
        raw_cfg["training"] = training_cfg

    if bool(getattr(args, "continual_enabled", False)):
        continual_updates["enabled"] = True
    if getattr(args, "base_adapter", None):
        continual_updates["base_adapter"] = str(args.base_adapter)
    if getattr(args, "base_adapter_run_id", None):
        continual_updates["base_adapter_run_id"] = str(args.base_adapter_run_id)
    if getattr(args, "added_tasks", None):
        continual_updates["added_tasks"] = _normalize_task_name_list(list(args.added_tasks))
    if getattr(args, "base_tasks_override", None):
        continual_updates["base_tasks_override"] = _normalize_task_name_list(list(args.base_tasks_override))
    if getattr(args, "selection_mode", None):
        continual_updates["selection_mode"] = str(args.selection_mode)
    if getattr(args, "selection_task_set", None):
        continual_updates["selection_task_set"] = str(args.selection_task_set)
    if getattr(args, "final_eval_include_speech_qa", None) is not None:
        continual_updates["final_eval_include_speech_qa"] = bool(args.final_eval_include_speech_qa)
        training_cfg = dict(raw_cfg.get("training", {}))
        training_cfg["final_eval_include_speech_qa"] = bool(args.final_eval_include_speech_qa)
        raw_cfg["training"] = training_cfg

    if continual_updates:
        continual_cfg = dict(raw_cfg.get("continual", {}))
        continual_cfg.update(continual_updates)
        raw_cfg["continual"] = continual_cfg


def _load_mtl_config(config_path: Path, args: Optional[argparse.Namespace] = None) -> tuple[MultiTaskConfig, Dict[str, Any]]:
    raw = load_config(config_path, base_config_path=None)
    if args is not None:
        _apply_cli_overrides(raw, args)
    parsed = parse_multitask_config(raw)
    return parsed, raw


def run_multitask_training(config_path: Path, args: Optional[argparse.Namespace] = None) -> None:
    cfg, raw_cfg = _load_mtl_config(config_path, args=args)
    set_global_seed(int(cfg.seed))

    training_cfg_raw = dict(raw_cfg.get("training", {}))
    logging_cfg_raw = dict(raw_cfg.get("logging", {}))
    _setup_wandb(training_cfg_raw, logging_cfg_raw)

    continual_plan = _resolve_continual_plan(cfg, raw_cfg)
    entry_by_name = {entry.name: entry for entry in cfg.tasks}

    # Shared final-eval augmentation for both joint and continual MTL:
    # allow held-out tasks and optional SpeechQA on final eval/test only.
    # This does not change train tasks or model-selection tasks.
    extra_final_eval_tasks = _normalize_task_name_list(
        list(training_cfg_raw.get("final_eval_extra_tasks", []) or [])
    ) or []
    include_speech_qa = bool(training_cfg_raw.get("final_eval_include_speech_qa", False))
    augmented_final_eval_tasks = _dedupe_preserve_order(
        list(continual_plan.final_eval_tasks) + extra_final_eval_tasks + (["speech_qa"] if include_speech_qa else [])
    )
    continual_plan = ContinualTaskPlan(
        enabled=continual_plan.enabled,
        base_adapter_path=continual_plan.base_adapter_path,
        train_tasks=list(continual_plan.train_tasks),
        selection_eval_tasks=list(continual_plan.selection_eval_tasks),
        final_eval_tasks=augmented_final_eval_tasks,
        constituent_tasks=list(continual_plan.constituent_tasks),
        selection_mode=continual_plan.selection_mode,
        base_tasks=list(continual_plan.base_tasks),
        added_tasks=list(continual_plan.added_tasks),
    )
    if extra_final_eval_tasks or include_speech_qa:
        mode_name = "continual" if continual_plan.enabled else "non-continual"
        print(
            f"🧪 {mode_name} final eval task override enabled: "
            f"final_eval_tasks={augmented_final_eval_tasks}"
        )

    if continual_plan.enabled:
        _override_lora_config_from_base_adapter(raw_cfg, Path(continual_plan.base_adapter_path))
        cfg = parse_multitask_config(raw_cfg)
        entry_by_name = {entry.name: entry for entry in cfg.tasks}
        print(
            "🔁 Continual MTL mode enabled: "
            f"base_adapter={continual_plan.base_adapter_path}, "
            f"added_tasks={continual_plan.added_tasks}, "
            f"base_tasks={continual_plan.base_tasks}, "
            f"selection_mode={continual_plan.selection_mode}"
        )

    model_path = PACKAGE_ROOT / cfg.model.path
    if continual_plan.enabled:
        model, processor = load_qwen_model(model_path, apply_lora=False)
        model = PeftModel.from_pretrained(
            model,
            str(continual_plan.base_adapter_path),
            is_trainable=True,
        )
    else:
        lora_config = None
        if cfg.model.lora is not None:
            lora_config = create_lora_config_from_dict(cfg.model.lora.model_dump())
        model, processor = load_qwen_model(model_path, lora_config=lora_config)

    bootstrap_tasks = _dedupe_preserve_order(
        continual_plan.train_tasks
        + continual_plan.selection_eval_tasks
        + continual_plan.final_eval_tasks
    )
    task_specs = []
    for task in bootstrap_tasks:
        entry = entry_by_name.get(task)
        task_specs.append({"name": task, "config": (entry.config if entry is not None else None)})

    eval_split_overrides: Dict[str, str] = {}
    if "speech_qa" in bootstrap_tasks and cfg.training.selection_split != "test":
        # SpeechQA is treated as held-out OOD in continual experiments.
        # Force bootstrap eval split to test so we never require validation for it.
        eval_split_overrides["speech_qa"] = "test"

    tasks = _load_bootstrapped_tasks(
        task_specs=task_specs,
        selection_split=cfg.training.selection_split,
        eval_split_overrides=eval_split_overrides,
        processor=processor,
    )

    train_task_names = list(continual_plan.train_tasks)
    if not train_task_names:
        raise ValueError("No training tasks resolved for MTL run.")
    selection_eval_task_names = list(continual_plan.selection_eval_tasks)
    final_eval_task_names = list(continual_plan.final_eval_tasks)
    final_validation_task_names = [
        task for task in final_eval_task_names if not (task == "speech_qa" and cfg.training.selection_split != "test")
    ]
    source_tasks_for_summary = (
        list(continual_plan.constituent_tasks) if continual_plan.enabled else [entry.name for entry in cfg.tasks]
    )

    train_datasets = {task: tasks[task].train_setup.dataset for task in train_task_names}
    selection_eval_setups = {task: tasks[task].eval_setup for task in selection_eval_task_names}
    final_eval_setups = {task: tasks[task].eval_setup for task in final_validation_task_names}
    test_setups = {task: tasks[task].test_setup for task in final_eval_task_names if tasks[task].test_setup is not None}
    collators = {task: tasks[task].train_setup.data_collator for task in train_task_names}
    generation_kwargs = {task: tasks[task].generation_kwargs for task in tasks.keys()}
    train_weights = {task: float(getattr(entry_by_name.get(task), "train_weight", 1.0) or 1.0) for task in train_task_names}

    multitask_train_dataset = MultiTaskDataset(train_datasets)
    artifacts_cfg_raw = raw_cfg.get("artifacts", {}) or {}
    artifacts_root_cfg = str(artifacts_cfg_raw.get("root", "artifacts/mtl"))
    artifacts_root = Path(artifacts_root_cfg)
    if not artifacts_root.is_absolute():
        artifacts_root = (PACKAGE_ROOT / artifacts_root).resolve()
    else:
        artifacts_root = artifacts_root.resolve()

    allow_mixed_output = bool(artifacts_cfg_raw.get("allow_mixed_output", False))
    run_mode = "continual_mtl_ft" if continual_plan.enabled else "joint_mtl"
    canonical_joint_root = (PACKAGE_ROOT / "artifacts" / "mtl").resolve()
    if continual_plan.enabled and artifacts_root == canonical_joint_root and not allow_mixed_output:
        raise ValueError(
            "Continual MTL run is configured to write into canonical joint MTL root "
            f"({artifacts_root}). Set artifacts.root to a dedicated namespace (e.g. "
            "'artifacts/mtl/continual') or set artifacts.allow_mixed_output=true."
        )

    paths = _resolve_mtl_paths(
        adapter_subdir=cfg.artifacts.adapter_subdir,
        task_names=source_tasks_for_summary,
        layout=cfg.artifacts.layout,
        task_set_slug_mode=cfg.artifacts.task_set_slug_mode,
        base_task_names=(list(continual_plan.base_tasks) if continual_plan.enabled else None),
        added_task_names=(list(continual_plan.added_tasks) if continual_plan.enabled else None),
        artifacts_root=artifacts_root,
    )
    print(f"📁 MTL artifact root: {paths['base']}")
    _write_mode_marker(
        task_set_root=paths["base"],
        mode=run_mode,
        config_hash=_compute_config_hash_short(raw_cfg),
        allow_mixed_output=allow_mixed_output,
    )

    selection_metric_for_best_model = "mtl_geometric_mean_interference_delta"
    if continual_plan.enabled and continual_plan.selection_mode == "added_task_metric":
        selection_metric_for_best_model = "added_tasks_primary_oriented_mean"

    task_defaults = {
        "metric_for_best_model": selection_metric_for_best_model,
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
        f"{task}={sampler.task_probabilities.get(task, 0.0):.3f}" for task in train_task_names
    )
    print(f"🧪 Task sampling probabilities (temperature={cfg.training.sampling.temperature}): {sampling_summary}")

    ranking_metric = str(train_config.metric_for_best_model)
    if not ranking_metric.startswith("eval_"):
        ranking_metric = f"eval_{ranking_metric}"
    run_manager = RunManager(
        adapter_dir=paths["output_dir"],
        metric_for_ranking=ranking_metric,
        greater_is_better=True,
    )
    run_dir = run_manager.create_run_directory()
    print(f"📁 Created MTL run directory: {run_dir.name}")
    metrics_run_dir = ensure_dir(paths["metrics"] / "runs" / run_dir.name)
    print(f"📊 MTL run metrics directory: {metrics_run_dir}")

    evaluator = MultiTaskEvaluator(
        tasks=selection_eval_task_names,
        eval_setups=selection_eval_setups,
        task_generation_kwargs=generation_kwargs,
        split=cfg.training.selection_split,
        batch_size=int(training_args.per_device_eval_batch_size),
        compute_missing_interference_baselines=bool(cfg.training.compute_missing_interference_baselines),
        metrics_dir=metrics_run_dir,
        selection_criterion=cfg.training.selection_criterion,
        selection_mode=continual_plan.selection_mode,
        selected_primary_tasks=(
            list(continual_plan.added_tasks) if continual_plan.enabled else []
        ),
        use_cache=True,
        eval_subset=(dict(cfg.training.selection_eval_subset) if cfg.training.selection_eval_subset else None),
        wandb_project=(cfg.logging.wandb_project if cfg.logging is not None else None),
        auto_plot=(bool(cfg.metrics.auto_plot) if cfg.metrics is not None else True),
        dataloader_num_workers=int(training_args.dataloader_num_workers or 0),
        dataloader_prefetch_factor=(
            int(training_args.dataloader_prefetch_factor)
            if training_args.dataloader_prefetch_factor is not None
            else None
        ),
    )

    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=multitask_train_dataset,
        # HF Trainer requires a non-null eval_dataset when eval_strategy != "no".
        # MultiTaskTrainer.evaluate() bypasses HF eval dataloader and runs the
        # multitask evaluator directly, so this acts only as an init-time anchor.
        eval_dataset=multitask_train_dataset,
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
        f"train_tasks={train_task_names}, temperature={cfg.training.sampling.temperature}, "
        f"selection={continual_plan.selection_mode}/{cfg.training.selection_criterion}/{cfg.training.selection_split}"
    )
    from core.training.training_loop import resolve_checkpoint_path
    checkpoint_path = resolve_checkpoint_path(
        train_config.resume_from_checkpoint,
        paths["output_dir"],
    )
    trainer.train(resume_from_checkpoint=checkpoint_path)

    print("🎯 Running final model-selection evaluation...")
    selection_metrics = evaluator.evaluate(
        model=trainer.model,
        processor=processor,
        global_step=int(trainer.state.global_step),
    )
    trainer.log(selection_metrics.metrics)

    print("🎯 Running final validation evaluation on reporting tasks...")
    if set(final_validation_task_names) == set(selection_eval_task_names):
        final_metrics = selection_metrics
    else:
        final_evaluator = MultiTaskEvaluator(
            tasks=final_validation_task_names,
            eval_setups=final_eval_setups,
            task_generation_kwargs=generation_kwargs,
            split=cfg.training.selection_split,
            batch_size=int(training_args.per_device_eval_batch_size),
            compute_missing_interference_baselines=bool(cfg.training.compute_missing_interference_baselines),
            metrics_dir=metrics_run_dir / "final_validation",
            selection_criterion=cfg.training.selection_criterion,
            selection_mode=continual_plan.selection_mode,
            selected_primary_tasks=(
                list(continual_plan.added_tasks) if continual_plan.enabled else []
            ),
            use_cache=False,
            eval_subset=None,
            wandb_project=None,
            auto_plot=False,
            dataloader_num_workers=int(training_args.dataloader_num_workers or 0),
            dataloader_prefetch_factor=(
                int(training_args.dataloader_prefetch_factor)
                if training_args.dataloader_prefetch_factor is not None
                else None
            ),
        )
        final_metrics = final_evaluator.evaluate(
            model=trainer.model,
            processor=processor,
            global_step=int(trainer.state.global_step),
        )
        trainer.log(final_metrics.metrics)

    test_metrics_flat: Dict[str, Any] = {}
    if test_setups:
        print(f"🧪 Running final test-split evaluation on {len(test_setups)}/{len(final_eval_task_names)} tasks...")
        test_evaluator = MultiTaskEvaluator(
            tasks=list(test_setups.keys()),
            eval_setups=test_setups,
            task_generation_kwargs=generation_kwargs,
            split="test",
            batch_size=int(training_args.per_device_eval_batch_size),
            compute_missing_interference_baselines=bool(cfg.training.compute_missing_interference_baselines),
            metrics_dir=metrics_run_dir,
            selection_criterion=cfg.training.selection_criterion,
            selection_mode=continual_plan.selection_mode,
            selected_primary_tasks=(
                list(continual_plan.added_tasks) if continual_plan.enabled else []
            ),
            use_cache=False,
            eval_subset=None,
            wandb_project=None,
            auto_plot=False,
            dataloader_num_workers=int(training_args.dataloader_num_workers or 0),
            dataloader_prefetch_factor=(
                int(training_args.dataloader_prefetch_factor)
                if training_args.dataloader_prefetch_factor is not None
                else None
            ),
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

    run_metadata = {
        "mode": run_mode,
        "base_adapter_path": (
            str(continual_plan.base_adapter_path) if continual_plan.base_adapter_path is not None else None
        ),
        "base_tasks": list(continual_plan.base_tasks),
        "added_tasks": list(continual_plan.added_tasks),
        "constituent_tasks": list(continual_plan.constituent_tasks),
        "selection_mode": str(continual_plan.selection_mode),
        "selection_eval_tasks": list(continual_plan.selection_eval_tasks),
        "final_eval_tasks": list(continual_plan.final_eval_tasks),
        "selection_split": str(cfg.training.selection_split),
    }
    with (run_dir / "mtl_run_metadata.json").open("w") as handle:
        json.dump(run_metadata, handle, indent=2, sort_keys=True)

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

    final_metrics_flat = dict(final_metrics.metrics)
    for key, value in selection_metrics.metrics.items():
        if key not in final_metrics_flat:
            final_metrics_flat[key] = value
    with (run_dir / "validation_metrics.json").open("w") as handle:
        json.dump(final_metrics_flat, handle, indent=2)

    if test_metrics_flat:
        with (run_dir / "test_metrics.json").open("w") as handle:
            json.dump(test_metrics_flat, handle, indent=2)

    if continual_plan.enabled:
        headline = _compute_continual_headline_metrics(
            validation_metrics=final_metrics_flat,
            test_metrics=test_metrics_flat,
            constituent_tasks=list(continual_plan.constituent_tasks),
            added_tasks=list(continual_plan.added_tasks),
        )
        with (run_dir / "continual_headline_metrics.json").open("w") as handle:
            json.dump(headline, handle, indent=2, sort_keys=True)
        print(f"📌 Continual headline metrics saved: {run_dir / 'continual_headline_metrics.json'}")

    run_manager.register_run(
        run_dir=run_dir,
        metrics=selection_metrics.metrics,
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

        _source_tasks = sorted(source_tasks_for_summary)
        _selection_criterion = str(cfg.training.selection_criterion).lower()
        _sel_key = str(run_manager.metric_for_ranking)

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
            selection_criterion=(
                str(continual_plan.selection_mode)
                if continual_plan.enabled
                else _selection_criterion
            ),
            num_tasks=len(_source_tasks),
            max_steps=_train_cfg.get("max_steps"),
        )
        _hp["mtl_mode"] = run_mode

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
                (
                    ["validation_metrics.json", "test_metrics.json", "continual_headline_metrics.json", "mtl_run_metadata.json"]
                    if (test_metrics_flat and continual_plan.enabled)
                    else (
                        ["validation_metrics.json", "test_metrics.json", "mtl_run_metadata.json"]
                        if test_metrics_flat
                        else (
                            ["validation_metrics.json", "continual_headline_metrics.json", "mtl_run_metadata.json"]
                            if continual_plan.enabled
                            else ["validation_metrics.json", "mtl_run_metadata.json"]
                        )
                    )
                )
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
    run_multitask_training(Path(args.config), args=args)


if __name__ == "__main__":
    main()
