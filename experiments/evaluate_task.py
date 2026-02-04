"""Evaluate a base model or adapter on a specific task split."""

from __future__ import annotations

import argparse
import sys
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

CURRENT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CURRENT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core import (
    compute_base_cache_path,
    compute_eval_subset_tag,
    print_metrics,
    resolve_merged_eval_dir,
    ensure_dir,
    load_config,
    load_model_and_processor,
    plot_confusion_matrix,
    prepare_task_for_evaluation,
    run_evaluation,
)
from merging.core.utils import resolve_merge_eval_dir
from tasks.asr import (
    TASK_NAME as ASR_TASK_NAME,
    get_artifact_directories as get_asr_artifact_directories,
    get_config_path as get_asr_config_path,
)
from tasks.emotion import (
    TASK_NAME as EMOTION_TASK_NAME,
    get_artifact_directories as get_emotion_artifact_directories,
    get_config_path as get_emotion_config_path,
)
from tasks.intent import (
    TASK_NAME as INTENT_TASK_NAME,
    get_artifact_directories as get_intent_artifact_directories,
    get_config_path as get_intent_config_path,
)
from tasks.speaker_id import (
    TASK_NAME as SPEAKER_TASK_NAME,
    get_artifact_directories as get_speaker_artifact_directories,
    get_config_path as get_speaker_config_path,
)
from tasks.speech_qa import (
    TASK_NAME as SPEECH_QA_TASK_NAME,
    get_artifact_directories as get_speech_qa_artifact_directories,
    get_config_path as get_speech_qa_config_path,
)
from tasks.st import (
    TASK_NAME as ST_TASK_NAME,
    get_artifact_directories as get_st_artifact_directories,
    get_config_path as get_st_config_path,
)
from tasks.kws import (
    TASK_NAME as KWS_TASK_NAME,
    get_artifact_directories as get_kws_artifact_directories,
    get_config_path as get_kws_config_path,
)
from tasks.langid import (
    TASK_NAME as LANGID_TASK_NAME,
    get_artifact_directories as get_langid_artifact_directories,
    get_config_path as get_langid_config_path,
)
from tasks.speaker_ver import (
    TASK_NAME as SPEAKER_VER_TASK_NAME,
    get_artifact_directories as get_speaker_ver_artifact_directories,
    get_config_path as get_speaker_ver_config_path,
)



@dataclass
class EvaluationResult:
    """Structured information about an evaluation run."""

    metrics: Dict[str, Any]
    target_label: str
    cache_used: bool = False
    cache_path: Optional[Path] = None
    save_path: Optional[Path] = None

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate base model or LoRA adapter on a task.")
    parser.add_argument("--task", default=ASR_TASK_NAME, help="Task name to evaluate.")
    parser.add_argument("--config", default=None, help="Optional config filename override.")
    parser.add_argument("--adapter", default=None, help="Path to the adapter directory to evaluate (or task name for cross-task eval).")
    parser.add_argument("--run-id", default=None, help="Specific run ID to evaluate (e.g., run_20251109_143022). Defaults to 'best'.")
    parser.add_argument(
        "--split",
        default="validation",
        choices=("train", "validation", "test"),
        help="Dataset split to run evaluation on.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device evaluation batch size.")
    parser.add_argument("--save-json", default=None, help="Optional path to save metrics as JSON.")
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse cached base-model metrics when available.",
    )
    parser.add_argument("--trained-on-task", default=None, help="Task the adapter was trained on (for cross-task eval).")
    parser.add_argument(
        "--confusion-matrix",
        action="store_true",
        dest="confusion_matrix",
        help="Generate confusion matrix for classification tasks.",
    )
    parser.add_argument(
        "--no-confusion-matrix",
        action="store_false",
        dest="confusion_matrix",
        help="Disable confusion matrix generation (default: False).",
    )
    parser.set_defaults(confusion_matrix=True)
    return parser.parse_args()


def _resolve_adapter_path(
    raw_path: str | None,
    run_id: Optional[str] = None,
    trained_on_task: Optional[str] = None,
) -> tuple[Optional[Path], Optional[str]]:
    """Resolve adapter path relative to the package root.

    Args:
        raw_path: Raw adapter path or task name
        run_id: Specific run ID to use (defaults to "best")
        trained_on_task: Task the adapter was trained on (for determining which artifacts dir)

    Returns:
        Tuple of (adapter_path, trained_on_task)
    """
    if raw_path is None:
        return None, None

    # If it looks like a simple task name (no path separators), treat it as cross-task eval
    if "/" not in raw_path and "\\" not in raw_path:
        # This is likely a task name for cross-task evaluation
        from core.training.run_manager import RunManager

        # Map task name to artifact directory and config path
        task_artifact_map = {
            ASR_TASK_NAME: get_asr_artifact_directories,
            EMOTION_TASK_NAME: get_emotion_artifact_directories,
            SPEAKER_TASK_NAME: get_speaker_artifact_directories,
            INTENT_TASK_NAME: get_intent_artifact_directories,
            SPEECH_QA_TASK_NAME: get_speech_qa_artifact_directories,
            ST_TASK_NAME: get_st_artifact_directories,
            KWS_TASK_NAME: get_kws_artifact_directories,
            LANGID_TASK_NAME: get_langid_artifact_directories,
            SPEAKER_VER_TASK_NAME: get_speaker_ver_artifact_directories,
        }

        task_config_map = {
            ASR_TASK_NAME: get_asr_config_path,
            EMOTION_TASK_NAME: get_emotion_config_path,
            SPEAKER_TASK_NAME: get_speaker_config_path,
            INTENT_TASK_NAME: get_intent_config_path,
            SPEECH_QA_TASK_NAME: get_speech_qa_config_path,
            ST_TASK_NAME: get_st_config_path,
            KWS_TASK_NAME: get_kws_config_path,
            LANGID_TASK_NAME: get_langid_config_path,
            SPEAKER_VER_TASK_NAME: get_speaker_ver_config_path,
        }

        if raw_path in task_artifact_map:
            artifact_dirs = task_artifact_map[raw_path](PACKAGE_ROOT)
            adapters_dir = artifact_dirs["adapters"]

            # Load the task's config to get the adapter_subdir
            task_config_path = task_config_map[raw_path](PACKAGE_ROOT)
            task_config = load_config(task_config_path)
            artifacts_cfg = task_config.get("artifacts", {})
            adapter_subdir = artifacts_cfg.get("adapter_subdir")

            if not adapter_subdir:
                raise ValueError(
                    f"No 'adapter_subdir' specified in config for task '{raw_path}'. "
                    f"Please add 'artifacts.adapter_subdir' to the config file."
                )

            # Use the adapter subdirectory from config
            adapter_base = adapters_dir / adapter_subdir
            if not adapter_base.exists():
                raise ValueError(
                    f"Adapter directory '{adapter_subdir}' not found for task '{raw_path}' at {adapters_dir}. "
                    f"Expected path: {adapter_base}"
                )

            # Now resolve run_id
            run_to_use = run_id or "best"
            if run_to_use in ["best", "latest"]:
                adapter_path = adapter_base / run_to_use
                if not adapter_path.exists():
                    raise ValueError(f"No '{run_to_use}' run found for {raw_path} adapter at {adapter_base}")
            else:
                # Specific run_id
                adapter_path = adapter_base / "runs" / run_to_use
                if not adapter_path.exists():
                    raise ValueError(f"Run '{run_to_use}' not found for {raw_path} adapter")

            return adapter_path.resolve(), raw_path

    # Otherwise, treat as a path
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PACKAGE_ROOT / path

    # If run_id is specified, append it to the path
    if run_id:
        if run_id in ["best", "latest"]:
            path = path / run_id
        else:
            path = path / "runs" / run_id

    return path, trained_on_task


def _get_model_path(config: Dict, task: str) -> Path:
    """Resolve the model checkpoint path from configuration."""
    model_cfg = config.get("model", {})
    model_rel = model_cfg.get("path", "data/models/Qwen2.5-Omni-3B")
    return (PACKAGE_ROOT / model_rel).resolve()


def _get_generation_kwargs(config: Dict) -> Dict:
    """Fetch generation kwargs from config, falling back to defaults."""
    training_cfg = config.get("training", {})
    return dict(training_cfg.get("generation_kwargs", {}))


def _resolve_batch_size(args: argparse.Namespace, config: Dict) -> int:
    """Determine evaluation batch size using CLI override or config defaults."""
    if args.batch_size:
        return args.batch_size
    eval_cfg = config.get("evaluation", {})
    if "per_device_eval_batch_size" in eval_cfg:
        return int(eval_cfg["per_device_eval_batch_size"])
    training_cfg = config.get("training", {})
    return int(training_cfg.get("per_device_eval_batch_size", 4))


def _prepare_dataset_cache(config: Dict, artifact_dirs: Dict[str, Path]) -> Dict:
    """Ensure dataset cache directories resolve relative to the artifact root."""
    dataset_cfg = config.setdefault("dataset", {})
    cache_dir = dataset_cfg.get("cache_dir")
    if cache_dir is not None:
        cache_path = Path(cache_dir)
        if not cache_path.is_absolute():
            cache_path = artifact_dirs["base"] / cache_path
    else:
        cache_path = ensure_dir(artifact_dirs["datasets"])
    dataset_cfg["cache_dir"] = cache_path
    return config


def _json_default(value: Any) -> str:
    """Fallback serializer for JSON encoding of pathlib and similar objects."""
    return str(value)


def _resolve_eval_subset_cache_path(
    *,
    dataset_cfg: Dict[str, Any],
    artifact_dirs: Dict[str, Path],
    task: str,
    split: str,
    eval_tag: str,
) -> Path:
    cache_root = Path(dataset_cfg.get("cache_dir") or artifact_dirs["datasets"])
    cache_dir = cache_root / "eval_subsets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"eval_subset_{task}_{split}_{eval_tag}.json"


def _load_eval_subset_cache(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and "indices" in payload:
            return payload
    except Exception:
        return None
    return None


def _write_eval_subset_cache(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
    tmp.replace(path)


def _save_metrics_to_locations(
    metrics: Dict[str, Any],
    adapter_path: Optional[Path],
    adapter_label: Optional[str],
    metrics_dir: Path,
    task: str,
    split: str,
    trained_on_task: Optional[str] = None,
    merged_tasks: Optional[list[str]] = None,
    merged_method: Optional[str] = None,
    eval_tag: Optional[str] = None,
    show_summary: bool = True,
) -> list[Path]:
    """Save metrics to task-centric evaluation structure.

    For cross-task evaluation (adapter from task X on task Y data):
    - Saves to Y's eval folder as best_X_adapter.json
    - Makes it easy to compare all adapters on the same test set

    Args:
        metrics: Evaluation metrics dictionary
        adapter_path: Path to the adapter being evaluated (None for base model)
        metrics_dir: Base metrics directory for the evaluated task
        task: Task being evaluated on (e.g., "asr")
        split: Dataset split (e.g., "test")
        trained_on_task: Task the adapter was trained on (for cross-task eval)
        show_summary: Whether to print save confirmations

    Returns:
        List of paths where metrics were saved
    """
    saved_paths = []

    def _apply_eval_tag(filename: str) -> str:
        if not eval_tag:
            return filename
        base, ext = os.path.splitext(filename)
        return f"{base}__{eval_tag}{ext or '.json'}"

    # Save merged evaluations into a dedicated subfolder to avoid clutter.
    if merged_tasks:
        method_name = merged_method or "merged"
        merged_eval_dir = resolve_merge_eval_dir(method_name, merged_tasks, split)
        merged_eval_dir.mkdir(parents=True, exist_ok=True)
        label = adapter_label or merged_method or "merged"
        label_for_paths = f"{label}__{eval_tag}" if eval_tag else label
        merged_filename = _apply_eval_tag(f"{task}_{label}_metrics.json")
        merged_metrics_dir = merged_eval_dir / "per_task" / task
        merged_metrics_dir.mkdir(parents=True, exist_ok=True)
        merged_metrics_path = merged_metrics_dir / merged_filename
        with merged_metrics_path.open("w") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True, default=_json_default)

        merged_dir = resolve_merged_eval_dir(
            metrics_dir=metrics_dir,
            split=split,
            task=task,
            merged_tasks=merged_tasks,
            adapter_label=label_for_paths,
            merged_method=merged_method,
        )
        save_path = merged_dir / "metrics.json"
        if save_path.exists() or save_path.is_symlink():
            save_path.unlink()
        relative_target = os.path.relpath(merged_metrics_path, start=merged_dir)
        os.symlink(relative_target, save_path)

        saved_paths.append(merged_metrics_path)
        if show_summary:
            print(f"💾 Saved metrics to: {merged_metrics_path}")
            print(f"🔗 Linked metrics at: {save_path}")
        return saved_paths

    # Save to task-centric eval directory
    eval_dir = ensure_dir(metrics_dir / "eval" / split)

    # Determine filename based on what's being evaluated
    if adapter_path is None:
        if adapter_label:
            filename = _apply_eval_tag(f"{adapter_label}.json")
        else:
            # Base model evaluation
            filename = _apply_eval_tag("base_model.json")
    elif trained_on_task is not None and trained_on_task != task:
        # Cross-task evaluation: adapter from trained_on_task evaluated on task
        filename = _apply_eval_tag(f"best_{trained_on_task}_adapter.json")
    else:
        # Same-task evaluation: task's own adapter
        filename = _apply_eval_tag(f"best_{task}_adapter.json")

    save_path = eval_dir / filename
    with save_path.open("w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True, default=_json_default)
    saved_paths.append(save_path)

    if show_summary:
        print(f"💾 Saved metrics to: {save_path}")

    # Also save to adapter directory for reference (timestamped)
    if adapter_path is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        adapter_metrics_path = adapter_path / f"{split}_metrics_{timestamp}.json"
        with adapter_metrics_path.open("w") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True, default=_json_default)
        saved_paths.append(adapter_metrics_path)
        if show_summary:
            print(f"💾 Also saved to adapter directory: {adapter_metrics_path}")

    return saved_paths


def evaluate(
    task: str = ASR_TASK_NAME,
    config_name: Optional[str] = None,
    adapter: Optional[str] = None,
    run_id: Optional[str] = None,
    trained_on_task: Optional[str] = None,
    split: str = "validation",
    batch_size: Optional[int] = None,
    save_json: Optional[str] = None,
    *,
    enable_cache: bool = False,
    show_summary: bool = True,
    generate_confusion_matrix: bool = False,
    delta_weights: Optional[Dict[str, Any]] = None,
    adapter_label: Optional[str] = None,
    merged_tasks: Optional[list[str]] = None,
    merged_method: Optional[str] = None,
    eval_subset: Optional[Dict[str, Any]] = None,
) -> EvaluationResult:
    """Run evaluation with optional caching for the base model."""
    if task == ASR_TASK_NAME:
        config_path = get_asr_config_path(PACKAGE_ROOT, config_name)
        artifact_dirs = get_asr_artifact_directories(PACKAGE_ROOT)
    elif task == EMOTION_TASK_NAME:
        config_path = get_emotion_config_path(PACKAGE_ROOT, config_name)
        artifact_dirs = get_emotion_artifact_directories(PACKAGE_ROOT)
    elif task == SPEAKER_TASK_NAME:
        config_path = get_speaker_config_path(PACKAGE_ROOT, config_name)
        artifact_dirs = get_speaker_artifact_directories(PACKAGE_ROOT)
    elif task == INTENT_TASK_NAME:
        config_path = get_intent_config_path(PACKAGE_ROOT, config_name)
        artifact_dirs = get_intent_artifact_directories(PACKAGE_ROOT)
    elif task == SPEECH_QA_TASK_NAME:
        config_path = get_speech_qa_config_path(PACKAGE_ROOT, config_name)
        artifact_dirs = get_speech_qa_artifact_directories(PACKAGE_ROOT)
    elif task == ST_TASK_NAME:
        config_path = get_st_config_path(PACKAGE_ROOT, config_name)
        # Load config early for ST to get language parameter
        config = load_config(config_path)
        language = config.get("language", "en_de")
        artifact_dirs = get_st_artifact_directories(PACKAGE_ROOT, language=language)
    elif task == KWS_TASK_NAME:
        config_path = get_kws_config_path(PACKAGE_ROOT, config_name)
        artifact_dirs = get_kws_artifact_directories(PACKAGE_ROOT)
    elif task == LANGID_TASK_NAME:
        config_path = get_langid_config_path(PACKAGE_ROOT, config_name)
        artifact_dirs = get_langid_artifact_directories(PACKAGE_ROOT)
    elif task == SPEAKER_VER_TASK_NAME:
        config_path = get_speaker_ver_config_path(PACKAGE_ROOT, config_name)
        artifact_dirs = get_speaker_ver_artifact_directories(PACKAGE_ROOT)
    else:
        raise NotImplementedError(f"Evaluation for task '{task}' is not implemented yet.")

    # Load config if not already loaded (for ST, it's already loaded above)
    if task != ST_TASK_NAME:
        config = load_config(config_path)
    config = _prepare_dataset_cache(config, artifact_dirs)

    model_path = _get_model_path(config, task)
    adapter_path, detected_trained_on_task = _resolve_adapter_path(adapter, run_id, trained_on_task)
    if delta_weights is not None and adapter_path is not None:
        raise ValueError("delta_weights cannot be used with adapter_path.")

    # Use detected task if not explicitly provided
    if detected_trained_on_task:
        trained_on_task = detected_trained_on_task

    resolved_batch_size = _resolve_batch_size(
        argparse.Namespace(batch_size=batch_size),
        config,
    )
    generation_kwargs = _get_generation_kwargs(config)

    eval_output_dir = ensure_dir(artifact_dirs["metrics"] / "eval")
    dataset_cfg = config.get("dataset", {})

    eval_tag: Optional[str] = None
    max_eval_samples: Optional[int] = None
    subset_shuffle = False
    subset_seed = int(config.get("seed", 0) or 0)
    subset_stratified = False
    subset_label_column: Optional[str] = None
    subset_cache_path: Optional[Path] = None
    if eval_subset and bool(eval_subset.get("enabled", True)):
        eval_tag = compute_eval_subset_tag(eval_subset)
        subset_shuffle = bool(eval_subset.get("shuffle", False))
        if "seed" in eval_subset and eval_subset["seed"] is not None:
            subset_seed = int(eval_subset["seed"])
        subset_stratified = bool(eval_subset.get("stratified", False))
        subset_label_column = eval_subset.get("label_column")

        per_task = eval_subset.get("per_task", {})
        task_override = per_task.get(task) if isinstance(per_task, dict) else None
        if isinstance(task_override, (int, float)):
            max_eval_samples = int(task_override)
        elif isinstance(task_override, dict):
            if "max_samples" in task_override and task_override["max_samples"] is not None:
                max_eval_samples = int(task_override["max_samples"])
            if "shuffle" in task_override and task_override["shuffle"] is not None:
                subset_shuffle = bool(task_override["shuffle"])
            if "seed" in task_override and task_override["seed"] is not None:
                subset_seed = int(task_override["seed"])
            if "stratified" in task_override and task_override["stratified"] is not None:
                subset_stratified = bool(task_override["stratified"])
            if "label_column" in task_override and task_override["label_column"]:
                subset_label_column = task_override["label_column"]

        if max_eval_samples is None and "max_samples" in eval_subset and eval_subset["max_samples"] is not None:
            max_eval_samples = int(eval_subset["max_samples"])
        if eval_tag is not None:
            subset_cache_path = _resolve_eval_subset_cache_path(
                dataset_cfg=dataset_cfg,
                artifact_dirs=artifact_dirs,
                task=task,
                split=split,
                eval_tag=eval_tag,
            )

    cache_path: Optional[Path] = None
    cache_used = False
    metrics: Optional[Dict[str, Any]] = None

    if enable_cache and adapter_path is None:
        dataset_cfg_for_cache = dict(dataset_cfg)
        if eval_tag is not None:
            dataset_cfg_for_cache["_eval_subset"] = {
                "tag": eval_tag,
                "max_samples": max_eval_samples,
                "shuffle": subset_shuffle,
                "seed": subset_seed,
            }
        cache_path = compute_base_cache_path(
            eval_output_dir,
            task=task,
            split=split,
            config_path=config_path,
            batch_size=resolved_batch_size,
            generation_kwargs=generation_kwargs,
            dataset_cfg=dataset_cfg_for_cache,
        )
        if cache_path.exists():
            with cache_path.open("r") as handle:
                metrics = json.load(handle)
            cache_used = True
            if show_summary:
                print(f"♻️ Loaded cached base metrics from {cache_path}")
            else:
                print(f"♻️ Loaded cached base metrics from {cache_path}")

    if metrics is None:
        model, processor = load_model_and_processor(
            model_path,
            adapter_path,
            delta_weights=delta_weights,
        )
        eval_setup = prepare_task_for_evaluation(
            task,
            processor,
            split=split,
            config=config,
        )

        if max_eval_samples is not None:
            max_eval_samples = int(max_eval_samples)
            if max_eval_samples <= 0:
                raise ValueError("eval_subset max_samples must be > 0.")
            try:
                original_size = len(eval_setup.dataset)
            except Exception:
                original_size = None
            if original_size is not None and max_eval_samples < original_size:
                selected_indices: Optional[list[int]] = None
                dataset_fingerprint = getattr(eval_setup.dataset, "_fingerprint", None)
                if subset_cache_path is not None:
                    cached = _load_eval_subset_cache(subset_cache_path)
                    if cached:
                        cached_meta = cached.get("metadata", {})
                        cached_size = cached_meta.get("dataset_size")
                        cached_fp = cached_meta.get("dataset_fingerprint")
                        if cached_size == original_size and (cached_fp is None or cached_fp == dataset_fingerprint):
                            selected_indices = list(cached.get("indices", []))
                if selected_indices is None:
                    if subset_stratified:
                        label_column = subset_label_column or dataset_cfg.get("label_column")
                        if not label_column or label_column not in eval_setup.dataset.column_names:
                            raise ValueError(
                                "eval_subset stratified requires a valid label_column in config or eval_subset."
                            )
                        labels = list(eval_setup.dataset[label_column])
                        by_label: Dict[Any, list[int]] = defaultdict(list)
                        for idx, label in enumerate(labels):
                            by_label[label].append(idx)
                        rng = random.Random(subset_seed)
                        total = len(labels)
                        targets = []
                        for label, indices in by_label.items():
                            frac = (len(indices) / total) * max_eval_samples
                            targets.append((label, frac))
                        base_counts = {label: int(frac) for label, frac in targets}
                        remainder = max_eval_samples - sum(base_counts.values())
                        for label, frac in sorted(targets, key=lambda x: x[1] - int(x[1]), reverse=True):
                            if remainder <= 0:
                                break
                            base_counts[label] += 1
                            remainder -= 1
                        selected = []
                        remaining = []
                        for label, indices in by_label.items():
                            k = min(base_counts.get(label, 0), len(indices))
                            pick = rng.sample(indices, k=k) if k > 0 else []
                            selected.extend(pick)
                            pick_set = set(pick)
                            remaining.extend([i for i in indices if i not in pick_set])
                        if len(selected) < max_eval_samples and remaining:
                            extra = rng.sample(remaining, k=min(max_eval_samples - len(selected), len(remaining)))
                            selected.extend(extra)
                        selected_indices = sorted(selected)
                    elif subset_shuffle:
                        selected_indices = sorted(random.Random(subset_seed).sample(range(original_size), k=max_eval_samples))
                    else:
                        selected_indices = list(range(max_eval_samples))
                    if subset_cache_path is not None:
                        _write_eval_subset_cache(
                            subset_cache_path,
                            {
                                "indices": selected_indices,
                                "metadata": {
                                    "task": task,
                                    "split": split,
                                    "eval_tag": eval_tag,
                                    "dataset_size": original_size,
                                    "dataset_fingerprint": dataset_fingerprint,
                                    "max_samples": max_eval_samples,
                                    "shuffle": subset_shuffle,
                                    "seed": subset_seed,
                                    "stratified": subset_stratified,
                                    "label_column": subset_label_column or dataset_cfg.get("label_column"),
                                },
                            },
                        )
                eval_setup.dataset = eval_setup.dataset.select(selected_indices)

        # Enable prediction storage if confusion matrix is requested
        store_predictions = generate_confusion_matrix and eval_setup.label_names is not None

        metrics = run_evaluation(
            model,
            eval_setup,
            batch_size=resolved_batch_size,
            generation_kwargs=generation_kwargs,
            output_dir=eval_output_dir,
            store_predictions=store_predictions,
            processor=processor,
        )

        # Generate confusion matrix if requested and predictions are available
        if generate_confusion_matrix and eval_setup.label_names:
            predictions = metrics.get("_predictions")
            labels = metrics.get("_labels")
            if predictions is not None and labels is not None:
                # Get confusion matrix settings from config
                metrics_cfg = config.get("metrics", {})
                normalize = metrics_cfg.get("normalize_confusion_matrix", True)

                # Determine save path
                if merged_tasks:
                    merged_dir = resolve_merged_eval_dir(
                        metrics_dir=artifact_dirs["metrics"],
                        split=split,
                        task=task,
                        merged_tasks=merged_tasks,
                        adapter_label=adapter_label,
                        merged_method=merged_method,
                    )
                    cm_path = merged_dir / "confusion_matrix.png"
                else:
                    eval_dir = ensure_dir(artifact_dirs["metrics"] / "eval" / split)
                    if adapter_path is None:
                        if adapter_label:
                            cm_filename = f"confusion_matrix_{adapter_label}.png"
                        else:
                            cm_filename = "confusion_matrix_base_model.png"
                    elif trained_on_task is not None and trained_on_task != task:
                        cm_filename = f"confusion_matrix_{trained_on_task}_adapter.png"
                    else:
                        cm_filename = f"confusion_matrix_{task}_adapter.png"

                    cm_path = eval_dir / cm_filename

                if show_summary:
                    print(f"📊 Generating confusion matrix...")

                plot_confusion_matrix(
                    y_true=labels,
                    y_pred=predictions,
                    label_names=eval_setup.label_names,
                    plot_path=cm_path,
                    title=f"Confusion Matrix - {split.title()} Set",
                    normalize=normalize,
                )

                # Remove prediction data from metrics before saving
                metrics.pop("_predictions", None)
                metrics.pop("_labels", None)

        if eval_tag is not None:
            used_size = None
            try:
                used_size = len(eval_setup.dataset)
            except Exception:
                used_size = None
            metrics["_eval_subset"] = {
                "tag": eval_tag,
                "max_samples": max_eval_samples,
                "shuffle": subset_shuffle,
                "seed": subset_seed,
                "stratified": subset_stratified,
                "label_column": subset_label_column or dataset_cfg.get("label_column"),
                "used_size": used_size,
                "cache_path": str(subset_cache_path) if subset_cache_path is not None else None,
            }

        if cache_path is not None:
            with cache_path.open("w") as handle:
                json.dump(metrics, handle, indent=2, sort_keys=True, default=_json_default)
            if show_summary:
                print(f"💾 Cached base metrics to {cache_path}")
            else:
                print(f"💾 Cached base metrics to {cache_path}")

    save_path: Optional[Path] = None
    saved_paths: list[Path] = []

    # Save metrics to task-centric evaluation structure
    saved_paths = _save_metrics_to_locations(
        metrics=metrics,
        adapter_path=adapter_path,
        adapter_label=adapter_label,
        metrics_dir=artifact_dirs["metrics"],
        task=task,
        split=split,
        trained_on_task=trained_on_task,
        merged_tasks=merged_tasks,
        merged_method=merged_method,
        eval_tag=eval_tag,
        show_summary=show_summary,
    )
    if saved_paths:
        save_path = saved_paths[0]

    # Handle custom save_json path if provided
    if save_json:
        save_path = Path(save_json).expanduser()
        if not save_path.is_absolute():
            save_path = eval_output_dir / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True, default=_json_default)
        saved_paths.append(save_path)
        if show_summary:
            print(f"💾 Saved metrics to custom path: {save_path}")

    if adapter_path is None:
        target_label = adapter_label or "base model"
    else:
        target_label = f"adapter@{adapter_path.name}"
    if show_summary:
        print_metrics(target_label, task, split, metrics)

    return EvaluationResult(
        metrics=metrics,
        target_label=target_label,
        cache_used=cache_used,
        cache_path=cache_path,
        save_path=save_path,
    )


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    evaluate(
        task=args.task,
        config_name=args.config,
        adapter=args.adapter,
        run_id=args.run_id,
        trained_on_task=args.trained_on_task,
        split=args.split,
        batch_size=args.batch_size,
        save_json=args.save_json,
        enable_cache=args.use_cache,
        show_summary=True,
        generate_confusion_matrix=args.confusion_matrix,
    )


if __name__ == "__main__":
    main()
