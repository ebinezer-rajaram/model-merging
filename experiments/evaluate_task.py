"""Evaluate a base model or adapter on a specific task split."""

from __future__ import annotations

import argparse
import json
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

from core import (
    ensure_dir,
    load_config,
    load_model_and_processor,
    plot_confusion_matrix,
    prepare_task_for_evaluation,
    run_evaluation,
)
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


PACKAGE_ROOT = Path(__file__).resolve().parent.parent


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

        # Map task name to artifact directory
        task_artifact_map = {
            ASR_TASK_NAME: get_asr_artifact_directories,
            EMOTION_TASK_NAME: get_emotion_artifact_directories,
            SPEAKER_TASK_NAME: get_speaker_artifact_directories,
            INTENT_TASK_NAME: get_intent_artifact_directories,
            SPEECH_QA_TASK_NAME: get_speech_qa_artifact_directories,
        }

        if raw_path in task_artifact_map:
            artifact_dirs = task_artifact_map[raw_path](PACKAGE_ROOT)
            # Find the adapter directory (should be only one in adapters/)
            adapters_dir = artifact_dirs["adapters"]

            # Look for adapter subdirectories
            adapter_subdirs = [d for d in adapters_dir.iterdir() if d.is_dir()]
            if not adapter_subdirs:
                raise ValueError(f"No adapter found for task '{raw_path}'")

            # Use the first adapter directory found (or could make this configurable)
            adapter_base = adapter_subdirs[0]

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
    model_rel = model_cfg.get("path", "models/Qwen2.5-Omni-3B")
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


def _compute_base_cache_path(
    eval_dir: Path,
    *,
    task: str,
    split: str,
    config_path: Path,
    batch_size: int,
    generation_kwargs: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
) -> Path:
    """Generate a deterministic cache file path for base-model metrics."""
    fingerprint_payload = {
        "task": task,
        "split": split,
        "config_path": str(config_path.resolve()),
        "batch_size": int(batch_size),
        "generation_kwargs": generation_kwargs,
        "dataset_cfg": dataset_cfg,
    }
    fingerprint_raw = json.dumps(fingerprint_payload, sort_keys=True, default=_json_default).encode("utf-8")
    fingerprint = hashlib.md5(fingerprint_raw).hexdigest()[:12]
    filename = f"{task}_{split}_base_{fingerprint}.json"
    return eval_dir / filename


def _print_metrics(label: str, task: str, split: str, metrics: Dict[str, Any]) -> None:
    """Pretty-print evaluation metrics."""
    print(f"✅ Evaluation complete for {label} on {task}/{split}")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def _save_metrics_to_locations(
    metrics: Dict[str, Any],
    adapter_path: Optional[Path],
    metrics_dir: Path,
    task: str,
    split: str,
    trained_on_task: Optional[str] = None,
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

    # Save to task-centric eval directory
    eval_dir = ensure_dir(metrics_dir / "eval" / split)

    # Determine filename based on what's being evaluated
    if adapter_path is None:
        # Base model evaluation
        filename = "base_model.json"
    elif trained_on_task is not None and trained_on_task != task:
        # Cross-task evaluation: adapter from trained_on_task evaluated on task
        filename = f"best_{trained_on_task}_adapter.json"
    else:
        # Same-task evaluation: task's own adapter
        filename = f"best_{task}_adapter.json"

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
    else:
        raise NotImplementedError(f"Evaluation for task '{task}' is not implemented yet.")

    config = load_config(config_path)
    config = _prepare_dataset_cache(config, artifact_dirs)

    model_path = _get_model_path(config, task)
    adapter_path, detected_trained_on_task = _resolve_adapter_path(adapter, run_id, trained_on_task)

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

    cache_path: Optional[Path] = None
    cache_used = False
    metrics: Optional[Dict[str, Any]] = None

    if enable_cache and adapter_path is None:
        cache_path = _compute_base_cache_path(
            eval_output_dir,
            task=task,
            split=split,
            config_path=config_path,
            batch_size=resolved_batch_size,
            generation_kwargs=generation_kwargs,
            dataset_cfg=dataset_cfg,
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
        model, processor = load_model_and_processor(model_path, adapter_path)
        eval_setup = prepare_task_for_evaluation(
            task,
            processor,
            split=split,
            config=config,
        )

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
                eval_dir = ensure_dir(artifact_dirs["metrics"] / "eval" / split)
                if adapter_path is None:
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
        metrics_dir=artifact_dirs["metrics"],
        task=task,
        split=split,
        trained_on_task=trained_on_task,
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

    target_label = "base model" if adapter_path is None else f"adapter@{adapter_path.name}"
    if show_summary:
        _print_metrics(target_label, task, split, metrics)

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
