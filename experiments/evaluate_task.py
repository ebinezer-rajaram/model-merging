"""Evaluate a base model or adapter on a specific task split."""

from __future__ import annotations

import argparse
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any

from core import (
    ensure_dir,
    load_config,
    load_model_and_processor,
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
    parser.add_argument("--adapter", default=None, help="Path to the adapter directory to evaluate.")
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
    return parser.parse_args()


def _resolve_adapter_path(raw_path: str | None) -> Optional[Path]:
    """Resolve adapter path relative to the package root."""
    if raw_path is None:
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PACKAGE_ROOT / path
    return path


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


def evaluate(
    task: str = ASR_TASK_NAME,
    config_name: Optional[str] = None,
    adapter: Optional[str] = None,
    split: str = "validation",
    batch_size: Optional[int] = None,
    save_json: Optional[str] = None,
    *,
    enable_cache: bool = False,
    show_summary: bool = True,
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
    adapter_path = _resolve_adapter_path(adapter)
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

        metrics = run_evaluation(
            model,
            eval_setup,
            batch_size=resolved_batch_size,
            generation_kwargs=generation_kwargs,
            output_dir=eval_output_dir,
        )

        if cache_path is not None:
            with cache_path.open("w") as handle:
                json.dump(metrics, handle, indent=2, sort_keys=True, default=_json_default)
            if show_summary:
                print(f"💾 Cached base metrics to {cache_path}")
            else:
                print(f"💾 Cached base metrics to {cache_path}")

    save_path: Optional[Path] = None
    if save_json:
        save_path = Path(save_json).expanduser()
        if not save_path.is_absolute():
            save_path = eval_output_dir / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True, default=_json_default)
        if show_summary:
            print(f"💾 Saved metrics to {save_path}")

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
        split=args.split,
        batch_size=args.batch_size,
        save_json=args.save_json,
        enable_cache=args.use_cache,
        show_summary=True,
    )


if __name__ == "__main__":
    main()
