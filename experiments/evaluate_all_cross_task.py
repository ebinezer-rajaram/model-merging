"""Evaluate all adapters on all tasks in a cross-task evaluation matrix.

This script evaluates:
1. Base model on all tasks
2. Each task's adapter on all tasks (including cross-task evaluation)

Results are saved in a structured format for easy comparison.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core import ensure_dir, load_config
from experiments.evaluate_task import evaluate, EvaluationResult
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

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# Map of all tasks with their config getters
TASK_MAP = {
    ASR_TASK_NAME: (get_asr_config_path, get_asr_artifact_directories),
    EMOTION_TASK_NAME: (get_emotion_config_path, get_emotion_artifact_directories),
    INTENT_TASK_NAME: (get_intent_config_path, get_intent_artifact_directories),
    SPEAKER_TASK_NAME: (get_speaker_config_path, get_speaker_artifact_directories),
    SPEECH_QA_TASK_NAME: (get_speech_qa_config_path, get_speech_qa_artifact_directories),
    ST_TASK_NAME: (get_st_config_path, get_st_artifact_directories),
    KWS_TASK_NAME: (get_kws_config_path, get_kws_artifact_directories),
    LANGID_TASK_NAME: (get_langid_config_path, get_langid_artifact_directories),
    SPEAKER_VER_TASK_NAME: (get_speaker_ver_config_path, get_speaker_ver_artifact_directories),
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate all adapters on all tasks (cross-task evaluation matrix)."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="List of tasks to evaluate on. If not specified, evaluates on all tasks.",
    )
    parser.add_argument(
        "--adapters",
        nargs="+",
        default=None,
        help="List of adapter tasks to evaluate. If not specified, evaluates all available adapters.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "validation", "test"),
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Per-device evaluation batch size (uses config default if not specified).",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse cached base-model metrics when available.",
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip base model evaluation (only evaluate adapters).",
    )
    parser.add_argument(
        "--confusion-matrix",
        action="store_true",
        dest="confusion_matrix",
        help="Generate confusion matrices for classification tasks.",
    )
    parser.add_argument(
        "--no-confusion-matrix",
        action="store_false",
        dest="confusion_matrix",
        help="Disable confusion matrix generation.",
    )
    parser.set_defaults(confusion_matrix=True)
    parser.add_argument(
        "--output-summary",
        default=None,
        help="Path to save summary JSON with all results.",
    )
    return parser.parse_args()


def get_available_adapters(tasks: List[str]) -> Dict[str, Optional[Path]]:
    """Find available trained adapters for each task.

    Returns:
        Dict mapping task name to adapter path (None if not found)
    """
    adapters = {}

    for task in tasks:
        if task not in TASK_MAP:
            print(f"⚠️  Unknown task '{task}', skipping adapter check")
            adapters[task] = None
            continue

        get_config, get_artifacts = TASK_MAP[task]
        config_path = get_config(PACKAGE_ROOT)

        # Special handling for ST task which needs language parameter
        if task == ST_TASK_NAME:
            config = load_config(config_path)
            language = config.get("language", "en_de")
            artifact_dirs = get_artifacts(PACKAGE_ROOT, language=language)
        else:
            artifact_dirs = get_artifacts(PACKAGE_ROOT)
            config = load_config(config_path)

        # Get adapter subdirectory from config
        artifacts_cfg = config.get("artifacts", {})
        adapter_subdir = artifacts_cfg.get("adapter_subdir")

        if not adapter_subdir:
            print(f"⚠️  No adapter_subdir configured for task '{task}'")
            adapters[task] = None
            continue

        # Check if best adapter exists
        adapter_base = artifact_dirs["adapters"] / adapter_subdir
        best_adapter = adapter_base / "best"

        if best_adapter.exists():
            adapters[task] = best_adapter
            print(f"✓ Found adapter for {task}: {best_adapter}")
        else:
            print(f"⚠️  No 'best' adapter found for {task} at {adapter_base}")
            adapters[task] = None

    return adapters


def run_cross_evaluation_matrix(
    eval_tasks: List[str],
    adapter_tasks: List[str],
    split: str = "test",
    batch_size: Optional[int] = None,
    use_cache: bool = False,
    skip_base: bool = False,
    generate_confusion_matrix: bool = True,
) -> Dict[str, Dict[str, Dict]]:
    """Run full cross-task evaluation matrix.

    Args:
        eval_tasks: List of tasks to evaluate on (columns)
        adapter_tasks: List of adapter tasks to evaluate (rows)
        split: Dataset split to use
        batch_size: Batch size for evaluation
        use_cache: Whether to use cached base model metrics
        skip_base: Whether to skip base model evaluation
        generate_confusion_matrix: Whether to generate confusion matrices

    Returns:
        Nested dict: results[eval_task][adapter_or_base] = metrics
    """
    results: Dict[str, Dict[str, Dict]] = {task: {} for task in eval_tasks}

    # Get available adapters
    available_adapters = get_available_adapters(adapter_tasks)

    # Filter to only adapters that exist
    valid_adapter_tasks = [task for task, path in available_adapters.items() if path is not None]

    if not valid_adapter_tasks:
        print("⚠️  No valid adapters found! Only base model will be evaluated.")

    print("\n" + "=" * 80)
    print("CROSS-TASK EVALUATION MATRIX")
    print("=" * 80)
    print(f"Evaluation tasks: {', '.join(eval_tasks)}")
    print(f"Adapter tasks: {', '.join(valid_adapter_tasks) if valid_adapter_tasks else 'None'}")
    print(f"Split: {split}")
    print(f"Generate confusion matrices: {generate_confusion_matrix}")
    print("=" * 80 + "\n")

    total_evals = len(eval_tasks) * (1 + len(valid_adapter_tasks)) if not skip_base else len(eval_tasks) * len(valid_adapter_tasks)
    current_eval = 0

    # Evaluate each task
    for eval_task in eval_tasks:
        print(f"\n{'=' * 80}")
        print(f"EVALUATING ON TASK: {eval_task.upper()}")
        print(f"{'=' * 80}\n")

        # 1. Evaluate base model
        if not skip_base:
            current_eval += 1
            print(f"\n[{current_eval}/{total_evals}] Base model on {eval_task}/{split}")
            print("-" * 80)

            try:
                result = evaluate(
                    task=eval_task,
                    adapter=None,
                    split=split,
                    batch_size=batch_size,
                    enable_cache=use_cache,
                    show_summary=True,
                    generate_confusion_matrix=generate_confusion_matrix,
                )
                results[eval_task]["base"] = {
                    "metrics": result.metrics,
                    "cache_used": result.cache_used,
                    "save_path": str(result.save_path) if result.save_path else None,
                }
                print(f"✅ Completed base model evaluation on {eval_task}")
            except Exception as e:
                print(f"❌ Error evaluating base model on {eval_task}: {e}")
                results[eval_task]["base"] = {"error": str(e)}

        # 2. Evaluate each adapter
        for adapter_task in valid_adapter_tasks:
            current_eval += 1
            adapter_path = available_adapters[adapter_task]

            eval_type = "same-task" if adapter_task == eval_task else "cross-task"
            print(f"\n[{current_eval}/{total_evals}] {adapter_task} adapter on {eval_task}/{split} ({eval_type})")
            print("-" * 80)

            try:
                result = evaluate(
                    task=eval_task,
                    adapter=adapter_task,  # Pass task name for cross-task eval
                    trained_on_task=adapter_task,
                    split=split,
                    batch_size=batch_size,
                    enable_cache=False,  # Don't cache adapter results
                    show_summary=True,
                    generate_confusion_matrix=generate_confusion_matrix,
                )
                results[eval_task][adapter_task] = {
                    "metrics": result.metrics,
                    "adapter_path": str(adapter_path),
                    "save_path": str(result.save_path) if result.save_path else None,
                }
                print(f"✅ Completed {adapter_task} adapter evaluation on {eval_task}")
            except Exception as e:
                print(f"❌ Error evaluating {adapter_task} adapter on {eval_task}: {e}")
                results[eval_task][adapter_task] = {"error": str(e)}

    return results


def print_summary_table(results: Dict[str, Dict[str, Dict]], primary_metric: str = "accuracy") -> None:
    """Print a formatted summary table of results.

    Args:
        results: Results dictionary from run_cross_evaluation_matrix
        primary_metric: Metric to display in table (accuracy, wer, f1, etc.)
    """
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80 + "\n")

    # Get all eval tasks and adapter types
    eval_tasks = sorted(results.keys())
    adapter_types = set()
    for task_results in results.values():
        adapter_types.update(task_results.keys())
    adapter_types = ["base"] + sorted([a for a in adapter_types if a != "base"])

    # Determine metric to display for each task
    task_metrics = {}
    for task in eval_tasks:
        task_results = results[task]
        # Look at base model results to determine primary metric
        if "base" in task_results and "metrics" in task_results["base"]:
            metrics = task_results["base"]["metrics"]
            # Prioritize common metrics
            for metric in ["accuracy", "wer", "f1", "bleu"]:
                if metric in metrics:
                    task_metrics[task] = metric
                    break
            if task not in task_metrics:
                # Use first available metric
                available = [k for k in metrics.keys() if not k.startswith("_")]
                if available:
                    task_metrics[task] = available[0]
                else:
                    task_metrics[task] = "N/A"
        else:
            task_metrics[task] = "N/A"

    # Print header
    header = "Model/Adapter".ljust(20)
    for task in eval_tasks:
        metric_name = task_metrics[task]
        header += f"{task} ({metric_name})".rjust(20)
    print(header)
    print("-" * (20 + 20 * len(eval_tasks)))

    # Print each row
    for adapter in adapter_types:
        row = adapter.ljust(20)
        for task in eval_tasks:
            if adapter in results[task]:
                result = results[task][adapter]
                if "error" in result:
                    row += "ERROR".rjust(20)
                elif "metrics" in result:
                    metric_name = task_metrics[task]
                    value = result["metrics"].get(metric_name)
                    if value is not None:
                        # Format based on metric type
                        if metric_name == "wer":
                            # Lower is better for WER
                            row += f"{value:.2%}".rjust(20)
                        elif isinstance(value, float):
                            if value <= 1.0:
                                # Percentage metric
                                row += f"{value:.2%}".rjust(20)
                            else:
                                # Raw score (e.g., BLEU)
                                row += f"{value:.4f}".rjust(20)
                        else:
                            row += str(value).rjust(20)
                    else:
                        row += "N/A".rjust(20)
                else:
                    row += "N/A".rjust(20)
            else:
                row += "N/A".rjust(20)
        print(row)

    print("\n")


def save_summary(results: Dict[str, Dict[str, Dict]], output_path: Path) -> None:
    """Save evaluation results summary to JSON file.

    Args:
        results: Results dictionary from run_cross_evaluation_matrix
        output_path: Path to save summary JSON
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"💾 Saved evaluation summary to: {output_path}")


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    # Determine which tasks to evaluate
    eval_tasks = args.tasks if args.tasks else list(TASK_MAP.keys())
    adapter_tasks = args.adapters if args.adapters else list(TASK_MAP.keys())

    # Validate task names
    for task in eval_tasks + adapter_tasks:
        if task not in TASK_MAP:
            print(f"❌ Unknown task: {task}")
            print(f"Available tasks: {', '.join(TASK_MAP.keys())}")
            return

    # Run cross-evaluation
    results = run_cross_evaluation_matrix(
        eval_tasks=eval_tasks,
        adapter_tasks=adapter_tasks,
        split=args.split,
        batch_size=args.batch_size,
        use_cache=args.use_cache,
        skip_base=args.skip_base,
        generate_confusion_matrix=args.confusion_matrix,
    )

    # Print summary table
    print_summary_table(results)

    # Save summary if requested
    if args.output_summary:
        output_path = Path(args.output_summary)
        if not output_path.is_absolute():
            output_path = PACKAGE_ROOT / output_path
        save_summary(results, output_path)
    else:
        # Save to default location
        default_path = PACKAGE_ROOT / "artifacts" / "evaluation" / f"cross_task_summary_{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_summary(results, default_path)


if __name__ == "__main__":
    main()
