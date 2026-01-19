"""Unified CLI entrypoint for speech merging workflows."""

import argparse
import sys

from experiments import evaluate_task
from experiments.train_task import main as train_task_main


def parse_args() -> argparse.Namespace:
    """Parse high-level command."""
    parser = argparse.ArgumentParser(description="Speech merging pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run task training.")
    train_parser.add_argument("--task", default="asr", help="Task name to run.")
    train_parser.add_argument("--config", default=None, help="Config filename override.")

    merge_parser = subparsers.add_parser("merge", help="Merge trained adapters.")
    merge_parser.add_argument(
        "--adapters",
        nargs="+",
        required=True,
        help="Task names or paths to merge (e.g., 'asr emotion')",
    )
    merge_parser.add_argument(
        "--method",
        default="uniform",
        choices=["uniform", "weighted", "task_vector"],
        help="Merging method: uniform (equal avg), weighted (lambda), task_vector (full space)",
    )
    merge_parser.add_argument(
        "--lambda",
        type=float,
        dest="lambda_weight",
        default=0.5,
        help="Lambda weight for weighted merging (0.0 to 1.0)",
    )
    merge_parser.add_argument(
        "--merge-mode",
        default="common",
        choices=["common", "strict"],
        help="Parameter handling: common (merge common params) or strict (require identical)",
    )
    merge_parser.add_argument(
        "--output",
        default=None,
        help="Output directory override (auto-generated if not specified)",
    )
    merge_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate merged adapter on source tasks after merging",
    )
    merge_parser.add_argument(
        "--eval-split",
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split for evaluation",
    )

    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation for a task.")
    eval_parser.add_argument("--task", default="asr", help="Task name to evaluate.")
    eval_parser.add_argument("--config", default=None, help="Config filename override.")
    eval_parser.add_argument("--adapter", default=None, help="Path to adapter directory.")
    eval_parser.add_argument(
        "--split",
        default="validation",
        choices=("train", "validation", "test"),
        help="Dataset split to evaluate.",
    )
    eval_parser.add_argument("--batch-size", type=int, default=None, help="Eval batch size override.")
    eval_parser.add_argument("--save-json", default=None, help="Optional metrics JSON output path.")
    eval_parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse cached metrics for individual evaluations when available.",
    )
    eval_parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare adapter with base model performance (runs base model eval automatically).",
    )
    eval_parser.add_argument(
        "--confusion-matrix",
        action="store_true",
        dest="confusion_matrix",
        help="Generate confusion matrix for classification tasks.",
    )
    eval_parser.add_argument(
        "--no-confusion-matrix",
        action="store_false",
        dest="confusion_matrix",
        help="Disable confusion matrix generation (default: True).",
    )
    eval_parser.set_defaults(confusion_matrix=True)

    eval_merged_parser = subparsers.add_parser("evaluate-merged", help="Run evaluation for a merged adapter.")
    eval_merged_parser.add_argument("--adapter-path", default=None, help="Path to merged adapter run/base directory.")
    eval_merged_parser.add_argument(
        "--method",
        default=None,
        choices=("uniform", "weighted", "task_vector"),
        help="Merge method used (for resolving adapter path).",
    )
    eval_merged_parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Tasks merged into the adapter (e.g., asr intent).",
    )
    eval_merged_parser.add_argument(
        "--lambda",
        type=float,
        dest="lambda_weight",
        default=None,
        help="Lambda weight for weighted merges (optional).",
    )
    eval_merged_parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier to resolve (best, latest, or run_YYYYMMDD_HHMMSS).",
    )
    eval_merged_parser.add_argument(
        "--eval-tasks",
        nargs="+",
        default=None,
        help="Tasks to evaluate on (defaults to merged source tasks).",
    )
    eval_merged_parser.add_argument(
        "--split",
        default="test",
        choices=("train", "validation", "test"),
        help="Dataset split to evaluate.",
    )
    eval_merged_parser.add_argument("--batch-size", type=int, default=None, help="Eval batch size override.")
    eval_merged_parser.add_argument(
        "--confusion-matrix",
        action="store_true",
        dest="confusion_matrix",
        help="Generate confusion matrix for classification tasks.",
    )
    eval_merged_parser.add_argument(
        "--no-confusion-matrix",
        action="store_false",
        dest="confusion_matrix",
        help="Disable confusion matrix generation (default: True).",
    )
    eval_merged_parser.add_argument(
        "--no-save-results",
        action="store_false",
        dest="save_results",
        help="Disable saving merged evaluation summary.",
    )
    eval_merged_parser.set_defaults(confusion_matrix=True, save_results=True)

    return parser.parse_args()


def dispatch_train(args: argparse.Namespace) -> None:
    """Invoke the training workflow."""
    argv = ["train_task.py"]
    if args.task:
        argv.extend(["--task", args.task])
    if args.config:
        argv.extend(["--config", args.config])
    sys.argv = argv
    train_task_main()


def dispatch_merge(args: argparse.Namespace) -> None:
    """Invoke the merge workflow."""
    from experiments.merge_vectors import merge_adapters_cli

    merge_adapters_cli(
        adapter_specs=args.adapters,
        method=args.method,
        lambda_weight=args.lambda_weight,
        merge_mode=args.merge_mode,
        output=args.output,
        evaluate=args.evaluate,
        eval_split=args.eval_split,
    )


def dispatch_evaluate(args: argparse.Namespace) -> None:
    """Invoke the evaluation workflow."""
    compare = bool(args.adapter) and args.compare

    if compare:
        base_result = evaluate_task.evaluate(
            task=args.task,
            config_name=args.config,
            adapter=None,
            split=args.split,
            batch_size=args.batch_size,
            save_json=None,
            enable_cache=True,
            show_summary=False,
        )
        adapter_result = evaluate_task.evaluate(
            task=args.task,
            config_name=args.config,
            adapter=args.adapter,
            split=args.split,
            batch_size=args.batch_size,
            save_json=args.save_json,
            enable_cache=args.use_cache,
            show_summary=False,
        )

        print(f"📊 Comparison for {args.task}/{args.split}")
        metrics_keys = sorted(set(base_result.metrics.keys()) | set(adapter_result.metrics.keys()))
        for key in metrics_keys:
            base_value = base_result.metrics.get(key)
            adapter_value = adapter_result.metrics.get(key)

            if isinstance(base_value, (int, float)) and isinstance(adapter_value, (int, float)):
                delta = adapter_value - base_value
                print(
                    f"  {key}: base={base_value:.4f}, adapter={adapter_value:.4f}, Δ={delta:+.4f}"
                )
            else:
                print(f"  {key}: base={base_value}, adapter={adapter_value}")

        if base_result.cache_path:
            if base_result.cache_used:
                print(f"♻️ Reused base metrics from {base_result.cache_path}")
            else:
                print(f"💾 Cached base metrics at {base_result.cache_path}")

        if adapter_result.save_path:
            print(f"💾 Adapter metrics saved to {adapter_result.save_path}")
    else:
        evaluate_task.evaluate(
            task=args.task,
            config_name=args.config,
            adapter=args.adapter,
            split=args.split,
            batch_size=args.batch_size,
            save_json=args.save_json,
            enable_cache=args.use_cache,
            show_summary=True,
            generate_confusion_matrix=args.confusion_matrix,
        )


def dispatch_evaluate_merged(args: argparse.Namespace) -> None:
    """Invoke the merged evaluation workflow."""
    from experiments.evaluate_merged import evaluate_from_args

    evaluate_from_args(args)


def main() -> None:
    """Dispatch commands."""
    args = parse_args()
    if args.command == "train":
        dispatch_train(args)
    elif args.command == "merge":
        dispatch_merge(args)
    elif args.command == "evaluate":
        dispatch_evaluate(args)
    elif args.command == "evaluate-merged":
        dispatch_evaluate_merged(args)
    else:
        raise NotImplementedError(f"Command '{args.command}' not supported.")


if __name__ == "__main__":
    main()
