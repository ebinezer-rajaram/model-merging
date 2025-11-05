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
        "--no-compare",
        action="store_true",
        help="Skip base-model comparison even if an adapter is provided.",
    )

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


def dispatch_evaluate(args: argparse.Namespace) -> None:
    """Invoke the evaluation workflow."""
    compare = bool(args.adapter) and not args.no_compare

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
        )


def main() -> None:
    """Dispatch commands."""
    args = parse_args()
    if args.command == "train":
        dispatch_train(args)
    elif args.command == "evaluate":
        dispatch_evaluate(args)
    else:
        raise NotImplementedError(f"Command '{args.command}' not supported.")


if __name__ == "__main__":
    main()
