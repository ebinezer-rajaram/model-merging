"""Unified CLI entrypoint for speech merging workflows."""

import argparse
import sys
from pathlib import Path

from experiments import evaluate_task
from experiments.train_task import main as train_task_main


def parse_args() -> argparse.Namespace:
    """Parse high-level command."""
    from merging.engine.registry import list_merge_methods

    merge_methods = list_merge_methods()
    parser = argparse.ArgumentParser(description="Speech merging pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run task training.")
    train_parser.add_argument("--task", default="asr", help="Task name to run.")
    train_parser.add_argument("--config", default=None, help="Config filename override.")

    merge_parser = subparsers.add_parser("merge", help="Merge trained adapters.")
    merge_parser.add_argument(
        "--adapters",
        nargs="+",
        required=False,
        help="Task names or paths to merge (e.g., 'asr emotion')",
    )
    merge_parser.add_argument(
        "--method",
        default="uniform",
        choices=merge_methods,
        help="Merging method.",
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
        "--config",
        default=None,
        help="Advanced merge YAML config path. If provided, adapters/method/lambda are taken from config.",
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
        choices=tuple(merge_methods),
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
    eval_merged_parser.add_argument(
        "--save",
        action="store_true",
        dest="save_merged",
        help="Save merged adapter before evaluation (default: in-memory only).",
    )
    eval_merged_parser.add_argument(
        "--no-compute-interference-baselines",
        action="store_false",
        dest="compute_missing_interference_baselines",
        help="Do not auto-compute base_model/best_adapter metrics needed for interference_delta.",
    )
    eval_merged_parser.set_defaults(compute_missing_interference_baselines=True)
    eval_merged_parser.set_defaults(confusion_matrix=True, save_results=True)

    sweep_parser = subparsers.add_parser("merge-sweep", help="Run a merge hyperparameter sweep.")
    sweep_parser.add_argument("--config", default=None, help="Path to unified merge YAML config.")
    sweep_parser.add_argument(
        "--adapters",
        nargs="+",
        default=None,
        help="Adapter specs (task names or paths) if not using --config.",
    )
    sweep_parser.add_argument(
        "--method",
        default=None,
        choices=merge_methods,
        help="Merge method for sweep (overrides config).",
    )
    sweep_parser.add_argument(
        "--grid",
        action="append",
        default=None,
        help="Grid override like key=0.1,0.2,0.3 (repeatable).",
    )
    sweep_parser.add_argument(
        "--search-type",
        default=None,
        choices=["grid", "bayes"],
        help="Search type override (grid or bayes).",
    )
    sweep_parser.add_argument(
        "--merge-mode",
        default=None,
        choices=["common", "strict"],
        help="Merge mode override.",
    )
    sweep_parser.add_argument(
        "--eval-tasks",
        nargs="+",
        default=None,
        help="Tasks to evaluate (overrides config).",
    )
    sweep_parser.add_argument(
        "--split",
        default=None,
        choices=["train", "validation", "test"],
        help="Dataset split override.",
    )
    sweep_parser.add_argument(
        "--save-merged",
        action="store_true",
        help="Save merged adapters during sweep (overrides config).",
    )
    sweep_parser.add_argument(
        "--allow-negative",
        action="store_true",
        help="Allow negative interference delta in ranking (default: disallow).",
    )
    sweep_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for sweep summary.",
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


def dispatch_merge(args: argparse.Namespace) -> None:
    """Invoke the merge workflow."""
    from merging.cli import merge_from_args

    merge_from_args(args)


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


def _parse_grid_args(grid_args: list[str] | None) -> dict:
    if not grid_args:
        return {}
    grid: dict = {}
    for entry in grid_args:
        if "=" not in entry:
            raise ValueError(f"Invalid grid entry '{entry}'. Expected key=val1,val2")
        key, values_str = entry.split("=", 1)
        values = []
        for raw in values_str.split(","):
            raw = raw.strip()
            if raw == "":
                continue
            try:
                if "." in raw or "e" in raw or "E" in raw:
                    values.append(float(raw))
                else:
                    values.append(int(raw))
            except ValueError:
                values.append(raw)
        grid[key] = values
    return grid


def dispatch_merge_sweep(args: argparse.Namespace) -> None:
    from merging.config.unified import load_merge_config, normalize_merge_config
    from merging.evaluation.sweep import run_sweep

    if args.config:
        config = load_merge_config(Path(args.config))
        config_dict = {
            "adapters": list(config.adapters),
            "method": config.method,
            "merge_mode": config.merge_mode,
            "method_params": dict(config.method_params),
            "search": (dict(config.search) if config.search is not None else None),
            "eval_tasks": (list(config.eval_tasks) if config.eval_tasks is not None else None),
            "split": config.split,
            "save_merged": config.save_merged,
            "constraint_nonnegative": config.constraint_nonnegative,
            "eval_subset": (dict(config.eval_subset) if config.eval_subset is not None else None),
            "output_dir": str(config.output_dir) if config.output_dir is not None else None,
            "compute_missing_interference_baselines": config.compute_missing_interference_baselines,
        }
        if config.lambda_policy is not None:
            config_dict["lambda_policy"] = {
                "type": config.lambda_policy.type,
                "value": config.lambda_policy.value,
                "default": config.lambda_policy.default,
                "overrides": dict(config.lambda_policy.overrides),
            }
        if config.optimizer is not None:
            config_dict["optimizer"] = {
                "type": config.optimizer.type,
                "params": dict(config.optimizer.params),
            }
    else:
        if not args.adapters or not args.method:
            raise ValueError("Provide --config or both --adapters and --method.")
        config_dict = {
            "adapters": args.adapters,
            "method": args.method,
            "search": {"type": "grid", "grid": _parse_grid_args(args.grid)},
        }

    # Overrides
    if args.adapters:
        config_dict["adapters"] = args.adapters
    if args.method:
        config_dict["method"] = args.method
    if args.grid:
        grid = _parse_grid_args(args.grid)
        search = config_dict.get("search", {"type": "grid"})
        search["grid"] = grid
        config_dict["search"] = search
    if args.search_type:
        search = config_dict.get("search") or {}
        if not isinstance(search, dict):
            search = {}
        search["type"] = args.search_type
        config_dict["search"] = search
    if args.merge_mode:
        config_dict["merge_mode"] = args.merge_mode
    if args.eval_tasks:
        config_dict["eval_tasks"] = args.eval_tasks
    if args.split:
        config_dict["split"] = args.split
    if args.save_merged:
        config_dict["save_merged"] = True
    if args.allow_negative:
        config_dict["constraint_nonnegative"] = False
    if args.output_dir:
        config_dict["output_dir"] = args.output_dir

    run_sweep(normalize_merge_config(config_dict))


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
    elif args.command == "merge-sweep":
        dispatch_merge_sweep(args)
    else:
        raise NotImplementedError(f"Command '{args.command}' not supported.")


if __name__ == "__main__":
    main()
