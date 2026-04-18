"""Unified CLI entrypoint for speech merging workflows."""

import argparse
import os
import sys
from pathlib import Path

import core.evaluation.evaluate_task as evaluate_task
from core.training.train_task import main as train_task_main
from core.training.train_multitask import main as train_multitask_main


def _configure_cuda_allocator_env() -> None:
    """Set safer CUDA allocator defaults unless explicitly overridden."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def parse_args() -> argparse.Namespace:
    """Parse high-level command."""
    from merging.engine.registry import list_merge_methods

    merge_methods = list_merge_methods()
    sweep_methods = sorted(set(merge_methods + ["continual"]))
    parser = argparse.ArgumentParser(description="Speech merging pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run task training.")
    train_parser.add_argument("--task", default="asr", help="Task name to run.")
    train_parser.add_argument("--config", default=None, help="Config filename override.")

    mtl_parser = subparsers.add_parser("mtl", help="Run joint multi-task training.")
    mtl_parser.add_argument("--config", required=True, help="MTL config path.")
    mtl_parser.add_argument("--continual-enabled", action="store_true", help="Enable continual stage-2 MTL mode.")
    mtl_parser.add_argument("--base-adapter", default=None, help="Base MTL adapter path for continual mode.")
    mtl_parser.add_argument(
        "--base-adapter-run-id",
        default=None,
        help="Optional alias under base adapter root (best/latest/run_*).",
    )
    mtl_parser.add_argument("--added-tasks", nargs="+", default=None, help="Added tasks for continual stage-2 updates.")
    mtl_parser.add_argument("--base-tasks-override", nargs="+", default=None, help="Optional base task override list.")
    mtl_parser.add_argument(
        "--selection-mode",
        default=None,
        choices=("mtl_interference", "added_task_metric"),
        help="Continual checkpoint-selection mode override.",
    )
    mtl_parser.add_argument(
        "--selection-task-set",
        default=None,
        choices=("base_plus_added",),
        help="Continual selection task-set override.",
    )
    mtl_parser.add_argument(
        "--final-eval-include-speech-qa",
        action="store_true",
        help="Include speech_qa in final continual evaluation.",
    )
    mtl_parser.add_argument(
        "--no-final-eval-include-speech-qa",
        action="store_false",
        dest="final_eval_include_speech_qa",
        help="Exclude speech_qa from final continual evaluation.",
    )
    mtl_parser.set_defaults(final_eval_include_speech_qa=None)

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
        "--config",
        default=None,
        help="Optional merge config YAML used to source method/tasks/params for checkpoint replay.",
    )
    eval_merged_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a SuperMerge checkpoint file (.pt) to evaluate directly.",
    )
    eval_merged_parser.add_argument(
        "--optimizer-steps",
        type=int,
        default=None,
        help="Override optimizer steps for checkpoint replay (defaults to checkpoint's optimizer_step).",
    )
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
        "--scale",
        type=float,
        default=None,
        help="Global scale for uniform_scalar_delta merges (optional).",
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
        choices=sweep_methods,
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

    materialize_parser = subparsers.add_parser(
        "materialize-merged-artifact",
        help="Materialize an existing merged run into a reusable continual compressed artifact.",
    )
    materialize_parser.add_argument(
        "--merged-run-path",
        required=True,
        help="Path to merged run directory containing merge_metadata.json.",
    )
    materialize_parser.add_argument(
        "--output",
        default=None,
        help="Output directory for the continual artifact.",
    )
    materialize_parser.add_argument(
        "--energy-threshold",
        type=float,
        default=0.99,
        help="Per-parameter retained-energy threshold for SVD compression.",
    )
    materialize_parser.add_argument(
        "--store-dtype",
        default="float16",
        choices=("float16", "bfloat16", "float32"),
        help="Storage dtype for factor tensors.",
    )
    materialize_parser.add_argument(
        "--merge-mode",
        default=None,
        choices=("common", "strict"),
        help="Optional override for key merge mode during materialization.",
    )

    continual_merge_parser = subparsers.add_parser(
        "continual-merge",
        help="Merge an existing reusable artifact (or adapter) with an incoming source.",
    )
    continual_merge_parser.add_argument("--x-source", required=True, help="Existing merged source (artifact path or adapter/task spec).")
    continual_merge_parser.add_argument("--y-source", required=True, help="Incoming source (artifact path or adapter/task spec).")
    continual_merge_parser.add_argument("--alpha", type=float, default=1.0, help="Global alpha coefficient.")
    continual_merge_parser.add_argument(
        "--lambda",
        dest="lambda_weight",
        type=float,
        default=0.5,
        help="Global lambda coefficient for x in alpha*(lambda*x + (1-lambda)*y).",
    )
    continual_merge_parser.add_argument(
        "--output",
        default=None,
        help="Output directory for merged continual artifact.",
    )
    continual_merge_parser.add_argument(
        "--energy-threshold",
        type=float,
        default=0.99,
        help="Per-parameter retained-energy threshold for SVD compression.",
    )
    continual_merge_parser.add_argument(
        "--store-dtype",
        default="float16",
        choices=("float16", "bfloat16", "float32"),
        help="Storage dtype for factor tensors.",
    )
    continual_merge_parser.add_argument(
        "--merge-mode",
        default="common",
        choices=("common", "strict"),
        help="Key-handling behavior across sources.",
    )

    eval_continual_parser = subparsers.add_parser(
        "evaluate-continual",
        help="Evaluate a continual compressed artifact, or sweep alpha/lambda by building artifacts on the fly.",
    )
    eval_continual_parser.add_argument("--artifact-path", default=None, help="Path to an existing continual artifact directory.")
    eval_continual_parser.add_argument("--x-source", default=None, help="Sweep mode: existing source spec.")
    eval_continual_parser.add_argument("--y-source", default=None, help="Sweep mode: incoming source spec.")
    eval_continual_parser.add_argument("--alpha", type=float, default=None, help="Single alpha value (single or sweep mode default).")
    eval_continual_parser.add_argument(
        "--lambda",
        dest="lambda_weight",
        type=float,
        default=None,
        help="Single lambda value (single or sweep mode default).",
    )
    eval_continual_parser.add_argument(
        "--alpha-values",
        nargs="+",
        default=None,
        help="Sweep mode alpha grid (space/comma separated).",
    )
    eval_continual_parser.add_argument(
        "--lambda-values",
        nargs="+",
        default=None,
        help="Sweep mode lambda grid (space/comma separated).",
    )
    eval_continual_parser.add_argument(
        "--output",
        default=None,
        help="Sweep output directory or run output override.",
    )
    eval_continual_parser.add_argument(
        "--eval-tasks",
        nargs="+",
        default=None,
        help="Tasks to evaluate (defaults to constituent task union).",
    )
    eval_continual_parser.add_argument(
        "--split",
        default="test",
        choices=("train", "validation", "test"),
        help="Dataset split to evaluate.",
    )
    eval_continual_parser.add_argument("--batch-size", type=int, default=None, help="Per-device eval batch size.")
    eval_continual_parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse cached base-model metrics when available.",
    )
    eval_continual_parser.add_argument(
        "--no-save-results",
        action="store_false",
        dest="save_results",
        help="Disable saving evaluation summaries.",
    )
    eval_continual_parser.add_argument(
        "--no-compute-interference-baselines",
        action="store_false",
        dest="compute_missing_interference_baselines",
        help="Do not auto-compute baseline metrics for interference_delta.",
    )
    eval_continual_parser.add_argument(
        "--energy-threshold",
        type=float,
        default=0.99,
        help="Sweep mode SVD retained-energy threshold.",
    )
    eval_continual_parser.add_argument(
        "--store-dtype",
        default="float16",
        choices=("float16", "bfloat16", "float32"),
        help="Sweep mode factor storage dtype.",
    )
    eval_continual_parser.add_argument(
        "--merge-mode",
        default="common",
        choices=("common", "strict"),
        help="Sweep mode key merge handling.",
    )
    eval_continual_parser.set_defaults(save_results=True, compute_missing_interference_baselines=True)

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


def dispatch_mtl(args: argparse.Namespace) -> None:
    """Invoke the multi-task training workflow."""
    argv = ["train_multitask.py"]
    if args.config:
        argv.extend(["--config", args.config])
    if args.continual_enabled:
        argv.append("--continual-enabled")
    if args.base_adapter:
        argv.extend(["--base-adapter", args.base_adapter])
    if args.base_adapter_run_id:
        argv.extend(["--base-adapter-run-id", args.base_adapter_run_id])
    if args.added_tasks:
        argv.extend(["--added-tasks", *args.added_tasks])
    if args.base_tasks_override:
        argv.extend(["--base-tasks-override", *args.base_tasks_override])
    if args.selection_mode:
        argv.extend(["--selection-mode", args.selection_mode])
    if args.selection_task_set:
        argv.extend(["--selection-task-set", args.selection_task_set])
    if args.final_eval_include_speech_qa is True:
        argv.append("--final-eval-include-speech-qa")
    elif args.final_eval_include_speech_qa is False:
        argv.append("--no-final-eval-include-speech-qa")
    sys.argv = argv
    train_multitask_main()


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
    from merging.evaluation.cli import evaluate_from_args

    _configure_cuda_allocator_env()
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

    _configure_cuda_allocator_env()
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
            "post_sweep_eval": (
                dict(config.post_sweep_eval) if config.post_sweep_eval is not None else None
            ),
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


def dispatch_materialize_merged_artifact(args: argparse.Namespace) -> None:
    from merging.cli import materialize_merged_artifact_from_cli_args

    materialize_merged_artifact_from_cli_args(args)


def dispatch_continual_merge(args: argparse.Namespace) -> None:
    from merging.cli import continual_merge_from_cli_args

    continual_merge_from_cli_args(args)


def dispatch_evaluate_continual(args: argparse.Namespace) -> None:
    from merging.cli import evaluate_continual_from_cli_args

    _configure_cuda_allocator_env()
    evaluate_continual_from_cli_args(args)


def main() -> None:
    """Dispatch commands."""
    args = parse_args()
    if args.command == "train":
        dispatch_train(args)
    elif args.command == "mtl":
        dispatch_mtl(args)
    elif args.command == "merge":
        dispatch_merge(args)
    elif args.command == "evaluate":
        dispatch_evaluate(args)
    elif args.command == "evaluate-merged":
        dispatch_evaluate_merged(args)
    elif args.command == "merge-sweep":
        dispatch_merge_sweep(args)
    elif args.command == "materialize-merged-artifact":
        dispatch_materialize_merged_artifact(args)
    elif args.command == "continual-merge":
        dispatch_continual_merge(args)
    elif args.command == "evaluate-continual":
        dispatch_evaluate_continual(args)
    else:
        raise NotImplementedError(f"Command '{args.command}' not supported.")


if __name__ == "__main__":
    main()
