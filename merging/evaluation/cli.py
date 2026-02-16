"""CLI for evaluating merged adapters."""

from __future__ import annotations

import argparse
from typing import List, Optional

from merging.engine.registry import list_merge_methods
from merging.evaluation.evaluate import evaluate_merged_adapter


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    merge_methods = tuple(list_merge_methods())
    parser = argparse.ArgumentParser(description="Evaluate merged adapters on one or more tasks.")
    parser.add_argument("--adapter-path", default=None, help="Path to merged adapter run or base directory.")
    parser.add_argument(
        "--method",
        default=None,
        choices=merge_methods,
        help="Merge method used (for resolving adapter path).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Tasks merged into the adapter (e.g., asr intent).",
    )
    parser.add_argument(
        "--lambda",
        type=float,
        dest="lambda_weight",
        default=None,
        help="Lambda weight for weighted merges (optional).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Global scale for uniform_scalar_delta merges (optional).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier to resolve (best, latest, or run_YYYYMMDD_HHMMSS).",
    )
    parser.add_argument(
        "--eval-tasks",
        nargs="+",
        default=None,
        help="Tasks to evaluate on (defaults to merged source tasks).",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "validation", "test"),
        help="Dataset split to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device eval batch size.")
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
        help="Disable confusion matrix generation (default: True).",
    )
    parser.add_argument(
        "--no-save-results",
        action="store_false",
        dest="save_results",
        help="Disable saving merged evaluation summary.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        dest="save_merged",
        help="Save merged adapter before evaluation (default: in-memory only).",
    )
    parser.add_argument(
        "--no-compute-interference-baselines",
        action="store_false",
        dest="compute_missing_interference_baselines",
        help="Do not auto-compute base_model/best_adapter metrics needed for interference_delta.",
    )
    parser.set_defaults(compute_missing_interference_baselines=True)
    parser.set_defaults(confusion_matrix=True, save_results=True)
    return parser.parse_args()


def evaluate_from_args(args: argparse.Namespace) -> dict:
    """Execute merged adapter evaluation based on CLI args."""
    params = {}
    if args.scale is not None:
        params["scale"] = float(args.scale)
    if args.lambda_weight is not None:
        params.setdefault("lambda", float(args.lambda_weight))

    return evaluate_merged_adapter(
        adapter_path=args.adapter_path,
        method=args.method,
        task_names=_normalize_list(args.tasks),
        lambda_weight=args.lambda_weight,
        params=params or None,
        run_id=args.run_id,
        eval_tasks=_normalize_list(args.eval_tasks),
        split=args.split,
        batch_size=args.batch_size,
        generate_confusion_matrix=args.confusion_matrix,
        save_merged=args.save_merged,
        save_results=args.save_results,
        show_summary=True,
        compute_missing_interference_baselines=args.compute_missing_interference_baselines,
    )


def _normalize_list(value: Optional[List[str]]) -> Optional[List[str]]:
    if value:
        return list(value)
    return None


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    evaluate_from_args(args)


if __name__ == "__main__":
    main()
