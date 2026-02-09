"""CLI helpers for merge workflows."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from merging import evaluate_merged_adapter
from merging.runtime.logging import banner
from merging.engine.registry import list_merge_methods
from merging.engine.runner import run_merge
from merging.config.unified import load_merge_config


def merge_adapters_cli(
    adapter_specs: Optional[List[str]],
    method: str = "uniform",
    lambda_weight: float = 0.5,
    merge_mode: str = "common",
    output: Optional[str] = None,
    config: Optional[str] = None,
    evaluate: bool = False,
    eval_split: str = "test",
) -> None:
    """CLI entry point for adapter merging."""
    banner("🔀 Adapter Merging")

    effective_specs = adapter_specs or []
    if not config and not effective_specs:
        raise ValueError("Provide --adapters or --config.")

    merge_result = run_merge(
        adapter_specs=effective_specs if not config else [],
        method=method,
        lambda_weight=lambda_weight,
        merge_mode=merge_mode,
        output=output,
        save_merged=True,
        show_progress=True,
        merge_spec=load_merge_config(config).to_merge_spec() if config else None,
    )
    merged_path = merge_result.output_path
    task_names = merge_result.task_names

    if evaluate:
        banner("📊 Evaluating Merged Adapter")

        results = evaluate_merged_adapter(
            adapter_path=merged_path,
            task_names=task_names,
            eval_tasks=task_names,
            split=eval_split,
            save_results=True,
            merge_mode=merge_mode,
        )

        banner("📈 Evaluation Summary")
        for task, metrics in results.items():
            print(f"\n{task.upper()}:")
            if "error" in metrics:
                print(f"  ❌ Error: {metrics['error']}")
            else:
                for key, value in sorted(metrics.items())[:5]:
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")

    banner("✅ Merging Complete")
    print(f"\nMerged adapter saved to:\n  {merged_path}")
    print(f"\nTo evaluate on a specific task:")
    print(f"  python main.py evaluate --task <task> --adapter {merged_path}")


def parse_args() -> argparse.Namespace:
    methods = list_merge_methods()
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters using various strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Uniform merging (equal weighting)
  python experiments/merge_vectors.py --adapters asr emotion --method uniform --evaluate

  # Weighted merging with lambda=0.7 (70% asr, 30% emotion)
  python experiments/merge_vectors.py --adapters asr emotion --method weighted --lambda 0.7

  # Task vector merging (works with different LoRA ranks)
  python experiments/merge_vectors.py --adapters asr emotion intent --method task_vector

  # Custom output path
  python experiments/merge_vectors.py --adapters asr emotion --output artifacts/my_merge
        """,
    )
    parser.add_argument(
        "--adapters",
        nargs="+",
        required=False,
        help="Adapter specifications: task names (e.g., 'asr emotion') or paths",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="uniform",
        choices=methods,
        help="Merging method: 'uniform' (equal averaging), 'weighted' (lambda-based), "
             "'task_vector' (merge via task vectors, works with different ranks)",
    )
    parser.add_argument(
        "--lambda",
        type=float,
        default=0.5,
        dest="lambda_weight",
        help="Lambda weight for weighted merging (0.0 to 1.0). "
             "E.g., 0.7 means 70%% first adapter, 30%% second adapter",
    )
    parser.add_argument(
        "--merge-mode",
        type=str,
        default="common",
        choices=["common", "strict"],
        help="How to handle different parameters: "
             "'common' (merge only common params) or 'strict' (require identical params)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for merged adapter (auto-generated if not specified)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Advanced merge YAML config path (overrides adapters/method/lambda when provided).",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate merged adapter on all source tasks after merging",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to use for evaluation (default: test)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        merge_adapters_cli(
            adapter_specs=args.adapters,
            method=args.method,
            lambda_weight=args.lambda_weight,
            merge_mode=args.merge_mode,
            output=args.output,
            config=args.config,
            evaluate=args.evaluate,
            eval_split=args.eval_split,
        )
    except Exception as exc:
        print(f"\n❌ Error: {exc}")
        sys.exit(1)


def merge_from_args(args: argparse.Namespace) -> None:
    merge_adapters_cli(
        adapter_specs=args.adapters,
        method=args.method,
        lambda_weight=args.lambda_weight,
        merge_mode=args.merge_mode,
        output=args.output,
        config=getattr(args, "config", None),
        evaluate=args.evaluate,
        eval_split=args.eval_split,
    )
