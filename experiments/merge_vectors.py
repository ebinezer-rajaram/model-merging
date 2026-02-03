"""Merge LoRA adapter vectors using various strategies.

This is the main entry point for adapter merging. It provides a CLI interface
and delegates to specific merging methods in the /merging package.

Usage:
    # Uniform merging (equal weighting)
    python experiments/merge_vectors.py \\
        --adapters asr emotion \\
        --method uniform \\
        --evaluate

    # Weighted merging (with lambda)
    python experiments/merge_vectors.py \\
        --adapters asr emotion \\
        --method weighted \\
        --lambda 0.7 \\
        --evaluate

    # Task vector merging (supports different LoRA ranks)
    python experiments/merge_vectors.py \\
        --adapters asr emotion intent \\
        --method task_vector \\
        --evaluate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add package root to path
CURRENT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CURRENT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from merging import evaluate_merged_adapter
from merging.core.runner import run_merge


def merge_adapters_cli(
    adapter_specs: List[str],
    method: str = "uniform",
    lambda_weight: float = 0.5,
    merge_mode: str = "common",
    output: Optional[str] = None,
    evaluate: bool = False,
    eval_split: str = "test",
) -> None:
    """CLI entry point for adapter merging.

    Args:
        adapter_specs: List of adapter specifications (task names or paths)
        method: Merging method ("uniform", "weighted", "task_vector")
        lambda_weight: Lambda weight for weighted merging (0.0 to 1.0)
        merge_mode: How to handle parameter mismatches ("common" or "strict")
        output: Optional output path override
        evaluate: Whether to evaluate merged adapter
        eval_split: Dataset split for evaluation
    """
    print("\n" + "="*60)
    print("🔀 Adapter Merging")
    print("="*60)

    output_path, _merge_output, task_names = run_merge(
        adapter_specs=adapter_specs,
        method=method,
        lambda_weight=lambda_weight,
        merge_mode=merge_mode,
        output=output,
        save_merged=True,
        show_progress=True,
    )
    merged_path = output_path

    # Evaluate if requested
    if evaluate:
        print("\n" + "="*60)
        print("📊 Evaluating Merged Adapter")
        print("="*60)

        results = evaluate_merged_adapter(
            adapter_path=merged_path,
            task_names=task_names,
            eval_tasks=task_names,
            split=eval_split,
            save_results=True,
            merge_mode=merge_mode,
        )

        # Print summary
        print("\n" + "="*60)
        print("📈 Evaluation Summary")
        print("="*60)
        for task, metrics in results.items():
            print(f"\n{task.upper()}:")
            if "error" in metrics:
                print(f"  ❌ Error: {metrics['error']}")
            else:
                for key, value in sorted(metrics.items())[:5]:
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")

    print("\n" + "="*60)
    print("✅ Merging Complete")
    print("="*60)
    print(f"\nMerged adapter saved to:")
    print(f"  {merged_path}")
    print(f"\nTo evaluate on a specific task:")
    print(f"  python main.py evaluate --task <task> --adapter {merged_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
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
        required=True,
        help="Adapter specifications: task names (e.g., 'asr emotion') or paths",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="uniform",
        choices=["uniform", "weighted", "task_vector"],
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
    """Entry point."""
    args = parse_args()

    try:
        merge_adapters_cli(
            adapter_specs=args.adapters,
            method=args.method,
            lambda_weight=args.lambda_weight,
            merge_mode=args.merge_mode,
            output=args.output,
            evaluate=args.evaluate,
            eval_split=args.eval_split,
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
