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

from merging import (
    merge_uniform,
    merge_weighted,
    resolve_best_adapter,
    create_merge_output_path,
    evaluate_merged_adapter,
)
from experiments.extract_vector import extract_task_vector_from_lora


def merge_uniform_via_task_vectors(
    adapter_paths: List[Path],
    output_path: Path,
    merge_mode: str = "common",
    show_progress: bool = True,
) -> None:
    """Merge adapters by extracting and merging task vectors.

    This approach works even when adapters have different LoRA ranks,
    as we merge in the full parameter space rather than the low-rank space.

    Args:
        adapter_paths: List of paths to adapter directories
        output_path: Directory to save merged task vector
        merge_mode: How to handle different parameters ("common" or "strict")
        show_progress: Whether to print progress messages

    Note:
        This is a legacy function maintained for backwards compatibility.
        For most use cases, prefer merge_uniform() which works on LoRA weights directly.
    """
    from merging.utils import save_merged_adapter
    import torch
    from datetime import datetime

    if show_progress:
        print(f"🔀 Merging {len(adapter_paths)} adapters via task vectors")
        print(f"   Mode: {merge_mode}")
        for i, path in enumerate(adapter_paths, 1):
            print(f"   {i}. {path}")

    # Extract task vectors from each adapter
    task_vectors = []
    for i, adapter_path in enumerate(adapter_paths, 1):
        if show_progress:
            print(f"\n📥 Extracting task vector {i}/{len(adapter_paths)}: {adapter_path.name}")
        tv = extract_task_vector_from_lora(adapter_path)
        task_vectors.append(tv)

    # Find common keys
    all_keys = [set(tv.keys()) for tv in task_vectors]
    common_keys = set.intersection(*all_keys)
    all_unique_keys = set.union(*all_keys)

    if merge_mode == "strict":
        reference_keys = all_keys[0]
        for i, tv_keys in enumerate(all_keys[1:], start=1):
            if tv_keys != reference_keys:
                missing = reference_keys - tv_keys
                extra = tv_keys - reference_keys
                raise ValueError(
                    f"Task vector {i} has different parameters than task vector 0.\n"
                    f"Missing: {missing}\nExtra: {extra}"
                )
        keys_to_merge = reference_keys
    else:  # common mode
        keys_to_merge = common_keys
        unique_keys = all_unique_keys - common_keys

        if unique_keys:
            print(f"⚠️  Warning: {len(unique_keys)} parameters are not common across all task vectors")
            print(f"   Only merging {len(keys_to_merge)} common parameters")

    # Average all common parameters
    merged_tv = {}
    num_vectors = len(task_vectors)

    if show_progress:
        print(f"\n🧮 Computing uniform average of task vectors...")

    for key in keys_to_merge:
        # Check shapes match
        shapes = [tv[key].shape for tv in task_vectors]
        if len(set(shapes)) > 1:
            print(f"⚠️  Skipping {key}: inconsistent shapes {shapes}")
            continue

        # Sum and average
        summed = torch.zeros_like(task_vectors[0][key])
        for tv in task_vectors:
            summed += tv[key]

        merged_tv[key] = summed / num_vectors

    # Build metadata
    metadata = {
        "merge_method": "task_vector",
        "merge_mode": merge_mode,
        "num_adapters": len(adapter_paths),
        "timestamp": datetime.now().isoformat(),
        "source_adapters": [{"path": str(p)} for p in adapter_paths],
        "num_parameters": len(merged_tv),
    }

    # Save merged task vector as an adapter
    if show_progress:
        print(f"\n💾 Saving merged task vector to {output_path}")

    save_merged_adapter(
        weights=merged_tv,
        output_path=output_path,
        reference_adapter_path=adapter_paths[0],
        metadata=metadata,
        register_run=True,
    )

    if show_progress:
        print(f"\n✅ Task vector merge complete!")


def resolve_adapter_specs(
    adapter_specs: List[str],
) -> List[tuple[Path, Optional[dict]]]:
    """Resolve adapter specifications to paths and metadata.

    Args:
        adapter_specs: List of task names or paths

    Returns:
        List of (adapter_path, metadata) tuples

    Raises:
        ValueError: If adapter spec cannot be resolved
    """
    resolved = []

    for spec in adapter_specs:
        path = Path(spec)

        # Check if it's a direct path
        if path.exists() and path.is_dir():
            # Direct path provided
            adapter_path = path.resolve()
            metadata = {"path": str(adapter_path)}
            resolved.append((adapter_path, metadata))
            print(f"✅ Using adapter at: {adapter_path}")

        else:
            # Treat as task name
            try:
                adapter_path, metadata = resolve_best_adapter(spec)
                print(f"✅ Resolved '{spec}' to: {adapter_path}")
                print(f"   Metrics: {metadata.get('metrics', {})}")
                resolved.append((adapter_path, metadata))

            except Exception as e:
                raise ValueError(
                    f"Could not resolve adapter spec '{spec}': {e}\n"
                    f"Provide either a task name (e.g., 'asr') or a valid adapter path."
                )

    return resolved


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

    # Resolve adapter specifications
    print(f"\n📂 Resolving {len(adapter_specs)} adapter(s)...")
    resolved = resolve_adapter_specs(adapter_specs)

    adapter_paths = [path for path, _ in resolved]
    source_metadata = [meta for _, meta in resolved]

    # Extract task names for output path
    task_names = [meta.get("task", f"adapter{i}") for i, meta in enumerate(source_metadata)]

    # Determine output path
    if output:
        output_path = Path(output)
        if not output_path.is_absolute():
            output_path = PACKAGE_ROOT / output_path
        # Create run directory
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_path / "runs" / f"run_{timestamp}"
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        # Auto-generate output path
        extra_params = {}
        if method == "weighted":
            extra_params["lambda"] = lambda_weight

        output_path = create_merge_output_path(
            method=method if method != "task_vector" else "uniform",  # Task vector uses uniform dir
            task_names=task_names,
            extra_params=extra_params,
        )

    print(f"\n💾 Output directory: {output_path}")

    # Perform merge based on method
    if method == "uniform":
        if len(adapter_paths) < 2:
            raise ValueError("Uniform merging requires at least 2 adapters")

        merged_path = merge_uniform(
            adapter_paths=adapter_paths,
            output_path=output_path,
            source_metadata=source_metadata,
            merge_mode=merge_mode,
        )

    elif method == "weighted":
        if len(adapter_paths) != 2:
            raise ValueError(
                f"Weighted merging requires exactly 2 adapters, got {len(adapter_paths)}"
            )

        merged_path = merge_weighted(
            adapter1_path=adapter_paths[0],
            adapter2_path=adapter_paths[1],
            lambda_weight=lambda_weight,
            output_path=output_path,
            source_metadata=source_metadata,
            merge_mode=merge_mode,
        )

    elif method == "task_vector":
        if len(adapter_paths) < 2:
            raise ValueError("Task vector merging requires at least 2 adapters")

        merge_uniform_via_task_vectors(
            adapter_paths=adapter_paths,
            output_path=output_path,
            merge_mode=merge_mode,
        )
        merged_path = output_path

    else:
        raise ValueError(f"Unknown merge method: {method}")

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
