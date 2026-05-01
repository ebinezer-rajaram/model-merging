#!/usr/bin/env python3
"""CLI entrypoint for the unified results collection pipeline.

Scans all experiment output directories under --artifacts-root, converts
results into a unified long-format DataFrame, and writes to parquet (and
optionally CSV).

Usage examples:

    # Collect everything, write parquet + CSV:
    python scripts/analysis/collect_all_results.py \\
        --artifacts-root artifacts/ \\
        --output analysis/results/all_experiments.parquet \\
        --csv

    # Merge results only, test split, restrict to two methods:
    python scripts/analysis/collect_all_results.py \\
        --artifacts-root artifacts/ \\
        --output analysis/results/merge_only.parquet \\
        --no-single-task --no-mtl \\
        --splits test \\
        --methods weighted_delta_n uniform_delta

    # Include subset evaluations and keep only best lambda per combo:
    python scripts/analysis/collect_all_results.py \\
        --artifacts-root artifacts/ \\
        --output analysis/results/all.parquet \\
        --include-subset-evals \\
        --best-lambda-only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

from _repo import find_repo_root

# Ensure the project root is on sys.path when run as a script.
_REPO_ROOT = find_repo_root(__file__)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect all experiment results into a unified DataFrame.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts"),
        help="Root directory of experiment artifacts (default: artifacts/).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/results/all_experiments.parquet"),
        help="Output path (.parquet). Default: analysis/results/all_experiments.parquet",
    )
    p.add_argument(
        "--csv",
        action="store_true",
        help="Also write a .csv file alongside the parquet output.",
    )

    # Collector toggles
    p.add_argument("--no-single-task", action="store_true", help="Skip single-task experiments.")
    p.add_argument("--no-mtl", action="store_true", help="Skip MTL experiments.")
    p.add_argument("--no-merge", action="store_true", help="Skip merge experiments.")

    # Single-task options
    p.add_argument(
        "--all-runs",
        action="store_true",
        help="For single-task: collect all run entries, not just the best.",
    )

    # Merge options
    p.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        metavar="SPLIT",
        help="Splits to collect for merge experiments (default: test).",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        metavar="METHOD",
        help="Restrict merge collection to these method names.",
    )
    p.add_argument(
        "--include-subset-evals",
        action="store_true",
        help="Include subset evaluations (files with __ in their stem).",
    )
    p.add_argument(
        "--best-lambda-only",
        action="store_true",
        help="Keep only the best lambda value per (method, task-combo).",
    )

    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )

    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s [%(name)s] %(message)s",
    )
    log = logging.getLogger("collect_all_results")

    artifacts_root = args.artifacts_root.resolve()
    if not artifacts_root.is_dir():
        log.error("artifacts-root does not exist or is not a directory: %s", artifacts_root)
        return 1

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build collectors.
    from core.results import (
        MergeCollector,
        MTLCollector,
        ResultsAggregator,
        SingleTaskCollector,
    )

    collector_overrides = {}

    if args.no_single_task:
        # Replace with an empty do-nothing collector.
        collector_overrides["single_task"] = _NullCollector()
    else:
        collector_overrides["single_task"] = SingleTaskCollector(
            artifacts_root,
            only_best=not args.all_runs,
        )

    if args.no_mtl:
        collector_overrides["mtl"] = _NullCollector()
    else:
        collector_overrides["mtl"] = MTLCollector(artifacts_root)

    if args.no_merge:
        collector_overrides["merge"] = _NullCollector()
    else:
        collector_overrides["merge"] = MergeCollector(
            artifacts_root,
            methods=args.methods,
            splits=args.splits,
            include_subset_evals=args.include_subset_evals,
            best_lambda_only=args.best_lambda_only,
        )

    aggregator = ResultsAggregator(list(collector_overrides.values()))

    log.info("Starting collection from %s ...", artifacts_root)
    df = aggregator.to_dataframe()
    log.info("Collected %d rows (%d unique experiments).",
             len(df), df["experiment_id"].nunique() if len(df) > 0 else 0)

    if len(df) == 0:
        log.warning("No results found. Check that artifacts_root contains experiment outputs.")

    # Write parquet.
    try:
        df.to_parquet(output_path, index=False)
        log.info("Wrote parquet -> %s", output_path)
    except Exception as exc:
        log.error("Failed to write parquet: %s", exc)
        log.info("Trying to write CSV instead ...")
        csv_path = output_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        log.info("Wrote CSV -> %s", csv_path)
        return 0

    # Optionally write CSV.
    if args.csv:
        csv_path = output_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        log.info("Wrote CSV   -> %s", csv_path)

    # Print a quick summary.
    if len(df) > 0:
        print("\n=== Collection summary ===")
        print(df.groupby(["experiment_type", "method"]).size().to_string())
        print(f"\nTotal rows : {len(df)}")
        print(f"Unique exp : {df['experiment_id'].nunique()}")
        print(f"Tasks      : {sorted(df['task'].unique().tolist())}")
        print(f"Splits     : {sorted(df['split'].unique().tolist())}")
        if "eval_context" in df.columns:
            print(f"Contexts   : {sorted(df['eval_context'].dropna().unique().tolist())}")

    return 0


# ---------------------------------------------------------------------------
# Null collector (replaces disabled experiment types)
# ---------------------------------------------------------------------------

class _NullCollector:
    def collect(self):
        return iter([])
    def collect_all(self):
        return []


if __name__ == "__main__":
    sys.exit(main())
