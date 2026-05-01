#!/usr/bin/env python3
"""Backfill experiment_summary.json for all existing artifact directories.

This script is read-only with respect to existing files — it only creates
new experiment_summary.json files alongside the raw outputs produced by
training and evaluation.

Usage:
    venv/bin/python scripts/analysis/generate_outputs.py \\
        --artifacts-root artifacts/ \\
        [--type single_task|mtl|merge|all] \\
        [--dry-run] \\
        [--overwrite]

The script mirrors the parsing logic of analysis/collect/{single_task,mtl,merge}_collector.py
so that the summaries are consistent with what the unified parquet collector sees.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

from _repo import find_repo_root
from typing import Any, Dict, List, Optional, Tuple

# Allow running from the repo root or from scripts/analysis/.
_REPO_ROOT = find_repo_root(__file__)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.output.summary_writer import (  # noqa: E402
    write_experiment_summary,
    build_hyperparameters,
    build_selection,
    TASK_METRICS,
)

logging.basicConfig(
    format="%(levelname)s %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

_EVAL_PREFIX = "eval_"
_MTL_SKIP_PREFIXES = ("eval_mtl_", "global_step", "step", "timestamp", "epoch")
_DEFAULT_MTL_CRITERION = "geometric_mean_interference_delta"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-root", default="artifacts/", type=Path,
        help="Root of the artifacts directory (default: artifacts/)",
    )
    parser.add_argument(
        "--type", default="all",
        choices=["single_task", "mtl", "merge", "all"],
        help="Which experiment type(s) to process (default: all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be written without writing anything",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-generate even if experiment_summary.json already exists",
    )
    args = parser.parse_args()

    root = args.artifacts_root.resolve()
    if not root.is_dir():
        log.error("Artifacts root not found: %s", root)
        sys.exit(1)

    totals = {"written": 0, "skipped": 0, "failed": 0}

    if args.type in ("single_task", "all"):
        _process_single_task(root, args.dry_run, args.overwrite, totals)

    if args.type in ("mtl", "all"):
        _process_mtl(root, args.dry_run, args.overwrite, totals)

    if args.type in ("merge", "all"):
        _process_merge(root, args.dry_run, args.overwrite, totals)

    log.info(
        "Done. written=%d  skipped=%d  failed=%d",
        totals["written"], totals["skipped"], totals["failed"],
    )


# ---------------------------------------------------------------------------
# Single-task
# ---------------------------------------------------------------------------

def _process_single_task(
    artifacts_root: Path,
    dry_run: bool,
    overwrite: bool,
    totals: Dict[str, int],
) -> None:
    """Write one experiment_summary.json per task at artifacts/{task}/experiment_summary.json.

    Only the best run across all adapter subdirs for that task is written.
    The summary lives at the task root so it is immediately visible.
    """
    log.info("=== Processing single-task experiments ===")

    # Collect all known tasks by scanning for adapters/ directories.
    # Pattern: {task_dir}/adapters/{subdir}/runs_registry.json
    # p.parent       = {task_dir}/adapters/{subdir}
    # p.parent.parent = {task_dir}/adapters
    # p.parent.parent.parent = {task_dir}
    task_dirs = sorted({
        p.parent.parent.parent
        for p in artifacts_root.glob("*/adapters/*/runs_registry.json")
    })

    for task_dir in task_dirs:
        task = task_dir.name
        # Output goes at the task root — visible without digging into adapter subdirs.
        output_path = task_dir / "experiment_summary.json"

        if not _should_write(output_path, overwrite):
            totals["skipped"] += 1
            continue

        # Find the best run across all adapter subdirs for this task.
        best_run_entry = None
        best_adapter_dir = None
        best_metric_name = None
        best_metric_val = None
        greater_is_better = False

        for registry_path in sorted(task_dir.glob("adapters/*/runs_registry.json")):
            adapter_dir = registry_path.parent
            registry = _load_json(registry_path)
            if not registry:
                continue
            metric_for_ranking = registry.get("metric_for_ranking", "")
            gib = bool(registry.get("greater_is_better", False))
            for run_entry in registry.get("runs", []):
                if not run_entry.get("is_best", False):
                    continue
                raw_metrics = run_entry.get("metrics") or {}
                val = _to_float(raw_metrics.get(metric_for_ranking))
                if val is None:
                    stripped = metric_for_ranking.removeprefix(_EVAL_PREFIX)
                    val = _to_float(raw_metrics.get(stripped))
                # Select best across adapter subdirs using the ranking metric.
                if best_run_entry is None or val is not None and (
                    (gib and val > (best_metric_val or float("-inf"))) or
                    (not gib and val < (best_metric_val or float("inf")))
                ):
                    best_run_entry = run_entry
                    best_adapter_dir = adapter_dir
                    best_metric_name = metric_for_ranking
                    best_metric_val = val
                    greater_is_better = gib

        if best_run_entry is None or best_adapter_dir is None:
            log.debug("  No best run found for task %s — skipping", task)
            totals["skipped"] += 1
            continue

        run_id = best_run_entry.get("run_id")
        run_dir = best_adapter_dir / "runs" / run_id if run_id else None
        if run_dir is None or not run_dir.is_dir():
            totals["skipped"] += 1
            continue

        if dry_run:
            log.info("[DRY-RUN] Would write: %s", output_path)
            totals["written"] += 1
            continue

        try:
            _write_single_task_summary(
                run_dir=run_dir,
                run_id=run_id,
                run_entry=best_run_entry,
                task=task,
                adapter_dir=best_adapter_dir,
                metric_for_ranking=best_metric_name or "",
                is_best=True,
                is_latest=bool(best_run_entry.get("is_latest", False)),
                output_path=output_path,
            )
            log.info("  Written: %s", output_path)
            totals["written"] += 1
        except Exception as exc:
                log.warning("  FAILED %s: %s", output_path, exc)
                totals["failed"] += 1


def _write_single_task_summary(
    run_dir: Path,
    run_id: str,
    run_entry: Dict[str, Any],
    task: str,
    adapter_dir: Path,
    metric_for_ranking: str,
    is_best: Optional[bool],
    is_latest: Optional[bool],
    output_path: Path,
) -> None:
    timestamp = run_entry.get("timestamp")
    config_name = run_entry.get("config_hash")

    # Hyperparameters from registry entry.
    hp_raw = run_entry.get("hyperparameters_summary") or {}
    hyperparameters = build_hyperparameters(
        learning_rate=_to_float(hp_raw.get("learning_rate")),
        lora_r=_to_int(hp_raw.get("lora_r")),
        lora_alpha=_to_int(hp_raw.get("lora_alpha")),
        num_train_epochs=_to_int(hp_raw.get("num_train_epochs")),
        per_device_train_batch_size=_to_int(hp_raw.get("per_device_train_batch_size")),
    )

    # Results: validation from validation_metrics.json, test from canonical eval cache.
    # Priority for test metrics:
    #   1. artifacts/{task}/metrics/eval/test/best_{task}_adapter.json (most up-to-date)
    #   2. run_dir/test_metrics.json (from original training run)
    #   3. run_dir/test_metrics_*.json (timestamped fallback)
    results: Dict[str, Dict[str, Dict[str, Any]]] = {task: {}}
    source_files: List[str] = []

    val_path = run_dir / "validation_metrics.json"
    if val_path.exists():
        val_metrics = _load_json(val_path) or {}
        results[task]["validation"] = dict(val_metrics)
        source_files.append("validation_metrics.json")

    # Prefer the shared eval cache as test metrics (more complete, re-evaluated on full test set).
    # Navigate from run_dir up to task root: run_dir → runs/ → adapter_subdir/ → adapters/ → task/
    task_root = run_dir.parent.parent.parent.parent  # artifacts/{task}/
    canonical_test_path = task_root / "metrics" / "eval" / "test" / f"best_{task}_adapter.json"
    if canonical_test_path.exists():
        test_metrics_data = _load_json(canonical_test_path)
        if test_metrics_data:
            results[task]["test"] = dict(test_metrics_data)
            source_files.append(str(canonical_test_path.relative_to(task_root.parent)))
    else:
        # Fallback to run-level test metrics.
        test_metrics, test_file = _find_test_metrics(run_dir)
        if test_metrics:
            results[task]["test"] = dict(test_metrics)
            source_files.append(test_file)

    # Fallback: if no validation_metrics.json, use metrics from registry.
    if "validation" not in results[task]:
        registry_metrics = run_entry.get("metrics") or {}
        if registry_metrics:
            results[task]["validation"] = dict(registry_metrics)
            source_files.append("runs_registry.json")

    # Selection provenance — use the best available test metrics for ranking value.
    _test_data = results[task].get("test")
    ranked_by_split = "test" if _test_data else "validation"
    _val_data = results[task].get("validation") or {}
    ranked_metrics = _test_data if _test_data else _val_data
    # Resolve the metric value (strip eval_ prefix from the key name).
    sel_value = _resolve_metric_value(ranked_metrics, metric_for_ranking)

    selection = build_selection(
        policy="best_run_metric",
        metric_name=metric_for_ranking or None,
        metric_value=sel_value,
        ranked_by_split=ranked_by_split,
    )

    write_experiment_summary(
        output_path=output_path,
        experiment_type="single_task",
        run_id=run_id,
        timestamp=timestamp,
        config_name=config_name,
        source_tasks=[task],
        method=task,
        results=results,
        adapter_path=str(run_dir),
        is_best=is_best,
        is_latest=is_latest,
        hyperparameters=hyperparameters,
        merge_info=None,
        mtl_aggregate=None,
        selection=selection,
        source_files=source_files,
    )


# ---------------------------------------------------------------------------
# MTL
# ---------------------------------------------------------------------------

def _process_mtl(
    artifacts_root: Path,
    dry_run: bool,
    overwrite: bool,
    totals: Dict[str, int],
) -> None:
    """Write one experiment_summary.json per MTL combo at artifacts/mtl/{N}_task/{combo}/."""
    log.info("=== Processing MTL experiments ===")
    mtl_root = artifacts_root / "mtl"
    if not mtl_root.is_dir():
        log.info("  No mtl/ directory found — skipping.")
        return

    # Scan for combo dirs by looking for metrics/best/mtl_eval_history.jsonl.
    for jsonl_path in sorted(mtl_root.rglob("metrics/best/mtl_eval_history.jsonl")):
        # combo_dir is artifacts/mtl/{N}_task/{combo}/
        combo_dir = jsonl_path.parent.parent.parent
        output_path = combo_dir / "experiment_summary.json"

        if not _should_write(output_path, overwrite):
            totals["skipped"] += 1
            continue

        if dry_run:
            log.info("[DRY-RUN] Would write: %s", output_path)
            totals["written"] += 1
            continue

        # metrics/best/ directory contains all canonical files for the best checkpoint.
        best_dir = jsonl_path.parent
        try:
            _write_mtl_summary(
                run_dir=best_dir,
                jsonl_path=jsonl_path,
                output_path=output_path,
                combo_dir=combo_dir,
            )
            log.info("  Written: %s", output_path)
            totals["written"] += 1
        except Exception as exc:
            log.warning("  FAILED %s: %s", output_path, exc)
            totals["failed"] += 1


def _write_mtl_summary(
    run_dir: Path,
    jsonl_path: Path,
    output_path: Path,
    combo_dir: Optional[Path] = None,
) -> None:
    # Load config — check combo_dir first (canonical location), then run_dir.
    config = None
    if combo_dir is not None:
        config = _load_json(combo_dir / "mtl_config_resolved.json")
    if config is None:
        config = _load_json(run_dir / "mtl_config_resolved.json")
    criterion, task_list, seed, hp = _parse_mtl_config(config)

    # Load JSONL rows and select best checkpoint.
    rows = _load_jsonl_rows(jsonl_path)
    if not rows:
        raise ValueError(f"No rows in {jsonl_path}")

    sel_key = f"{_EVAL_PREFIX}mtl_{criterion}"
    best_row, best_value = _select_best_row(rows, sel_key)
    if best_row is None:
        raise ValueError(f"Could not select best row from {jsonl_path}")

    # Infer task list if not from config.
    if not task_list:
        task_list = _infer_tasks_from_row(best_row)
    if not task_list:
        raise ValueError(f"Could not determine task list from {jsonl_path}")

    source_tasks = sorted(task_list)
    best_step = best_row.get("step") or best_row.get("global_step")

    # MTL aggregate metrics.
    mtl_geom = _row_float(best_row, f"{_EVAL_PREFIX}mtl_geometric_mean_interference_delta")
    mtl_arith = _row_float(best_row, f"{_EVAL_PREFIX}mtl_arithmetic_mean_interference_delta")
    mtl_num = _row_int(best_row, f"{_EVAL_PREFIX}mtl_num_tasks_with_delta")

    mtl_aggregate = {
        "best_step": int(best_step) if best_step is not None else None,
        "geometric_mean_interference_delta": mtl_geom,
        "arithmetic_mean_interference_delta": mtl_arith,
        "num_tasks_with_delta": mtl_num,
    }

    # Build results from validation_metrics.json (JSONL is always validation).
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    source_files: List[str] = []

    val_path = run_dir / "validation_metrics.json"
    if val_path.exists():
        val_raw = _load_json(val_path) or {}
        for task in source_tasks:
            task_metrics = _extract_task_metrics_from_flat(val_raw, task)
            if task_metrics:
                results.setdefault(task, {})["validation"] = task_metrics
        source_files.append("validation_metrics.json")
    else:
        # Fallback: extract from the best JSONL row.
        for task in source_tasks:
            task_metrics = _extract_task_metrics_from_jsonl_row(best_row, task)
            if task_metrics:
                results.setdefault(task, {})["validation"] = task_metrics
        source_files.append("mtl_eval_history.jsonl")

    # Test metrics: find test_metrics_*.json or test_metrics.json.
    test_metrics, test_file = _find_test_metrics(run_dir)
    if test_metrics:
        # test_metrics is a flat dict like validation_metrics.json.
        for task in source_tasks:
            task_mets = _extract_task_metrics_from_flat(test_metrics, task)
            if task_mets:
                results.setdefault(task, {})["test"] = task_mets
        source_files.append(test_file)

    # is_best / is_latest: for the best checkpoint summary these are always True.
    is_best = True
    is_latest = (run_dir.name == "latest") if run_dir.name in ("best", "latest") else None

    # run_id: try to find the underlying run directory name from the adapter subdir.
    run_id = None
    if combo_dir is not None:
        for run_entry_path in sorted((combo_dir / "adapters").rglob("runs_registry.json") if (combo_dir / "adapters").is_dir() else []):
            reg = _load_json(run_entry_path)
            if reg:
                for entry in reg.get("runs", []):
                    if entry.get("is_best"):
                        run_id = entry.get("run_id")
                        break
            if run_id:
                break
    timestamp = _infer_timestamp_from_run_id(run_id) if run_id else None

    # Adapter path: point to the combo's adapters directory rather than metrics/best/.
    adapter_path = str(combo_dir) if combo_dir is not None else str(run_dir)

    selection = build_selection(
        policy="best_delta",
        metric_name=sel_key,
        metric_value=best_value,
        ranked_by_split="validation",
    )

    # Build hyperparameters dict.
    hp_dict = None
    if hp:
        from dataclasses import asdict
        hp_dict = {k: v for k, v in asdict(hp).items() if v is not None}
        if config:
            training_cfg = config.get("training", {}) or {}
            max_steps = _to_int(training_cfg.get("max_steps"))
            if max_steps is not None:
                hp_dict["max_steps"] = max_steps

    write_experiment_summary(
        output_path=output_path,
        experiment_type="mtl",
        run_id=run_id,
        timestamp=timestamp,
        config_name=None,
        source_tasks=source_tasks,
        method="mtl",
        results=results,
        adapter_path=adapter_path,
        is_best=is_best,
        is_latest=is_latest,
        hyperparameters=hp_dict,
        merge_info=None,
        mtl_aggregate=mtl_aggregate,
        selection=selection,
        source_files=source_files,
    )


def _extract_task_metrics_from_flat(flat: Dict[str, Any], task: str) -> Dict[str, Any]:
    """Extract task-specific metrics from a flat eval_ dict (validation_metrics.json)."""
    prefix = f"{_EVAL_PREFIX}{task}_"
    out: Dict[str, Any] = {}
    for k, v in flat.items():
        if k.startswith(prefix):
            subkey = k[len(prefix):]
            out[subkey] = v
    return out


def _extract_task_metrics_from_jsonl_row(row: Dict[str, Any], task: str) -> Dict[str, Any]:
    """Extract task-specific metrics from a JSONL best-row dict."""
    prefix = f"{_EVAL_PREFIX}{task}_"
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if k.startswith(prefix):
            subkey = k[len(prefix):]
            out[subkey] = v
    return out


def _parse_mtl_config(
    config: Optional[Dict[str, Any]],
) -> Tuple[str, List[str], Optional[int], Any]:
    """Mirror of MTLCollector._parse_mtl_config()."""
    if config is None:
        return _DEFAULT_MTL_CRITERION, [], None, None

    training_cfg = config.get("training", {}) or {}
    criterion = training_cfg.get("selection_criterion", _DEFAULT_MTL_CRITERION) or _DEFAULT_MTL_CRITERION
    criterion = str(criterion).strip().lower()

    tasks_raw = config.get("tasks", []) or []
    task_list: List[str] = []
    for entry in tasks_raw:
        if isinstance(entry, dict):
            name = entry.get("name")
            if name:
                task_list.append(str(name))
        elif isinstance(entry, str):
            task_list.append(entry)

    seed_raw = config.get("seed")
    seed = int(seed_raw) if seed_raw is not None else None

    lora_cfg = (config.get("model", {}) or {}).get("lora", {}) or {}
    sampling_cfg = training_cfg.get("sampling", {}) or {}

    try:
        from analysis.collect.schema import TrainingHyperparameters
        hp = TrainingHyperparameters(
            learning_rate=_to_float(training_cfg.get("learning_rate")),
            lora_r=_to_int(lora_cfg.get("r")),
            lora_alpha=_to_int(lora_cfg.get("alpha")),
            num_train_epochs=_to_int(training_cfg.get("num_train_epochs")),
            per_device_train_batch_size=_to_int(training_cfg.get("per_device_train_batch_size")),
            sampling_temperature=_to_float(sampling_cfg.get("temperature")),
            selection_criterion=criterion,
            num_tasks=len(task_list) if task_list else None,
        )
    except Exception:
        hp = None

    return criterion, task_list, seed, hp


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def _process_merge(
    artifacts_root: Path,
    dry_run: bool,
    overwrite: bool,
    totals: Dict[str, int],
) -> None:
    """Write per-sweep-point and top-level summaries for merge experiments.

    Per-sweep-point: eval/{split}/summary_lambda{value}.json (or summary.json if no lambda).
    Top-level best:  artifacts/merged/{method}/{combo}/experiment_summary.json
    """
    log.info("=== Processing merge experiments ===")
    merged_root = artifacts_root / "merged"
    if not merged_root.is_dir():
        log.info("  No merged/ directory found — skipping.")
        return

    _SKIP_DIRS = frozenset({"comparisons", "eval", ".git"})

    for method_dir in sorted(merged_root.iterdir()):
        if not method_dir.is_dir() or method_dir.name in _SKIP_DIRS:
            continue

        for task_combo_dir in sorted(method_dir.iterdir()):
            if not task_combo_dir.is_dir():
                continue
            eval_dir = task_combo_dir / "eval"
            if not eval_dir.is_dir():
                continue

            # Collect all (result_file, lambda_val, split) tuples for this combo.
            # We use this to determine the best lambda at the end.
            combo_sweep_points: List[Tuple[Path, Optional[float], str]] = []

            for split_dir in sorted(eval_dir.iterdir()):
                if not split_dir.is_dir():
                    continue
                split = split_dir.name

                for result_file in sorted(split_dir.glob("eval_results_*.json")):
                    # Skip subset-only eval files (contain "__") and generated summaries.
                    if "__" in result_file.stem or result_file.stem.endswith("_summary"):
                        continue
                    # Skip old-style *_summary.json files (naming legacy).
                    if result_file.stem.startswith("summary_") or result_file.stem == "summary":
                        continue

                    # Extract lambda from payload to build canonical filename.
                    payload = _load_json(result_file)
                    if payload is None:
                        continue
                    lambda_val = _to_float(payload.get("lambda"))
                    if lambda_val is not None:
                        # Format to at most 6 significant figures to keep filenames readable.
                        lambda_str = f"{lambda_val:.6g}"
                        summary_name = f"summary_lambda{lambda_str}.json"
                    else:
                        summary_name = "summary.json"

                    output_path = split_dir / summary_name
                    combo_sweep_points.append((result_file, lambda_val, split))

                    if not _should_write(output_path, overwrite):
                        totals["skipped"] += 1
                        continue

                    if dry_run:
                        log.info("[DRY-RUN] Would write: %s", output_path)
                        totals["written"] += 1
                        continue

                    try:
                        _write_merge_summary(
                            result_file=result_file,
                            method_name=method_dir.name,
                            task_combo_dir=task_combo_dir,
                            split=split,
                            output_path=output_path,
                        )
                        log.info("  Written: %s", output_path)
                        totals["written"] += 1
                    except Exception as exc:
                        log.warning("  FAILED %s: %s", output_path, exc)
                        totals["failed"] += 1

            # Write top-level experiment_summary.json for the best lambda.
            top_level_path = task_combo_dir / "experiment_summary.json"
            if combo_sweep_points and _should_write(top_level_path, overwrite):
                best_rf, best_lambda, best_split = _select_best_merge_point(
                    combo_sweep_points
                )
                if best_rf is not None:
                    if dry_run:
                        log.info("[DRY-RUN] Would write: %s", top_level_path)
                        totals["written"] += 1
                    else:
                        try:
                            _write_merge_summary(
                                result_file=best_rf,
                                method_name=method_dir.name,
                                task_combo_dir=task_combo_dir,
                                split=best_split,
                                output_path=top_level_path,
                                is_best=True,
                            )
                            log.info("  Written (best): %s", top_level_path)
                            totals["written"] += 1
                        except Exception as exc:
                            log.warning("  FAILED %s: %s", top_level_path, exc)
                            totals["failed"] += 1
            elif top_level_path.exists() and not overwrite:
                totals["skipped"] += 1


def _select_best_merge_point(
    sweep_points: List[Tuple[Path, Optional[float], str]],
) -> Tuple[Optional[Path], Optional[float], str]:
    """Select the best (result_file, lambda, split) by primary metric score.

    Preference order:
    1. "test" split over "validation"
    2. Highest geometric-mean interference delta (if present in payload)
    3. Fall back to last test point, or last point overall
    """
    test_points = [(rf, lv, sp) for rf, lv, sp in sweep_points if sp == "test"]
    candidates = test_points if test_points else sweep_points

    best_rf: Optional[Path] = None
    best_lv: Optional[float] = None
    best_sp: str = ""
    best_score: Optional[float] = None

    for rf, lv, sp in candidates:
        payload = _load_json(rf)
        if payload is None:
            continue
        # Try to get geometric mean interference delta as ranking signal.
        results_raw = payload.get("results") or {}
        deltas: List[float] = []
        for task_metrics in results_raw.values():
            if not isinstance(task_metrics, dict):
                continue
            meta = task_metrics.get("interference_delta_meta") or {}
            d = _to_float(meta.get("interference_delta"))
            if d is not None:
                deltas.append(d)
        score: Optional[float] = None
        if deltas:
            try:
                score = math.exp(sum(math.log(max(d, 1e-9)) for d in deltas) / len(deltas))
            except Exception:
                score = sum(deltas) / len(deltas)
        if best_rf is None or (score is not None and (best_score is None or score > best_score)):
            best_rf = rf
            best_lv = lv
            best_sp = sp
            best_score = score

    # Fallback to first candidate if no scores found.
    if best_rf is None and candidates:
        best_rf, best_lv, best_sp = candidates[0]
    return best_rf, best_lv, best_sp


def _write_merge_summary(
    result_file: Path,
    method_name: str,
    task_combo_dir: Path,
    split: str,
    output_path: Path,
    is_best: Optional[bool] = None,
) -> None:
    payload = _load_json(result_file)
    if payload is None:
        raise ValueError(f"Could not load {result_file}")

    split_from_file = payload.get("split", split)
    timestamp = payload.get("timestamp")
    merge_tag = payload.get("merge_tag")
    source_tasks: List[str] = sorted(list(payload.get("source_tasks") or []))
    merge_method = payload.get("merge_method") or method_name
    merge_lambda = _to_float(payload.get("lambda"))
    eval_tag = payload.get("eval_tag") or None

    results_raw: Dict[str, Any] = dict(payload.get("results") or {})

    # Build per-task per-split results dict.
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for task, task_metrics in results_raw.items():
        if not isinstance(task_metrics, dict):
            continue
        results[task] = {split_from_file: dict(task_metrics)}

    if not results:
        raise ValueError(f"No results in {result_file}")

    merge_info = {
        "lambda": merge_lambda,
        "merge_tag": merge_tag,
        "params": {},
    }

    selection = build_selection(
        policy="sweep_point",
        metric_name=None,
        metric_value=None,
        ranked_by_split=None,
    )

    # Try to infer run_id from path.
    run_id = None
    for part in result_file.parts:
        if part.startswith("run_"):
            run_id = part
            break

    # Adapter path: check if a run directory with saved weights exists.
    adapter_path = None
    if run_id:
        candidate = task_combo_dir / "runs" / run_id
        if candidate.is_dir():
            adapter_path = str(candidate)

    write_experiment_summary(
        output_path=output_path,
        experiment_type="merge",
        run_id=run_id,
        timestamp=timestamp,
        config_name=merge_tag,
        source_tasks=source_tasks,
        method=merge_method,
        results=results,
        adapter_path=adapter_path,
        is_best=is_best,
        is_latest=None,
        hyperparameters=None,
        merge_info=merge_info,
        mtl_aggregate=None,
        selection=selection,
        source_files=[result_file.name],
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _should_write(output_path: Path, overwrite: bool) -> bool:
    if output_path.exists() and not overwrite:
        return False
    return True


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r") as fh:
            return json.load(fh)
    except Exception as exc:
        log.debug("Could not load %s: %s", path, exc)
        return None


def _find_test_metrics(run_dir: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return (metrics_dict, filename) for the canonical test metrics file."""
    # Prefer the non-timestamped canonical file.
    canonical = run_dir / "test_metrics.json"
    if canonical.exists():
        data = _load_json(canonical)
        if data:
            return data, "test_metrics.json"

    # Fallback: most-recent timestamped file.
    candidates = sorted(run_dir.glob("test_metrics_*.json"), reverse=True)
    for candidate in candidates:
        data = _load_json(candidate)
        if data:
            return data, candidate.name

    return None, ""


def _get_registry_flags(run_dir: Path) -> Tuple[Optional[bool], Optional[bool]]:
    """Read is_best, is_latest from the parent registry."""
    run_id = run_dir.name
    # Navigate up: runs/ → adapter_subdir/ → runs_registry.json
    registry_path = run_dir.parent.parent / "runs_registry.json"
    if not registry_path.exists():
        return None, None
    registry = _load_json(registry_path)
    if not registry:
        return None, None
    for entry in registry.get("runs", []):
        if entry.get("run_id") == run_id:
            return bool(entry.get("is_best", False)), bool(entry.get("is_latest", False))
    return None, None


def _resolve_metric_value(metrics: Dict[str, Any], metric_name: str) -> Optional[float]:
    """Resolve a metric value, trying both raw and eval_-stripped forms."""
    if not metric_name or not metrics:
        return None
    v = metrics.get(metric_name)
    if v is None:
        # Try stripping eval_ prefix.
        stripped = metric_name.removeprefix(_EVAL_PREFIX)
        v = metrics.get(stripped)
    if v is None:
        # Try adding eval_ prefix.
        v = metrics.get(f"{_EVAL_PREFIX}{metric_name}")
    return _to_float(v)


def _load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    try:
        with path.open("r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        log.warning("Could not read %s: %s", path, exc)
    return rows


def _select_best_row(
    rows: List[Dict[str, Any]],
    sel_key: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    best_row = None
    best_value: Optional[float] = None
    for row in rows:
        raw = row.get(sel_key)
        if raw is None:
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v):
            continue
        if best_value is None or v > best_value:
            best_value = v
            best_row = row
    if best_row is None and rows:
        best_row = rows[-1]
    return best_row, best_value


def _infer_tasks_from_row(row: Dict[str, Any]) -> List[str]:
    """Mirror of MTLCollector._infer_tasks_from_row()."""
    known = set(TASK_METRICS.keys())
    sorted_tasks = sorted(known, key=len, reverse=True)
    tasks = set()
    for k in row.keys():
        if not k.startswith(_EVAL_PREFIX):
            continue
        without_prefix = k[len(_EVAL_PREFIX):]
        if without_prefix.startswith("mtl_"):
            continue
        for task in sorted_tasks:
            if without_prefix.startswith(task + "_"):
                tasks.add(task)
                break
    return sorted(tasks)


def _row_float(row: Dict[str, Any], key: str) -> Optional[float]:
    v = row.get(key)
    return _to_float(v)


def _row_int(row: Dict[str, Any], key: str) -> Optional[int]:
    v = row.get(key)
    return _to_int(v)


def _infer_timestamp_from_run_id(run_id: str) -> Optional[str]:
    """Try to parse ISO timestamp from run_YYYYMMDD_HHMMSS."""
    try:
        parts = run_id.split("_", 1)
        if len(parts) == 2:
            from datetime import datetime
            dt = datetime.strptime(parts[1], "%Y%m%d_%H%M%S")
            return dt.isoformat()
    except Exception:
        pass
    return None


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return None if not math.isfinite(f) else f
    except (TypeError, ValueError):
        return None


def _to_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
