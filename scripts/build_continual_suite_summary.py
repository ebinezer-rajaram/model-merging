#!/usr/bin/env python3
"""Aggregate continual-suite outputs into thesis-ready CSVs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from continual_suite_lib import (
    build_stage_result_record,
    compute_stage_tables,
    load_eval_results_payload,
    write_csv_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CSV summaries for a continual suite root.")
    parser.add_argument("--suite-root", required=True, help="Suite root directory, e.g. artifacts/continual_suite/<suite_id>.")
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected mapping JSON payload at {path}")
    return dict(payload)


def _latest_sweep_summary(sweeps_dir: Path) -> Optional[Path]:
    candidates = sorted(sweeps_dir.glob("sweep_*.json"))
    return candidates[-1] if candidates else None


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _parse_iso_timestamp(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _sum_eval_runtime(results_payload: Mapping[str, Any]) -> Optional[float]:
    results = results_payload.get("results")
    if not isinstance(results, Mapping):
        return None
    values = []
    for metrics in results.values():
        if isinstance(metrics, Mapping):
            runtime = _safe_float(metrics.get("runtime"))
            if runtime is not None:
                values.append(runtime)
    if not values:
        return None
    return float(sum(values))


def _dir_max_mtime(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    max_mtime = path.stat().st_mtime
    if path.is_dir():
        for child in path.rglob("*"):
            try:
                child_mtime = child.stat().st_mtime
            except FileNotFoundError:
                continue
            if child_mtime > max_mtime:
                max_mtime = child_mtime
    return max_mtime


def _estimate_duration_from_start_and_path(start_dt: Optional[datetime], path: Path) -> Optional[float]:
    if start_dt is None:
        return None
    max_mtime = _dir_max_mtime(path)
    if max_mtime is None:
        return None
    return max(0.0, max_mtime - start_dt.timestamp())


def _extract_run_id_timestamp(run_id: str) -> Optional[datetime]:
    text = str(run_id).strip()
    if not text.startswith("run_"):
        return None
    suffix = text.replace("run_", "", 1)
    try:
        return datetime.strptime(suffix, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def _estimate_mtl_train_runtime_seconds(mtl: Mapping[str, Any]) -> Optional[float]:
    runtime = mtl.get("runtime")
    if isinstance(runtime, Mapping):
        value = _safe_float(runtime.get("train_runtime_seconds"))
        if value is not None:
            return value

    metrics_root = Path(str(mtl.get("expected_metrics_dir", "")))
    runs_dir = metrics_root / "runs"
    if not runs_dir.exists():
        return None
    run_dirs = sorted([path for path in runs_dir.glob("run_*") if path.is_dir()])
    if not run_dirs:
        return None
    run_dir = run_dirs[-1]
    start_dt = _extract_run_id_timestamp(run_dir.name)
    return _estimate_duration_from_start_and_path(start_dt, run_dir)


def _build_merge_runtime_info(manifest: Mapping[str, Any], summary_path: Path, summary: Mapping[str, Any], payload: Mapping[str, Any]) -> Dict[str, Any]:
    merge = manifest.get("merge")
    runtime_meta = dict(merge.get("runtime", {})) if isinstance(merge, Mapping) and isinstance(merge.get("runtime"), Mapping) else {}

    search_runtime = _safe_float(runtime_meta.get("search_runtime_seconds"))
    if search_runtime is None:
        started_at = _parse_iso_timestamp(summary.get("timestamp"))
        if started_at is not None:
            search_runtime = max(0.0, summary_path.stat().st_mtime - started_at.timestamp())

    posteval_runtime = _safe_float(runtime_meta.get("posteval_runtime_seconds"))
    if posteval_runtime is None:
        post_eval = summary.get("post_sweep_eval")
        if isinstance(post_eval, Mapping):
            posteval_runtime = _sum_eval_runtime(post_eval)

    selection_eval_runtime = None
    best_index = summary.get("best_index")
    runs = summary.get("runs") or []
    if isinstance(best_index, int) and 0 <= best_index < len(runs):
        run_payload = {"results": dict(runs[best_index].get("results", {}))}
        selection_eval_runtime = _sum_eval_runtime(run_payload)
    if selection_eval_runtime is None:
        selection_eval_runtime = _sum_eval_runtime(payload)

    stage_total = _safe_float(runtime_meta.get("stage_total_runtime_seconds"))
    if stage_total is None:
        stage_total = None
        if search_runtime is not None or posteval_runtime is not None:
            stage_total = float((search_runtime or 0.0) + (posteval_runtime or 0.0))

    return {
        "merge_search_runtime_seconds": search_runtime,
        "merge_selection_eval_runtime_seconds": selection_eval_runtime,
        "merge_posteval_runtime_seconds": posteval_runtime,
        "merge_stage_total_runtime_seconds": stage_total,
        "stage_total_runtime_seconds": stage_total,
    }


def _build_mtl_runtime_info(manifest: Mapping[str, Any], mtl: Mapping[str, Any], checkpoint_view: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    runtime_meta = dict(mtl.get("runtime", {})) if isinstance(mtl.get("runtime"), Mapping) else {}
    train_runtime = _estimate_mtl_train_runtime_seconds(mtl)
    if train_runtime is None:
        train_runtime = _safe_float(runtime_meta.get("train_runtime_seconds"))

    added_eval_runtime = None
    balanced_eval_runtime = None
    if checkpoint_view == "added_task_best":
        added_eval_runtime = _sum_eval_runtime(payload)
        balanced_info = mtl.get("balanced_best")
        if isinstance(balanced_info, Mapping):
            balanced_eval_path = Path(str(balanced_info.get("metrics_dir", ""))) / f"eval_results_{manifest.get('report_split', 'test')}.json"
            if balanced_eval_path.exists():
                balanced_eval_runtime = _sum_eval_runtime(load_eval_results_payload(balanced_eval_path))
    elif checkpoint_view == "balanced_best":
        balanced_eval_runtime = _safe_float(
            ((mtl.get("balanced_best") or {}).get("runtime") or {}).get("eval_runtime_seconds")
            if isinstance(mtl.get("balanced_best"), Mapping)
            else None
        )
        if balanced_eval_runtime is None:
            balanced_eval_runtime = _sum_eval_runtime(payload)
        best_eval_path = Path(str(mtl.get("expected_best_metrics_dir", ""))) / f"eval_results_{manifest.get('report_split', 'test')}.json"
        if best_eval_path.exists():
            added_eval_runtime = _sum_eval_runtime(load_eval_results_payload(best_eval_path))

    stage_total = None
    if train_runtime is not None:
        stage_total = train_runtime + (balanced_eval_runtime if checkpoint_view == "balanced_best" else (added_eval_runtime or 0.0))

    return {
        "mtl_train_runtime_seconds": train_runtime,
        "mtl_added_task_best_eval_runtime_seconds": added_eval_runtime,
        "mtl_balanced_best_eval_runtime_seconds": balanced_eval_runtime,
        "mtl_stage_total_runtime_seconds": stage_total,
        "stage_total_runtime_seconds": stage_total,
    }


def _load_merge_record(manifest: Mapping[str, Any]) -> Optional[Any]:
    merge = manifest.get("merge")
    if not isinstance(merge, Mapping):
        return None
    sweeps_dir = Path(str(merge.get("sweeps_dir", "")))
    summary_path = _latest_sweep_summary(sweeps_dir)
    if summary_path is None or not summary_path.exists():
        return None

    summary = _load_json(summary_path)
    payload: Optional[Mapping[str, Any]] = None
    split = str(manifest.get("report_split", "test"))
    post_eval = summary.get("post_sweep_eval")
    if isinstance(post_eval, Mapping) and post_eval.get("enabled") and isinstance(post_eval.get("results"), Mapping):
        payload = {
            "results": dict(post_eval["results"]),
        }
        split = str(post_eval.get("split", split))
    else:
        best_index = summary.get("best_index")
        runs = summary.get("runs") or []
        if isinstance(best_index, int) and 0 <= best_index < len(runs):
            payload = {
                "results": dict(runs[best_index].get("results", {})),
            }
            split = str(summary.get("split", manifest.get("selection_split", "validation")))
    if payload is None:
        return None
    runtime_info = _build_merge_runtime_info(manifest, summary_path, summary, payload)

    summary_method = str(summary.get("method", "continual"))
    method_label = "continual_supermerge" if summary_method == "continual_supermerge" else "continual_merge"
    return build_stage_result_record(
        suite_id=str(manifest["suite_id"]),
        path_id=str(manifest["path_id"]),
        stage_index=int(manifest["stage_index"]),
        stage_name=str(manifest["stage_name"]),
        method=method_label,
        checkpoint_view="best",
        split=split,
        seen_tasks=list(manifest.get("seen_tasks", [])),
        prior_tasks=list(manifest.get("prior_tasks", [])),
        added_task=manifest.get("added_task"),
        eval_only_tasks=list(manifest.get("eval_only_tasks", [])),
        report_tasks=list(manifest.get("report_tasks", [])),
        results_payload=payload,
        source_path=summary_path,
        runtime_info=runtime_info,
    )


def _load_mtl_eval_record(manifest: Mapping[str, Any], *, checkpoint_view: str, metrics_dir: Path) -> Optional[Any]:
    split = str(manifest.get("report_split", "test"))
    eval_path = metrics_dir / f"eval_results_{split}.json"
    if not eval_path.exists():
        return None
    payload = load_eval_results_payload(eval_path)
    mtl = manifest.get("mtl")
    runtime_info = _build_mtl_runtime_info(
        manifest,
        dict(mtl) if isinstance(mtl, Mapping) else {},
        checkpoint_view,
        payload,
    )
    return build_stage_result_record(
        suite_id=str(manifest["suite_id"]),
        path_id=str(manifest["path_id"]),
        stage_index=int(manifest["stage_index"]),
        stage_name=str(manifest["stage_name"]),
        method="continual_mtl",
        checkpoint_view=checkpoint_view,
        split=split,
        seen_tasks=list(manifest.get("seen_tasks", [])),
        prior_tasks=list(manifest.get("prior_tasks", [])),
        added_task=manifest.get("added_task"),
        eval_only_tasks=list(manifest.get("eval_only_tasks", [])),
        report_tasks=list(manifest.get("report_tasks", [])),
        results_payload=payload,
        source_path=eval_path,
        runtime_info=runtime_info,
    )


def _load_mtl_records(manifest: Mapping[str, Any]) -> List[Any]:
    mtl = manifest.get("mtl")
    if not isinstance(mtl, Mapping):
        return []
    records = []
    best_metrics_dir = Path(str(mtl.get("expected_best_metrics_dir", "")))
    added_task_best = _load_mtl_eval_record(manifest, checkpoint_view="added_task_best", metrics_dir=best_metrics_dir)
    if added_task_best is not None:
        records.append(added_task_best)

    balanced_info = mtl.get("balanced_best")
    if isinstance(balanced_info, Mapping):
        balanced_metrics_dir = Path(str(balanced_info.get("metrics_dir", "")))
        balanced = _load_mtl_eval_record(manifest, checkpoint_view="balanced_best", metrics_dir=balanced_metrics_dir)
        if balanced is not None:
            records.append(balanced)

    fallback_balanced_dir = Path(str(mtl.get("expected_metrics_dir", ""))) / "balanced_best"
    if balanced_info is None and fallback_balanced_dir.exists():
        balanced = _load_mtl_eval_record(manifest, checkpoint_view="balanced_best", metrics_dir=fallback_balanced_dir)
        if balanced is not None:
            records.append(balanced)
    return records


def _collect_manifests(suite_root: Path) -> List[Path]:
    return sorted(suite_root.glob("*/stage_*/stage_manifest.json"))


def main() -> None:
    args = parse_args()
    suite_root = Path(args.suite_root).expanduser().resolve()
    manifests = _collect_manifests(suite_root)
    if not manifests:
        raise SystemExit(f"No stage manifests found under {suite_root}")

    records = []
    for manifest_path in manifests:
        manifest = _load_json(manifest_path)
        merge_record = _load_merge_record(manifest)
        if merge_record is not None:
            records.append(merge_record)
        records.extend(_load_mtl_records(manifest))

    if not records:
        raise SystemExit(f"No evaluable suite results found under {suite_root}")

    tables = compute_stage_tables(records)
    write_csv_rows(suite_root / "continual_stage_metrics.csv", tables["stage_rows"])
    write_csv_rows(suite_root / "continual_growth_curves.csv", tables["growth_rows"])
    write_csv_rows(suite_root / "continual_forgetting_curves.csv", tables["forgetting_rows"])
    write_csv_rows(suite_root / "continual_final_comparison.csv", tables["final_rows"])

    print(f"Wrote summaries under {suite_root}")


if __name__ == "__main__":
    main()
