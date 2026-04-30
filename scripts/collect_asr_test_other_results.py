"""Collect ASR test-clean vs test-other WER comparison rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _asr_metrics(payload: Optional[dict[str, Any]]) -> tuple[Optional[float], Optional[int]]:
    if not payload:
        return None, None
    if "wer" in payload:
        wer = payload.get("wer")
        samples = payload.get("num_samples_evaluated") or payload.get("num_samples_after_filter") or payload.get("num_samples")
        return (float(wer) if isinstance(wer, (int, float)) else None), (
            int(samples) if isinstance(samples, (int, float)) else None
        )
    results = payload.get("results")
    if isinstance(results, dict) and isinstance(results.get("asr"), dict):
        return _asr_metrics(results["asr"])
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        return _asr_metrics(metrics)
    for key in ("eval_asr_wer", "asr_wer"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value), None
    return None, None


def _row(
    *,
    family: str,
    name: str,
    clean_path: Path,
    other_path: Path,
) -> dict[str, str]:
    clean_wer, clean_samples = _asr_metrics(_load_json(clean_path))
    other_wer, other_samples = _asr_metrics(_load_json(other_path))
    status = "complete" if other_wer is not None else "missing_test_other"
    samples = other_samples if other_samples is not None else clean_samples
    return {
        "family": family,
        "name": name,
        "test_clean_wer": "" if clean_wer is None else f"{clean_wer:.12g}",
        "test_other_wer": "" if other_wer is None else f"{other_wer:.12g}",
        "sample_count": "" if samples is None else str(samples),
        "status": status,
        "test_clean_path": str(clean_path.relative_to(REPO_ROOT)),
        "test_other_path": str(other_path.relative_to(REPO_ROOT)),
    }


def collect_single_task_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    clean_dir = REPO_ROOT / "artifacts" / "asr" / "metrics" / "eval" / "test"
    other_dir = REPO_ROOT / "artifacts" / "asr" / "metrics" / "eval" / "test_other"
    if not clean_dir.exists():
        return rows
    for clean_path in sorted(clean_dir.glob("*.json")):
        if "__subset_" in clean_path.name:
            continue
        if clean_path.name != "base_model.json" and not clean_path.name.startswith("best_"):
            continue
        clean_wer, _ = _asr_metrics(_load_json(clean_path))
        if clean_wer is None:
            continue
        rows.append(
            _row(
                family="single_task",
                name=clean_path.stem,
                clean_path=clean_path,
                other_path=other_dir / clean_path.name,
            )
        )
    return rows


def collect_merged_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    merged_root = REPO_ROOT / "artifacts" / "merged"
    if not merged_root.exists():
        return rows
    for clean_path in sorted(merged_root.rglob("eval/test/per_task/asr/*metrics.json")):
        clean_wer, _ = _asr_metrics(_load_json(clean_path))
        if clean_wer is None:
            continue
        other_path = Path(str(clean_path).replace("/eval/test/", "/eval/test_other/"))
        method = clean_path.relative_to(merged_root).parts[0]
        rows.append(
            _row(
                family=f"merged/{method}",
                name=clean_path.stem,
                clean_path=clean_path,
                other_path=other_path,
            )
        )
    return rows


def collect_mtl_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    mtl_root = REPO_ROOT / "artifacts" / "mtl"
    if not mtl_root.exists():
        return rows
    for clean_path in sorted(mtl_root.rglob("metrics/*/test_metrics.json")):
        clean_wer, _ = _asr_metrics(_load_json(clean_path))
        if clean_wer is None:
            continue
        metrics_dir = clean_path.parent
        other_path = metrics_dir / "eval_results_test_other.json"
        rows.append(
            _row(
                family="mtl",
                name=str(metrics_dir.parent.parent.relative_to(mtl_root)),
                clean_path=clean_path,
                other_path=other_path,
            )
        )
    for clean_path in sorted(mtl_root.rglob("metrics/*/per_task/asr/eval_results_test.json")):
        clean_wer, _ = _asr_metrics(_load_json(clean_path))
        if clean_wer is None:
            continue
        other_path = clean_path.with_name("eval_results_test_other.json")
        rows.append(
            _row(
                family="mtl",
                name=str(clean_path.parents[3].relative_to(mtl_root)),
                clean_path=clean_path,
                other_path=other_path,
            )
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="artifacts/asr_test_other_comparison.csv",
        help="CSV path to write.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_single_task_rows() + collect_merged_rows() + collect_mtl_rows()
    output = Path(args.output)
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "family",
        "name",
        "test_clean_wer",
        "test_other_wer",
        "sample_count",
        "status",
        "test_clean_path",
        "test_other_path",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    missing = sum(1 for row in rows if row["status"] != "complete")
    print(f"Wrote {len(rows)} rows to {_display_path(output)} ({missing} missing test_other).")


if __name__ == "__main__":
    main()
