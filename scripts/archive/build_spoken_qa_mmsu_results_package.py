#!/usr/bin/env python3
"""Build a complete Spoken QA (MMSU) analysis package for paper-ready reporting."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


BASE_ADAPTER_DIR = Path("artifacts/speech_qa/metrics/eval/test")
MERGED_SPEECH_QA_GLOB = "artifacts/merged/**/eval/test/per_task/speech_qa/*.json"
OUTPUT_DIR = Path("analysis/results/mmsu_spoken_qa")


REQUIRED_FIELDS = {"accuracy", "exact_match", "f1", "loss", "num_samples", "subtask_accuracy"}
EXPECTED_NUM_SAMPLES = 4924.0
EXPECTED_SUBTASKS = 47
EXPECTED_BASE_ADAPTER_COUNT = 8
EXPECTED_COMPARABLE_MERGED_COUNT = 4


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    source_type: str
    path: Path
    metrics: Dict[str, Any]

    @property
    def accuracy(self) -> float:
        return float(self.metrics["accuracy"])

    @property
    def exact_match(self) -> float:
        return float(self.metrics["exact_match"])

    @property
    def f1(self) -> float:
        return float(self.metrics["f1"])

    @property
    def loss(self) -> float:
        return float(self.metrics["loss"])

    @property
    def num_samples(self) -> float:
        return float(self.metrics["num_samples"])

    @property
    def subtasks(self) -> Mapping[str, float]:
        return self.metrics["subtask_accuracy"]


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(payload)!r}")
    return payload


def _classify_subtask(name: str) -> str:
    if name.endswith("_reasoning"):
        return "reasoning"
    if name.endswith("_perception"):
        return "perception"
    if name.endswith("_detection"):
        return "detection"
    if name.endswith("_recognition"):
        return "recognition"
    if name.endswith("_comparison"):
        return "comparison"
    if name.endswith("_prediction"):
        return "prediction"
    if name.endswith("_classification"):
        return "classification"
    if name.endswith("_counting"):
        return "counting"
    if name.endswith("_matching"):
        return "matching"
    if name.endswith("_identification"):
        return "identification"
    if name.endswith("_grounding"):
        return "grounding"
    if name.endswith("_summarization"):
        return "summarization"
    if name.endswith("_translation"):
        return "translation"
    if "question_answering" in name:
        return "qa"
    return "other"


def _is_comparable(payload: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            reasons.append(f"missing_field:{field}")
    if "num_samples" in payload and float(payload["num_samples"]) != EXPECTED_NUM_SAMPLES:
        reasons.append(f"num_samples_mismatch:{payload['num_samples']}")
    if "subtask_accuracy" in payload and isinstance(payload["subtask_accuracy"], Mapping):
        if len(payload["subtask_accuracy"]) != EXPECTED_SUBTASKS:
            reasons.append(f"subtask_count_mismatch:{len(payload['subtask_accuracy'])}")
    return (len(reasons) == 0), reasons


def _collect_base_and_adapters() -> List[RunRecord]:
    records: List[RunRecord] = []
    for path in sorted(BASE_ADAPTER_DIR.glob("*.json")):
        payload = _load_json(path)
        comparable, reasons = _is_comparable(payload)
        if not comparable:
            raise ValueError(f"Base/adapter file does not meet schema: {path} :: {reasons}")
        records.append(
            RunRecord(
                run_id=path.stem,
                source_type="base_adapter",
                path=path,
                metrics=payload,
            )
        )
    return records


def _collect_merged() -> Tuple[List[RunRecord], List[Dict[str, Any]]]:
    comparable: List[RunRecord] = []
    anomalies: List[Dict[str, Any]] = []
    for path in sorted(Path(".").glob(MERGED_SPEECH_QA_GLOB)):
        payload = _load_json(path)
        ok, reasons = _is_comparable(payload)
        if ok:
            comparable.append(
                RunRecord(
                    run_id=path.stem,
                    source_type="merged",
                    path=path,
                    metrics=payload,
                )
            )
        else:
            anomalies.append(
                {
                    "path": str(path),
                    "reasons": reasons,
                    "num_samples": payload.get("num_samples"),
                    "has_accuracy": "accuracy" in payload,
                    "has_subtask_accuracy": "subtask_accuracy" in payload,
                    "keys": sorted(payload.keys()),
                }
            )
    return comparable, anomalies


def _write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _to_markdown_table(fieldnames: List[str], rows: List[Mapping[str, Any]]) -> str:
    header = "| " + " | ".join(fieldnames) + " |"
    sep = "| " + " | ".join(["---"] * len(fieldnames)) + " |"
    body = ["| " + " | ".join(str(row.get(col, "")) for col in fieldnames) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def main() -> None:
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    base_adapter_records = _collect_base_and_adapters()
    merged_records, anomalies = _collect_merged()

    all_records = [*base_adapter_records, *merged_records]
    base_record = next((record for record in all_records if record.run_id == "base_model"), None)
    if base_record is None:
        raise ValueError("Could not find base_model record.")

    # Overall table.
    overall_rows: List[Dict[str, Any]] = []
    for rank, record in enumerate(sorted(all_records, key=lambda r: r.accuracy, reverse=True), start=1):
        overall_rows.append(
            {
                "rank": rank,
                "run_id": record.run_id,
                "source_type": record.source_type,
                "accuracy": f"{record.accuracy:.6f}",
                "exact_match": f"{record.exact_match:.6f}",
                "f1": f"{record.f1:.6f}",
                "loss": f"{record.loss:.6f}",
                "num_samples": f"{record.num_samples:.1f}",
                "delta_accuracy_vs_base": f"{record.accuracy - base_record.accuracy:+.6f}",
                "delta_em_vs_base": f"{record.exact_match - base_record.exact_match:+.6f}",
                "delta_f1_vs_base": f"{record.f1 - base_record.f1:+.6f}",
                "delta_loss_vs_base": f"{record.loss - base_record.loss:+.6f}",
                "path": str(record.path),
            }
        )
    _write_csv(
        output_dir / "overall_metrics_table.csv",
        [
            "rank",
            "run_id",
            "source_type",
            "accuracy",
            "exact_match",
            "f1",
            "loss",
            "num_samples",
            "delta_accuracy_vs_base",
            "delta_em_vs_base",
            "delta_f1_vs_base",
            "delta_loss_vs_base",
            "path",
        ],
        overall_rows,
    )

    # Per-subtask deltas.
    subtasks = sorted(base_record.subtasks.keys())
    per_subtask_rows: List[Dict[str, Any]] = []
    for subtask in subtasks:
        row: Dict[str, Any] = {
            "subtask": subtask,
            "category": _classify_subtask(subtask),
            "base_accuracy": f"{float(base_record.subtasks[subtask]):.6f}",
        }
        for record in all_records:
            row[f"{record.run_id}__delta_vs_base"] = f"{float(record.subtasks[subtask]) - float(base_record.subtasks[subtask]):+.6f}"
        per_subtask_rows.append(row)
    per_subtask_fieldnames = ["subtask", "category", "base_accuracy"] + [
        f"{record.run_id}__delta_vs_base" for record in all_records
    ]
    _write_csv(output_dir / "per_subtask_delta_vs_base.csv", per_subtask_fieldnames, per_subtask_rows)

    # Top gains/regressions per run.
    top_rows: List[Dict[str, Any]] = []
    for record in all_records:
        if record.run_id == "base_model":
            continue
        deltas = sorted(
            (
                {
                    "subtask": subtask,
                    "delta": float(record.subtasks[subtask]) - float(base_record.subtasks[subtask]),
                    "category": _classify_subtask(subtask),
                }
                for subtask in subtasks
            ),
            key=lambda x: x["delta"],
            reverse=True,
        )
        for idx, item in enumerate(deltas[:5], start=1):
            top_rows.append(
                {
                    "run_id": record.run_id,
                    "type": "top_gain",
                    "rank_within_type": idx,
                    "subtask": item["subtask"],
                    "category": item["category"],
                    "delta_vs_base": f"{item['delta']:+.6f}",
                }
            )
        for idx, item in enumerate(reversed(deltas[-5:]), start=1):
            top_rows.append(
                {
                    "run_id": record.run_id,
                    "type": "top_regression",
                    "rank_within_type": idx,
                    "subtask": item["subtask"],
                    "category": item["category"],
                    "delta_vs_base": f"{item['delta']:+.6f}",
                }
            )
    _write_csv(
        output_dir / "top_subtask_gains_regressions.csv",
        ["run_id", "type", "rank_within_type", "subtask", "category", "delta_vs_base"],
        top_rows,
    )

    # Category rollups.
    categories = [
        "reasoning",
        "perception",
        "detection",
        "recognition",
        "comparison",
        "prediction",
        "classification",
        "counting",
        "matching",
        "identification",
        "grounding",
        "summarization",
        "translation",
        "qa",
        "other",
    ]
    by_category: Dict[str, List[str]] = {category: [] for category in categories}
    for subtask in subtasks:
        by_category[_classify_subtask(subtask)].append(subtask)

    rollup_rows: List[Dict[str, Any]] = []
    for category in categories:
        row: Dict[str, Any] = {"category": category, "num_subtasks": len(by_category[category])}
        for record in all_records:
            if not by_category[category]:
                row[f"{record.run_id}__mean_delta_vs_base"] = "nan"
                continue
            mean_delta = sum(
                float(record.subtasks[subtask]) - float(base_record.subtasks[subtask])
                for subtask in by_category[category]
            ) / float(len(by_category[category]))
            row[f"{record.run_id}__mean_delta_vs_base"] = f"{mean_delta:+.6f}"
        rollup_rows.append(row)
    rollup_fieldnames = ["category", "num_subtasks"] + [f"{record.run_id}__mean_delta_vs_base" for record in all_records]
    _write_csv(output_dir / "category_rollup_mean_delta_vs_base.csv", rollup_fieldnames, rollup_rows)

    # Validation and anomalies.
    validation = {
        "checks": {
            "base_adapter_count_is_8": len(base_adapter_records) == EXPECTED_BASE_ADAPTER_COUNT,
            "merged_comparable_count_is_4": len(merged_records) == EXPECTED_COMPARABLE_MERGED_COUNT,
            "all_included_num_samples_4924": all(record.num_samples == EXPECTED_NUM_SAMPLES for record in all_records),
            "all_included_have_47_subtasks": all(len(record.subtasks) == EXPECTED_SUBTASKS for record in all_records),
        },
        "counts": {
            "base_adapter_count": len(base_adapter_records),
            "merged_comparable_count": len(merged_records),
            "included_total_count": len(all_records),
            "excluded_anomaly_count": len(anomalies),
        },
    }
    (output_dir / "validation_summary.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    (output_dir / "excluded_noncomparable_runs.json").write_text(json.dumps(anomalies, indent=2), encoding="utf-8")

    # Paper-ready markdown with concise findings and anomaly note.
    overall_md = _to_markdown_table(
        ["rank", "run_id", "source_type", "accuracy", "exact_match", "f1", "loss", "delta_accuracy_vs_base"],
        overall_rows,
    )
    top_by_run: Dict[str, Dict[str, List[Mapping[str, Any]]]] = {}
    for row in top_rows:
        top_by_run.setdefault(row["run_id"], {"top_gain": [], "top_regression": []})
        top_by_run[row["run_id"]][row["type"]].append(row)

    best_adapter = max((record for record in base_adapter_records if record.run_id != "base_model"), key=lambda r: r.accuracy)
    best_merged = max(merged_records, key=lambda r: r.accuracy)

    lines: List[str] = []
    lines.append("# Spoken QA and MMSU Evaluation Fixes and Results")
    lines.append("")
    lines.append("## Overall Metrics (Comparable Runs)")
    lines.append("")
    lines.append(overall_md)
    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    lines.append(
        f"- Best adapter vs base: `{best_adapter.run_id}` improves accuracy by "
        f"`{best_adapter.accuracy - base_record.accuracy:+.6f}` "
        f"({base_record.accuracy:.6f} -> {best_adapter.accuracy:.6f})."
    )
    lines.append(
        f"- Best merged comparable run vs base: `{best_merged.run_id}` improves accuracy by "
        f"`{best_merged.accuracy - base_record.accuracy:+.6f}` "
        f"({base_record.accuracy:.6f} -> {best_merged.accuracy:.6f})."
    )
    lines.append(
        f"- Best merged comparable run vs best adapter: accuracy delta "
        f"`{best_merged.accuracy - best_adapter.accuracy:+.6f}` "
        f"({best_adapter.accuracy:.6f} vs {best_merged.accuracy:.6f})."
    )
    lines.append("")
    lines.append("## Category-Level Trend Summary (Mean Δaccuracy vs base)")
    lines.append("")
    lines.append(
        _to_markdown_table(
            ["category", "num_subtasks", f"{best_adapter.run_id}__mean_delta_vs_base", f"{best_merged.run_id}__mean_delta_vs_base"],
            rollup_rows,
        )
    )
    lines.append("")
    lines.append("## Subtask Highlights (Top Gains and Regressions)")
    lines.append("")
    for run_id in [best_adapter.run_id, best_merged.run_id]:
        lines.append(f"### `{run_id}`")
        lines.append("")
        gains = sorted(top_by_run[run_id]["top_gain"], key=lambda x: int(x["rank_within_type"]))
        drops = sorted(top_by_run[run_id]["top_regression"], key=lambda x: int(x["rank_within_type"]))
        lines.append("Top gains:")
        for row in gains:
            lines.append(f"- `{row['subtask']}` ({row['category']}): `{row['delta_vs_base']}`")
        lines.append("Top regressions:")
        for row in drops:
            lines.append(f"- `{row['subtask']}` ({row['category']}): `{row['delta_vs_base']}`")
        lines.append("")
    lines.append("## Evaluation Fixes / Anomalies")
    lines.append("")
    if anomalies:
        lines.append(
            "- Excluded non-comparable merged run(s) from core MMSU comparison due to schema/sample mismatch."
        )
        for anomaly in anomalies:
            lines.append(
                f"- `{anomaly['path']}` excluded: reasons={anomaly['reasons']}, "
                f"num_samples={anomaly['num_samples']}, has_accuracy={anomaly['has_accuracy']}, "
                f"has_subtask_accuracy={anomaly['has_subtask_accuracy']}."
            )
    else:
        lines.append("- No excluded anomalies detected.")
    lines.append("")
    lines.append("## Validation Checklist")
    lines.append("")
    for key, value in validation["checks"].items():
        lines.append(f"- `{key}`: `{value}`")

    (output_dir / "spoken_qa_mmsu_results_package.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote analysis package to: {output_dir}")
    print(f"Included runs: {len(all_records)} (base/adapters={len(base_adapter_records)}, merged={len(merged_records)})")
    print(f"Excluded anomalies: {len(anomalies)}")


if __name__ == "__main__":
    main()
