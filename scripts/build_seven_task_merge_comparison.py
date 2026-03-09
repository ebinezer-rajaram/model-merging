#!/usr/bin/env python3
import csv
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple


ROOT = Path(".")
DATE_STAMP = datetime.now().strftime("%Y%m%d")

SOURCE_TASKS = ["emotion", "intent", "kws", "langid", "speaker_ver", "asr", "vocalsound"]

MERGED_SOURCES = {
    "uniform_delta": ROOT
    / "artifacts/merged/uniform_delta/asr_emotion_intent_kws_langid_speaker_ver_vocalsound/runs/run_20260224_105145/eval_results_test.json",
    "uniform_scalar_delta": ROOT
    / "artifacts/merged/uniform_scalar_delta/asr_emotion_intent_kws_langid_speaker_ver_vocalsound/runs/run_20260224_163137/eval_results_test.json",
    "weighted_delta_n_main": ROOT
    / "artifacts/merged/weighted_delta_n/asr_emotion_intent_kws_langid_speaker_ver_vocalsound/runs/run_supermerge_layer_wise_20260304_065318/eval_results_test.json",
    "weighted_delta_n_rerun": ROOT
    / "artifacts/merged/weighted_delta_n/asr_emotion_intent_kws_langid_speaker_ver_vocalsound/runs/run_20260304_140418/eval_results_test.json",
}

CONSOLIDATED_REFERENCE = (
    ROOT / "artifacts/merged/comparisons/seven_task_plus_speechqa_merge_consolidated_20260304.csv"
)

OUT_DIR = ROOT / "artifacts/merged/comparisons"
OUT_MAIN_CSV = OUT_DIR / f"seven_task_merge_with_baselines_and_interference_{DATE_STAMP}.csv"
OUT_DETAIL_CSV = OUT_DIR / f"seven_task_merge_interference_detail_{DATE_STAMP}.csv"
OUT_MD = OUT_DIR / f"seven_task_merge_with_baselines_and_interference_{DATE_STAMP}.md"


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _assert_exists(paths: List[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def _fmt(value: Optional[float], precision: int = 10) -> str:
    if value is None:
        return ""
    return f"{value:.{precision}f}"


def _extract_row_from_results(
    row_name: str, results: Dict[str, Dict], notes: str
) -> Tuple[Dict[str, Optional[float]], List[Dict[str, Optional[float]]]]:
    row = {
        "method": row_name,
        "asr_wer": results["asr"].get("wer"),
        "emotion_macro_f1": results["emotion"].get("macro_f1"),
        "intent_acc": results["intent"].get("accuracy"),
        "kws_macro_f1": results["kws"].get("macro_f1"),
        "langid_acc": results["langid"].get("accuracy"),
        "speaker_ver_acc": results["speaker_ver"].get("accuracy"),
        "vocalsound_acc": results["vocalsound"].get("accuracy"),
        "spoken_qa_acc": results.get("speech_qa", {}).get("accuracy"),
        "min_interference_delta": None,
        "mean_interference_delta": None,
        "notes": notes,
    }

    detail_rows = []
    deltas: List[float] = []
    for task in SOURCE_TASKS:
        value = results[task].get("interference_delta")
        detail_rows.append({"method": row_name, "task": task, "interference_delta": value})
        if isinstance(value, (int, float)):
            deltas.append(float(value))

    if deltas:
        row["min_interference_delta"] = min(deltas)
        row["mean_interference_delta"] = mean(deltas)

    return row, detail_rows


def _build_baseline_rows() -> List[Dict[str, Optional[float]]]:
    task_metrics_dir = ROOT / "artifacts"
    base = {
        "method": "base_model",
        "asr_wer": _load_json(task_metrics_dir / "asr/metrics/eval/test/base_model.json").get("wer"),
        "emotion_macro_f1": _load_json(task_metrics_dir / "emotion/metrics/eval/test/base_model.json").get(
            "macro_f1"
        ),
        "intent_acc": _load_json(task_metrics_dir / "intent/metrics/eval/test/base_model.json").get(
            "accuracy"
        ),
        "kws_macro_f1": _load_json(task_metrics_dir / "kws/metrics/eval/test/base_model.json").get("macro_f1"),
        "langid_acc": _load_json(task_metrics_dir / "langid/metrics/eval/test/base_model.json").get(
            "accuracy"
        ),
        "speaker_ver_acc": _load_json(task_metrics_dir / "speaker_ver/metrics/eval/test/base_model.json").get(
            "accuracy"
        ),
        "vocalsound_acc": _load_json(task_metrics_dir / "vocalsound/metrics/eval/test/base_model.json").get(
            "accuracy"
        ),
        "spoken_qa_acc": _load_json(task_metrics_dir / "speech_qa/metrics/eval/test/base_model.json").get(
            "accuracy"
        ),
        "min_interference_delta": None,
        "mean_interference_delta": None,
        "notes": "Per-task base_model.json files; no interference delta for baseline.",
    }

    best_single = {
        "method": "best_single_task",
        "asr_wer": _load_json(task_metrics_dir / "asr/metrics/eval/test/best_asr_adapter.json").get("wer"),
        "emotion_macro_f1": _load_json(task_metrics_dir / "emotion/metrics/eval/test/best_emotion_adapter.json").get(
            "macro_f1"
        ),
        "intent_acc": _load_json(task_metrics_dir / "intent/metrics/eval/test/best_intent_adapter.json").get(
            "accuracy"
        ),
        "kws_macro_f1": _load_json(task_metrics_dir / "kws/metrics/eval/test/best_kws_adapter.json").get(
            "macro_f1"
        ),
        "langid_acc": _load_json(task_metrics_dir / "langid/metrics/eval/test/best_langid_adapter.json").get(
            "accuracy"
        ),
        "speaker_ver_acc": _load_json(
            task_metrics_dir / "speaker_ver/metrics/eval/test/best_speaker_ver_adapter.json"
        ).get("accuracy"),
        "vocalsound_acc": _load_json(
            task_metrics_dir / "vocalsound/metrics/eval/test/best_vocalsound_adapter.json"
        ).get("accuracy"),
        "spoken_qa_acc": None,
        "min_interference_delta": None,
        "mean_interference_delta": None,
        "notes": "Per-task best_{task}_adapter.json; Spoken QA best-single intentionally omitted.",
    }
    return [base, best_single]


def _stitch_weighted_delta_results(main_payload: Dict, rerun_payload: Dict) -> Tuple[Dict, List[str]]:
    stitched = deepcopy(main_payload.get("results", {}))
    replaced = []
    rerun_results = rerun_payload.get("results", {})
    for task, rerun_task_metrics in rerun_results.items():
        if task not in stitched or "error" in stitched.get(task, {}):
            stitched[task] = rerun_task_metrics
            replaced.append(task)
    return stitched, replaced


def _validate_row_coverage(main_rows: List[Dict[str, Optional[float]]]) -> List[str]:
    issues = []
    required_cols = [
        "asr_wer",
        "emotion_macro_f1",
        "intent_acc",
        "kws_macro_f1",
        "langid_acc",
        "speaker_ver_acc",
        "vocalsound_acc",
    ]
    for row in main_rows:
        for col in required_cols:
            if row[col] is None:
                issues.append(f"{row['method']} missing required value: {col}")
        if row["method"] in {"base_model", "uniform_scalar_delta", "weighted_delta_n_stitched"}:
            if row["spoken_qa_acc"] is None:
                issues.append(f"{row['method']} missing spoken_qa_acc")
        if row["method"] == "best_single_task" and row["spoken_qa_acc"] is not None:
            issues.append("best_single_task spoken_qa_acc should be blank")
    return issues


def _validate_interference_rows(
    detail_rows: List[Dict[str, Optional[float]]], merged_methods: List[str]
) -> List[str]:
    issues = []
    grouped: Dict[str, List[Dict[str, Optional[float]]]] = {}
    for row in detail_rows:
        grouped.setdefault(row["method"], []).append(row)
    for method in merged_methods:
        vals = grouped.get(method, [])
        if len(vals) != 7:
            issues.append(f"{method} interference row count != 7 (got {len(vals)})")
        non_numeric = [r for r in vals if not isinstance(r.get("interference_delta"), (int, float))]
        if non_numeric:
            issues.append(f"{method} has non-numeric interference deltas")
    return issues


def _cross_check_against_consolidated(main_rows: List[Dict[str, Optional[float]]]) -> List[str]:
    if not CONSOLIDATED_REFERENCE.exists():
        return ["Consolidated reference CSV not found; skipped cross-check."]

    by_method = {row["method"]: row for row in main_rows}
    consolidated_by_method = {}
    with CONSOLIDATED_REFERENCE.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            consolidated_by_method[row["method"]] = row

    checks = []
    mapping = [
        ("uniform_delta", "uniform_delta"),
        ("uniform_scalar_delta", "uniform_scalar_delta_post_sweep_eval"),
        ("weighted_delta_n_stitched", "weighted_delta_n_supermerge_per_task"),
    ]
    for local_name, ref_name in mapping:
        local = by_method.get(local_name)
        ref = consolidated_by_method.get(ref_name)
        if not local or not ref:
            checks.append(f"Missing method for cross-check: local={local_name}, ref={ref_name}")
            continue

        # Overlapping fields with matching definitions in the consolidated file.
        comparisons = [
            ("asr_wer", "asr_wer"),
            ("emotion_macro_f1", "emotion_macro_f1"),
            ("kws_macro_f1", "kws_macro_f1"),
            ("min_interference_delta", "min_interference_delta"),
            ("mean_interference_delta", "mean_interference_delta"),
        ]
        for local_key, ref_key in comparisons:
            ref_value = ref.get(ref_key)
            if ref_value in (None, ""):
                checks.append(f"{local_name}: reference {ref_key} empty, skipped")
                continue
            local_value = local.get(local_key)
            if local_value is None:
                checks.append(f"{local_name}: local {local_key} missing")
                continue
            diff = abs(float(local_value) - float(ref_value))
            if diff > 1e-9:
                checks.append(
                    f"{local_name}: mismatch {local_key} local={local_value:.12f} ref={float(ref_value):.12f}"
                )
    if not checks:
        return ["Cross-check passed for overlapping fields against consolidated CSV."]
    return checks


def _write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: List[Dict], columns: List[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in rows:
        values = []
        for col in columns:
            val = row.get(col)
            if isinstance(val, float):
                values.append(_fmt(val))
            elif val is None:
                values.append("")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    required = [
        MERGED_SOURCES["uniform_delta"],
        MERGED_SOURCES["uniform_scalar_delta"],
        MERGED_SOURCES["weighted_delta_n_main"],
        MERGED_SOURCES["weighted_delta_n_rerun"],
        CONSOLIDATED_REFERENCE,
        ROOT / "artifacts/asr/metrics/eval/test/base_model.json",
        ROOT / "artifacts/asr/metrics/eval/test/best_asr_adapter.json",
        ROOT / "artifacts/emotion/metrics/eval/test/base_model.json",
        ROOT / "artifacts/emotion/metrics/eval/test/best_emotion_adapter.json",
        ROOT / "artifacts/intent/metrics/eval/test/base_model.json",
        ROOT / "artifacts/intent/metrics/eval/test/best_intent_adapter.json",
        ROOT / "artifacts/kws/metrics/eval/test/base_model.json",
        ROOT / "artifacts/kws/metrics/eval/test/best_kws_adapter.json",
        ROOT / "artifacts/langid/metrics/eval/test/base_model.json",
        ROOT / "artifacts/langid/metrics/eval/test/best_langid_adapter.json",
        ROOT / "artifacts/speaker_ver/metrics/eval/test/base_model.json",
        ROOT / "artifacts/speaker_ver/metrics/eval/test/best_speaker_ver_adapter.json",
        ROOT / "artifacts/vocalsound/metrics/eval/test/base_model.json",
        ROOT / "artifacts/vocalsound/metrics/eval/test/best_vocalsound_adapter.json",
        ROOT / "artifacts/speech_qa/metrics/eval/test/base_model.json",
    ]
    _assert_exists(required)

    main_rows = _build_baseline_rows()
    detail_rows: List[Dict[str, Optional[float]]] = []

    uniform_delta_payload = _load_json(MERGED_SOURCES["uniform_delta"])
    uniform_delta_has_speechqa = "speech_qa" in uniform_delta_payload.get("results", {})
    uniform_delta_notes = f"Source: {MERGED_SOURCES['uniform_delta']}"
    if not uniform_delta_has_speechqa:
        uniform_delta_notes += "; speech_qa not evaluated in this run (N/A)."
    uniform_delta_row, uniform_delta_details = _extract_row_from_results(
        "uniform_delta",
        uniform_delta_payload["results"],
        uniform_delta_notes,
    )
    main_rows.append(uniform_delta_row)
    detail_rows.extend(uniform_delta_details)

    uniform_scalar_payload = _load_json(MERGED_SOURCES["uniform_scalar_delta"])
    uniform_scalar_row, uniform_scalar_details = _extract_row_from_results(
        "uniform_scalar_delta",
        uniform_scalar_payload["results"],
        f"Source: {MERGED_SOURCES['uniform_scalar_delta']}",
    )
    main_rows.append(uniform_scalar_row)
    detail_rows.extend(uniform_scalar_details)

    weighted_main_payload = _load_json(MERGED_SOURCES["weighted_delta_n_main"])
    weighted_rerun_payload = _load_json(MERGED_SOURCES["weighted_delta_n_rerun"])
    weighted_stitched_results, replaced_tasks = _stitch_weighted_delta_results(
        weighted_main_payload, weighted_rerun_payload
    )
    weighted_notes = (
        f"Stitched: base={MERGED_SOURCES['weighted_delta_n_main']}; "
        f"rerun_fill={MERGED_SOURCES['weighted_delta_n_rerun']}; "
        f"replaced_tasks={','.join(replaced_tasks) if replaced_tasks else 'none'}."
    )
    weighted_row, weighted_details = _extract_row_from_results(
        "weighted_delta_n_stitched", weighted_stitched_results, weighted_notes
    )
    main_rows.append(weighted_row)
    detail_rows.extend(weighted_details)

    fieldnames_main = [
        "method",
        "asr_wer",
        "emotion_macro_f1",
        "intent_acc",
        "kws_macro_f1",
        "langid_acc",
        "speaker_ver_acc",
        "vocalsound_acc",
        "spoken_qa_acc",
        "min_interference_delta",
        "mean_interference_delta",
        "notes",
    ]
    _write_csv(OUT_MAIN_CSV, main_rows, fieldnames_main)

    fieldnames_detail = ["method", "task", "interference_delta"]
    _write_csv(OUT_DETAIL_CSV, detail_rows, fieldnames_detail)

    coverage_issues = _validate_row_coverage(main_rows)
    interference_issues = _validate_interference_rows(
        detail_rows,
        merged_methods=["uniform_delta", "uniform_scalar_delta", "weighted_delta_n_stitched"],
    )
    cross_check_lines = _cross_check_against_consolidated(main_rows)

    validation_lines = []
    if not coverage_issues:
        validation_lines.append("Row coverage validation: PASS")
    else:
        validation_lines.append("Row coverage validation: FAIL")
        validation_lines.extend(coverage_issues)

    if not interference_issues:
        validation_lines.append("Interference validation: PASS")
    else:
        validation_lines.append("Interference validation: FAIL")
        validation_lines.extend(interference_issues)

    validation_lines.extend(cross_check_lines)

    with OUT_MD.open("w", encoding="utf-8") as f:
        f.write("# Seven-Task Merge Comparison with Baselines and Interference\n\n")
        f.write("## Main Comparison Table\n\n")
        f.write(_markdown_table(main_rows, fieldnames_main))
        f.write("\n\n## Interference Detail Table\n\n")
        f.write(_markdown_table(detail_rows, fieldnames_detail))
        f.write("\n\n## Validation\n\n")
        for line in validation_lines:
            f.write(f"- {line}\n")

    print(f"Wrote: {OUT_MAIN_CSV}")
    print(f"Wrote: {OUT_DETAIL_CSV}")
    print(f"Wrote: {OUT_MD}")
    for line in validation_lines:
        print(line)


if __name__ == "__main__":
    main()
