#!/usr/bin/env python3
"""Build cross-language ST analysis package for base/single-task/merged adapters."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ST_ROOT = Path("artifacts/st")
MERGED_PER_TASK_ST_GLOBS = [
    Path("artifacts/merged/uniform_delta/asr_emotion_intent_kws_langid_speaker_ver/eval/test/per_task/st"),
    Path("artifacts/merged/uniform_scalar_delta/asr_emotion_intent_kws_langid_speaker_ver/eval/test/per_task/st"),
    Path("artifacts/merged/weighted_delta_n/asr_emotion_intent_kws_langid_speaker_ver/eval/test/per_task/st"),
]
OUTPUT_DIR = Path("analysis/results/st_multilingual")

EXPECTED_LANGUAGES = ["en_ar", "en_de", "en_zh-CN"]
EXPECTED_BASE_SINGLE_RUNS_PER_LANG = 7
EXPECTED_MERGED_METHOD_COUNT = 3
SAMPLE_TARGET = 5000.0
SAMPLE_TOLERANCE = 10.0


@dataclass(frozen=True)
class Row:
    language: str
    run_family: str
    run_id: str
    merge_method: str
    bleu: float
    chrf: float
    loss: float
    runtime: float
    samples_per_second: float
    est_num_samples: float
    path: str


@dataclass(frozen=True)
class DeltaRow:
    language: str
    run_family: str
    run_id: str
    merge_method: str
    bleu: float
    chrf: float
    loss: float
    delta_bleu_vs_base: float
    delta_chrf_vs_base: float
    delta_loss_vs_base: float
    runtime: float
    samples_per_second: float
    est_num_samples: float
    path: str


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(payload)!r}")
    return payload


def _require_metrics(path: Path, payload: Mapping[str, Any]) -> None:
    for key in ("bleu", "chrf", "loss", "runtime", "samples_per_second"):
        if key not in payload:
            raise ValueError(f"Missing required field '{key}' in {path}")
        value = payload[key]
        if value is None:
            raise ValueError(f"Null value for '{key}' in {path}")
        if isinstance(value, float) and math.isnan(value):
            raise ValueError(f"NaN value for '{key}' in {path}")


def _fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def _fmt_signed(value: float, digits: int = 6) -> str:
    return f"{value:+.{digits}f}"


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _to_markdown_table(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    header = "| " + " | ".join(fieldnames) + " |"
    sep = "| " + " | ".join(["---"] * len(fieldnames)) + " |"
    body = ["| " + " | ".join(str(row.get(col, "")) for col in fieldnames) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _collect_base_single_rows() -> List[Row]:
    rows: List[Row] = []
    for language in EXPECTED_LANGUAGES:
        eval_dir = ST_ROOT / language / "metrics" / "eval" / "test"
        if not eval_dir.exists():
            raise ValueError(f"Missing ST eval directory: {eval_dir}")
        for path in sorted(eval_dir.glob("*.json")):
            payload = _load_json(path)
            _require_metrics(path, payload)
            runtime = float(payload["runtime"])
            sps = float(payload["samples_per_second"])
            rows.append(
                Row(
                    language=language,
                    run_family="base_or_single",
                    run_id=path.stem,
                    merge_method="",
                    bleu=float(payload["bleu"]),
                    chrf=float(payload["chrf"]),
                    loss=float(payload["loss"]),
                    runtime=runtime,
                    samples_per_second=sps,
                    est_num_samples=runtime * sps,
                    path=str(path),
                )
            )
    return rows


def _parse_merged_filename(path: Path) -> Tuple[str, str]:
    stem = path.stem
    if not stem.startswith("st_"):
        raise ValueError(f"Unexpected merged ST filename (missing st_ prefix): {path}")

    if "_merged_" in stem:
        language_part, suffix = stem.split("_merged_", 1)
        language = language_part.replace("st_", "", 1)
        run_id = f"merged_{suffix}"
        return language, run_id

    raise ValueError(f"Unexpected merged ST filename format: {path}")


def _extract_merge_method(path: Path) -> str:
    parts = path.parts
    if "merged" not in parts:
        raise ValueError(f"Cannot infer merge method from path: {path}")
    idx = parts.index("merged")
    if idx + 1 >= len(parts):
        raise ValueError(f"Cannot infer merge method from path: {path}")
    return parts[idx + 1]


def _collect_merged_rows() -> List[Row]:
    rows: List[Row] = []
    for base_dir in MERGED_PER_TASK_ST_GLOBS:
        if not base_dir.exists():
            raise ValueError(f"Missing merged per-task ST directory: {base_dir}")
        for path in sorted(base_dir.glob("st_en_*_merged_*.json")):
            language, run_id = _parse_merged_filename(path)
            if language not in EXPECTED_LANGUAGES:
                continue
            payload = _load_json(path)
            _require_metrics(path, payload)
            runtime = float(payload["runtime"])
            sps = float(payload["samples_per_second"])
            rows.append(
                Row(
                    language=language,
                    run_family="merged",
                    run_id=run_id,
                    merge_method=_extract_merge_method(path),
                    bleu=float(payload["bleu"]),
                    chrf=float(payload["chrf"]),
                    loss=float(payload["loss"]),
                    runtime=runtime,
                    samples_per_second=sps,
                    est_num_samples=runtime * sps,
                    path=str(path),
                )
            )
    return rows


def _build_delta_rows(all_rows: Sequence[Row]) -> List[DeltaRow]:
    base_by_language: Dict[str, Row] = {}
    for row in all_rows:
        if row.run_id == "base_model":
            base_by_language[row.language] = row

    missing_bases = [lang for lang in EXPECTED_LANGUAGES if lang not in base_by_language]
    if missing_bases:
        raise ValueError(f"Missing base_model rows for languages: {missing_bases}")

    delta_rows: List[DeltaRow] = []
    for row in all_rows:
        base = base_by_language[row.language]
        delta_rows.append(
            DeltaRow(
                language=row.language,
                run_family=row.run_family,
                run_id=row.run_id,
                merge_method=row.merge_method,
                bleu=row.bleu,
                chrf=row.chrf,
                loss=row.loss,
                delta_bleu_vs_base=row.bleu - base.bleu,
                delta_chrf_vs_base=row.chrf - base.chrf,
                delta_loss_vs_base=row.loss - base.loss,
                runtime=row.runtime,
                samples_per_second=row.samples_per_second,
                est_num_samples=row.est_num_samples,
                path=row.path,
            )
        )
    return delta_rows


def _compute_stability(delta_rows: Sequence[DeltaRow]) -> Dict[str, float]:
    by_run: Dict[str, List[float]] = {}
    for row in delta_rows:
        if row.run_id == "base_model":
            continue
        by_run.setdefault(row.run_id, []).append(row.delta_chrf_vs_base)

    stability: Dict[str, float] = {}
    for run_id, vals in by_run.items():
        if len(vals) < 2:
            stability[run_id] = 0.0
        else:
            stability[run_id] = float(pstdev(vals))
    return stability


def _build_rankings(delta_rows: Sequence[DeltaRow]) -> List[Dict[str, Any]]:
    output_rows: List[Dict[str, Any]] = []
    for language in EXPECTED_LANGUAGES:
        rows = [row for row in delta_rows if row.language == language and row.run_id != "base_model"]
        rows_sorted = sorted(
            rows,
            key=lambda r: (-r.delta_chrf_vs_base, -r.delta_bleu_vs_base, r.delta_loss_vs_base, r.run_id),
        )
        for rank, row in enumerate(rows_sorted, start=1):
            output_rows.append(
                {
                    "language": row.language,
                    "rank": rank,
                    "run_family": row.run_family,
                    "run_id": row.run_id,
                    "merge_method": row.merge_method,
                    "delta_chrf_vs_base": _fmt_signed(row.delta_chrf_vs_base),
                    "delta_bleu_vs_base": _fmt_signed(row.delta_bleu_vs_base),
                    "delta_loss_vs_base": _fmt_signed(row.delta_loss_vs_base),
                    "chrf": _fmt(row.chrf),
                    "bleu": _fmt(row.bleu),
                    "loss": _fmt(row.loss),
                }
            )
    return output_rows


def _build_group_summary(delta_rows: Sequence[DeltaRow], stability_by_run: Mapping[str, float]) -> List[Dict[str, Any]]:
    groups = ["base_or_single", "merged"]
    group_rows: List[Dict[str, Any]] = []

    for group in groups:
        rows = [row for row in delta_rows if row.run_family == group and row.run_id != "base_model"]
        if not rows:
            continue
        group_rows.append(
            {
                "level": "group",
                "name": group,
                "macro_avg_delta_chrf_vs_base": _fmt(mean(row.delta_chrf_vs_base for row in rows)),
                "macro_avg_delta_bleu_vs_base": _fmt(mean(row.delta_bleu_vs_base for row in rows)),
                "macro_avg_delta_loss_vs_base": _fmt(mean(row.delta_loss_vs_base for row in rows)),
                "num_rows": len(rows),
                "stability_std_delta_chrf": "",
            }
        )

    run_ids = sorted({row.run_id for row in delta_rows if row.run_id != "base_model"})
    for run_id in run_ids:
        rows = [row for row in delta_rows if row.run_id == run_id]
        if not rows:
            continue
        group_rows.append(
            {
                "level": "run",
                "name": run_id,
                "macro_avg_delta_chrf_vs_base": _fmt(mean(row.delta_chrf_vs_base for row in rows)),
                "macro_avg_delta_bleu_vs_base": _fmt(mean(row.delta_bleu_vs_base for row in rows)),
                "macro_avg_delta_loss_vs_base": _fmt(mean(row.delta_loss_vs_base for row in rows)),
                "num_rows": len(rows),
                "stability_std_delta_chrf": _fmt(stability_by_run.get(run_id, 0.0)),
            }
        )

    return group_rows


def _compute_collisions(all_rows: Sequence[Row]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str, str], List[Row]] = {}
    for row in all_rows:
        key = (row.language, _fmt(row.bleu), _fmt(row.chrf))
        buckets.setdefault(key, []).append(row)

    collisions: List[Dict[str, Any]] = []
    for (language, bleu, chrf), rows in sorted(buckets.items()):
        unique_ids = sorted({row.run_id for row in rows})
        if len(unique_ids) > 1:
            collisions.append(
                {
                    "language": language,
                    "bleu": bleu,
                    "chrf": chrf,
                    "run_ids": unique_ids,
                }
            )
    return collisions


def _compute_metric_disagreements(delta_rows: Sequence[DeltaRow]) -> List[Dict[str, Any]]:
    disagreements: List[Dict[str, Any]] = []
    for row in delta_rows:
        if row.run_id == "base_model":
            continue
        bleu_up = row.delta_bleu_vs_base > 0
        chrf_up = row.delta_chrf_vs_base > 0
        bleu_down = row.delta_bleu_vs_base < 0
        chrf_down = row.delta_chrf_vs_base < 0
        if (bleu_up and chrf_down) or (bleu_down and chrf_up):
            disagreements.append(
                {
                    "language": row.language,
                    "run_id": row.run_id,
                    "run_family": row.run_family,
                    "delta_bleu_vs_base": _fmt_signed(row.delta_bleu_vs_base),
                    "delta_chrf_vs_base": _fmt_signed(row.delta_chrf_vs_base),
                }
            )
    return sorted(disagreements, key=lambda d: (d["language"], d["run_id"]))


def _coverage_checks(all_rows: Sequence[Row]) -> Dict[str, Any]:
    language_set = sorted({row.language for row in all_rows})

    base_single_counts: Dict[str, int] = {}
    for language in EXPECTED_LANGUAGES:
        base_single_counts[language] = sum(
            1 for row in all_rows if row.language == language and row.run_family == "base_or_single"
        )

    merged_methods = sorted({row.merge_method for row in all_rows if row.run_family == "merged" and row.merge_method})

    metric_integrity = all(
        all(not math.isnan(v) for v in (row.bleu, row.chrf, row.loss, row.runtime, row.samples_per_second))
        for row in all_rows
    )

    approx_samples = all(abs(row.est_num_samples - SAMPLE_TARGET) <= SAMPLE_TOLERANCE for row in all_rows)

    base_delta_zero = True
    for language in EXPECTED_LANGUAGES:
        base_rows = [row for row in all_rows if row.language == language and row.run_id == "base_model"]
        if len(base_rows) != 1:
            base_delta_zero = False
            break

    checks = {
        "languages_exactly_expected": language_set == EXPECTED_LANGUAGES,
        "base_single_count_per_language_is_7": all(
            count == EXPECTED_BASE_SINGLE_RUNS_PER_LANG for count in base_single_counts.values()
        ),
        "merged_method_count_is_3": len(merged_methods) == EXPECTED_MERGED_METHOD_COUNT,
        "all_rows_have_required_numeric_metrics": metric_integrity,
        "all_rows_est_num_samples_approx_5000": approx_samples,
        "base_model_present_once_per_language": base_delta_zero,
    }

    return {
        "checks": checks,
        "details": {
            "languages_found": language_set,
            "base_single_counts": base_single_counts,
            "merged_methods": merged_methods,
            "sample_target": SAMPLE_TARGET,
            "sample_tolerance": SAMPLE_TOLERANCE,
            "total_rows": len(all_rows),
        },
    }


def _build_report(
    all_rows: Sequence[Row],
    delta_rows: Sequence[DeltaRow],
    rankings: Sequence[Mapping[str, Any]],
    group_summary: Sequence[Mapping[str, Any]],
    collisions: Sequence[Mapping[str, Any]],
    disagreements: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
) -> str:
    best_single_by_lang: List[Dict[str, Any]] = []
    best_merged_by_lang: List[Dict[str, Any]] = []

    for language in EXPECTED_LANGUAGES:
        candidates_single = [
            row for row in delta_rows if row.language == language and row.run_family == "base_or_single" and row.run_id != "base_model"
        ]
        candidates_merged = [row for row in delta_rows if row.language == language and row.run_family == "merged"]

        best_single = sorted(
            candidates_single,
            key=lambda r: (-r.delta_chrf_vs_base, -r.delta_bleu_vs_base, r.delta_loss_vs_base, r.run_id),
        )[0]
        best_merged = sorted(
            candidates_merged,
            key=lambda r: (-r.delta_chrf_vs_base, -r.delta_bleu_vs_base, r.delta_loss_vs_base, r.run_id),
        )[0]

        best_single_by_lang.append(
            {
                "language": language,
                "run_id": best_single.run_id,
                "delta_chrf_vs_base": _fmt_signed(best_single.delta_chrf_vs_base),
                "delta_bleu_vs_base": _fmt_signed(best_single.delta_bleu_vs_base),
                "delta_loss_vs_base": _fmt_signed(best_single.delta_loss_vs_base),
            }
        )
        best_merged_by_lang.append(
            {
                "language": language,
                "run_id": best_merged.run_id,
                "merge_method": best_merged.merge_method,
                "delta_chrf_vs_base": _fmt_signed(best_merged.delta_chrf_vs_base),
                "delta_bleu_vs_base": _fmt_signed(best_merged.delta_bleu_vs_base),
                "delta_loss_vs_base": _fmt_signed(best_merged.delta_loss_vs_base),
            }
        )

    lines: List[str] = []
    lines.append("# Cross-Language ST Results Package")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Compared base model, single-task adapters, and merged adapters for `en_ar`, `en_de`, and `en_zh-CN`.")
    lines.append("- Primary ranking metric: `chrF` (BLEU and loss reported as secondary).")
    lines.append("- Included only merged runs with per-language ST files (`st_en_{lang}_...`).")
    lines.append("")

    lines.append("## Best Single-Task Adapter per Language (chrF-first)")
    lines.append("")
    lines.append(_to_markdown_table(["language", "run_id", "delta_chrf_vs_base", "delta_bleu_vs_base", "delta_loss_vs_base"], best_single_by_lang))
    lines.append("")

    lines.append("## Best Merged Adapter per Language (chrF-first)")
    lines.append("")
    lines.append(
        _to_markdown_table(
            ["language", "run_id", "merge_method", "delta_chrf_vs_base", "delta_bleu_vs_base", "delta_loss_vs_base"],
            best_merged_by_lang,
        )
    )
    lines.append("")

    lines.append("## Group and Run Macro Summary")
    lines.append("")
    lines.append(
        _to_markdown_table(
            [
                "level",
                "name",
                "macro_avg_delta_chrf_vs_base",
                "macro_avg_delta_bleu_vs_base",
                "macro_avg_delta_loss_vs_base",
                "num_rows",
                "stability_std_delta_chrf",
            ],
            list(group_summary),
        )
    )
    lines.append("")

    lines.append("## Anomaly Flags")
    lines.append("")

    if collisions:
        lines.append("### Metric Collisions (same language + BLEU + chrF across different runs)")
        lines.append("")
        for col in collisions:
            lines.append(
                f"- `{col['language']}` BLEU=`{col['bleu']}` chrF=`{col['chrf']}` shared by: {', '.join(f'`{rid}`' for rid in col['run_ids'])}"
            )
        lines.append("")
    else:
        lines.append("- No metric collisions detected.")
        lines.append("")

    if disagreements:
        lines.append("### BLEU-vs-chrF Disagreements")
        lines.append("")
        lines.append(_to_markdown_table(["language", "run_id", "run_family", "delta_bleu_vs_base", "delta_chrf_vs_base"], list(disagreements)))
        lines.append("")
    else:
        lines.append("- No BLEU-vs-chrF disagreement rows detected.")
        lines.append("")

    lines.append("### Aggregate File Mismatch Note")
    lines.append("")
    lines.append(
        "- `st_merged_*` and some `eval_results_*` files can reflect a single language slice instead of a language-weighted multilingual aggregate."
    )
    lines.append("- For this package, only `st_en_ar_*`, `st_en_de_*`, and `st_en_zh-CN_*` per-task files were used for comparable analysis.")
    lines.append("")

    lines.append("## Validation Checklist")
    lines.append("")
    for key, value in validation["checks"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")

    # Keep a compact snapshot of top ranking rows in the report.
    top_rank_rows = [row for row in rankings if int(row["rank"]) <= 3]
    lines.append("## Top-3 per Language")
    lines.append("")
    lines.append(
        _to_markdown_table(
            [
                "language",
                "rank",
                "run_family",
                "run_id",
                "merge_method",
                "delta_chrf_vs_base",
                "delta_bleu_vs_base",
                "delta_loss_vs_base",
            ],
            top_rank_rows,
        )
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    base_single_rows = _collect_base_single_rows()
    merged_rows = _collect_merged_rows()
    all_rows = [*base_single_rows, *merged_rows]

    validation = _coverage_checks(all_rows)
    if not all(validation["checks"].values()):
        failing = [k for k, v in validation["checks"].items() if not v]
        raise ValueError(f"Validation checks failed before artifact writing: {failing}")

    delta_rows = _build_delta_rows(all_rows)
    # Validate baseline deltas are exactly zero.
    for row in delta_rows:
        if row.run_id == "base_model":
            if row.delta_bleu_vs_base != 0.0 or row.delta_chrf_vs_base != 0.0 or row.delta_loss_vs_base != 0.0:
                raise ValueError(f"Base delta is not zero for language={row.language}")

    stability = _compute_stability(delta_rows)
    rankings = _build_rankings(delta_rows)
    group_summary = _build_group_summary(delta_rows, stability)

    collisions = _compute_collisions(all_rows)
    disagreements = _compute_metric_disagreements(delta_rows)

    normalized_rows: List[Dict[str, Any]] = []
    for row in sorted(all_rows, key=lambda r: (r.language, r.run_family, r.run_id)):
        normalized_rows.append(
            {
                "language": row.language,
                "run_family": row.run_family,
                "run_id": row.run_id,
                "merge_method": row.merge_method,
                "bleu": _fmt(row.bleu),
                "chrf": _fmt(row.chrf),
                "loss": _fmt(row.loss),
                "runtime": _fmt(row.runtime, 4),
                "samples_per_second": _fmt(row.samples_per_second, 4),
                "est_num_samples": _fmt(row.est_num_samples, 1),
                "path": row.path,
            }
        )

    delta_out_rows: List[Dict[str, Any]] = []
    for row in sorted(delta_rows, key=lambda r: (r.language, r.run_family, r.run_id)):
        delta_out_rows.append(
            {
                "language": row.language,
                "run_family": row.run_family,
                "run_id": row.run_id,
                "merge_method": row.merge_method,
                "bleu": _fmt(row.bleu),
                "chrf": _fmt(row.chrf),
                "loss": _fmt(row.loss),
                "delta_bleu_vs_base": _fmt_signed(row.delta_bleu_vs_base),
                "delta_chrf_vs_base": _fmt_signed(row.delta_chrf_vs_base),
                "delta_loss_vs_base": _fmt_signed(row.delta_loss_vs_base),
                "runtime": _fmt(row.runtime, 4),
                "samples_per_second": _fmt(row.samples_per_second, 4),
                "est_num_samples": _fmt(row.est_num_samples, 1),
                "path": row.path,
            }
        )

    _write_csv(
        output_dir / "st_metrics_normalized.csv",
        [
            "language",
            "run_family",
            "run_id",
            "merge_method",
            "bleu",
            "chrf",
            "loss",
            "runtime",
            "samples_per_second",
            "est_num_samples",
            "path",
        ],
        normalized_rows,
    )

    _write_csv(
        output_dir / "st_delta_vs_base_by_language.csv",
        [
            "language",
            "run_family",
            "run_id",
            "merge_method",
            "bleu",
            "chrf",
            "loss",
            "delta_bleu_vs_base",
            "delta_chrf_vs_base",
            "delta_loss_vs_base",
            "runtime",
            "samples_per_second",
            "est_num_samples",
            "path",
        ],
        delta_out_rows,
    )

    _write_csv(
        output_dir / "st_rankings_by_language.csv",
        [
            "language",
            "rank",
            "run_family",
            "run_id",
            "merge_method",
            "delta_chrf_vs_base",
            "delta_bleu_vs_base",
            "delta_loss_vs_base",
            "chrf",
            "bleu",
            "loss",
        ],
        rankings,
    )

    _write_csv(
        output_dir / "st_group_summary.csv",
        [
            "level",
            "name",
            "macro_avg_delta_chrf_vs_base",
            "macro_avg_delta_bleu_vs_base",
            "macro_avg_delta_loss_vs_base",
            "num_rows",
            "stability_std_delta_chrf",
        ],
        group_summary,
    )

    report = _build_report(
        all_rows=all_rows,
        delta_rows=delta_rows,
        rankings=rankings,
        group_summary=group_summary,
        collisions=collisions,
        disagreements=disagreements,
        validation=validation,
    )
    (output_dir / "st_analysis_report.md").write_text(report, encoding="utf-8")

    print(f"Wrote ST multilingual analysis to: {output_dir}")
    print(f"Rows included: {len(all_rows)} (base_or_single={len(base_single_rows)}, merged={len(merged_rows)})")
    print(f"Metric collisions: {len(collisions)}")
    print(f"BLEU-vs-chrF disagreements: {len(disagreements)}")


if __name__ == "__main__":
    main()
