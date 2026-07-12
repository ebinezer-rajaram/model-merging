#!/usr/bin/env python3
"""Package thesis result artifacts from existing repository outputs.

This script is intentionally read-only with respect to experiment outputs.  It
discovers, indexes, validates, copies, and lightly plots existing files into
``thesis_artifacts``.  It does not launch training, inference, or evaluation.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import re
import shutil
import struct
import subprocess
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "thesis_artifacts"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
MANIFESTS = OUT / "manifests"
THESIS_MAIN = ROOT / "thesis" / "main.tex"

DISCOVERY_ROOTS = [
    "artifacts",
    "analysis",
]
EXTENSIONS = {
    ".csv",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".txt",
    ".log",
    ".tex",
    ".pkl",
    ".parquet",
    ".npz",
    ".npy",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
}

TASK_ALIASES = {
    "asr": "ASR",
    "wer": "ASR",
    "emotion": "Emotion",
    "meld": "Emotion",
    "intent": "Intent",
    "kws": "KWS",
    "keyword": "KWS",
    "langid": "LangID",
    "language_id": "LangID",
    "language-identification": "LangID",
    "speaker_ver": "SpeakerVer",
    "speaker-ver": "SpeakerVer",
    "speaker_verification": "SpeakerVer",
    "speaker_id": "SpeakerVer",
    "speaker": "SpeakerVer",
    "vocalsound": "VocalSound",
    "vocal_sound": "VocalSound",
    "speech_qa": "SpeechQA",
    "spoken_qa": "SpeechQA",
    "mmsu": "SpeechQA",
    "st": "SpeechTranslation",
    "speech_translation": "SpeechTranslation",
    "en_de": "SpeechTranslation",
    "en_ar": "SpeechTranslation",
    "en_zh": "SpeechTranslation",
}
METHOD_ALIASES = {
    "base_model": "Base",
    "base": "Base",
    "best_single_task": "SingleTask",
    "single_task": "SingleTask",
    "best_": "SingleTask",
    "cross_task": "CrossTaskAdapter",
    "uniform_delta": "Uniform",
    "uniform": "Uniform",
    "uniform_scalar_delta": "UniformScalar",
    "weighted_delta_n": "LayerWise",
    "weighted_delta": "TaskWeighted",
    "weighted": "TaskWeighted",
    "supermerge": "GradientOptimised",
    "gradient": "GradientOptimised",
    "ties": "TIES",
    "dare": "DARE",
    "adamerging": "AdaMerging",
    "mtl": "MTL",
    "sequential": "SequentialFT",
    "continual_supermerge": "ContinualMerge",
    "continual_merge": "ContinualMerge",
    "continual_mtl": "ContinualMTL",
    "reoptim": "ReoptimisedUpperBound",
    "upper_bound": "ReoptimisedUpperBound",
}
METRIC_HINTS = [
    "wer",
    "accuracy",
    "acc",
    "macro_f1",
    "weighted_f1",
    "f1",
    "loss",
    "interference_delta",
    "recovery",
    "delta",
    "runtime",
    "samples_per_second",
]
CANON_TASK_ORDER = [
    "ASR",
    "Emotion",
    "Intent",
    "KWS",
    "LangID",
    "SpeakerVer",
    "VocalSound",
    "SpeechQA",
    "SpeechTranslation",
]


used_sources: set[str] = set()
generated_tables: list[str] = []
generated_figures: list[str] = []
missing_items: list[dict[str, str]] = []
inconsistencies: list[str] = []
artifact_status: list[dict[str, Any]] = []


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def ensure_dirs() -> None:
    for p in (OUT, TABLES, FIGURES, MANIFESTS):
        p.mkdir(parents=True, exist_ok=True)


def norm_task(text: str) -> list[str]:
    s = text.lower()
    found = []
    for alias, canon in TASK_ALIASES.items():
        if re.search(rf"(^|[^a-z0-9]){re.escape(alias)}([^a-z0-9]|$)", s):
            found.append(canon)
    return sorted(set(found), key=CANON_TASK_ORDER.index)


def norm_method(text: str) -> list[str]:
    s = text.lower()
    found = []
    for alias, canon in METHOD_ALIASES.items():
        if alias in s:
            found.append(canon)
    return sorted(set(found))


def infer_metrics(text: str) -> list[str]:
    s = text.lower()
    return sorted({m for m in METRIC_HINTS if m in s})


def read_text_sample(path: Path, limit: int = 32768) -> str:
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".pkl", ".parquet", ".npz", ".npy"}:
        return ""
    r = rel(path)
    sample_roots = (
        "analysis/results/",
        "analysis/evaluation/",
        "analysis/merge_comparison/",
        "artifacts/merged/comparisons/",
        "artifacts/continual_suite/",
        "configs/",
        "main.tex",
    )
    if not r.startswith(sample_roots):
        return ""
    parts = set(path.parts)
    # The wandb/log trees are large and noisy.  They are still indexed by
    # metadata and path, but content sampling is reserved for likely canonical
    # structured outputs to keep packaging interactive on network filesystems.
    if "wandb" in parts or ("logs" in parts and "artifacts" not in parts and "analysis" not in parts):
        return ""
    try:
        if path.stat().st_size > 5_000_000:
            return ""
        return path.read_text(errors="ignore")[:limit]
    except Exception:
        return ""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="ignore") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for k in row:
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def fmt_value(v: Any) -> str:
    if v is None:
        return ""
    s = str(v)
    if s == "":
        return ""
    try:
        x = float(s)
    except Exception:
        return s
    if math.isnan(x):
        return ""
    if abs(x) >= 1000:
        return f"{x:.1f}"
    return f"{x:.3f}"


def escape_tex(s: Any) -> str:
    out = str(s)
    for a, b in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    ]:
        out = out.replace(a, b)
    return out


def write_tex_table(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    lines = ["\\begin{tabular}{" + "l" * len(fieldnames) + "}", "\\toprule"]
    lines.append(" & ".join(escape_tex(k) for k in fieldnames) + r" \\")
    lines.append("\\midrule")
    for row in rows:
        lines.append(" & ".join(escape_tex(fmt_value(row.get(k, ""))) for k in fieldnames) + r" \\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def save_table(name: str, rows: list[dict[str, Any]], fields: list[str] | None, sources: list[Path], note: str = "") -> None:
    csv_path = TABLES / f"{name}.csv"
    tex_path = TABLES / f"{name}.tex"
    write_csv(csv_path, rows, fields)
    write_tex_table(tex_path, rows, fields)
    generated_tables.extend([rel(csv_path), rel(tex_path)])
    for p in sources:
        used_sources.add(rel(p))
    artifact_status.append(
        {
            "artifact": name,
            "status": "complete" if rows else "missing",
            "sources": "; ".join(rel(p) for p in sources),
            "tasks": ", ".join(sorted({t for p in sources for t in norm_task(str(p))})),
            "methods": ", ".join(sorted({m for p in sources for m in norm_method(str(p))})),
            "notes": note or f"{len(rows)} rows",
        }
    )


def copy_if_exists(src: Path, dst_name: str) -> bool:
    if not src.exists():
        return False
    dst = FIGURES / dst_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    generated_figures.append(rel(dst))
    used_sources.add(rel(src))
    return True


def png_write(path: Path, width: int, height: int, pixels: list[tuple[int, int, int]]) -> None:
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            raw.extend(pixels[y * width + x])
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    payload = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            chunk(b"IDAT", zlib.compress(bytes(raw), 9)),
            chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(payload)


def draw_heatmap(name: str, matrix: list[list[float | None]], labels_x: list[str], labels_y: list[str], source: Path) -> None:
    if not matrix:
        return
    cell = 38
    left = 120
    top = 60
    width = left + cell * len(labels_x) + 20
    height = top + cell * len(labels_y) + 30
    vals = [v for row in matrix for v in row if v is not None]
    if not vals:
        return
    lo, hi = min(vals), max(vals)
    if lo == hi:
        lo -= 1
        hi += 1
    pixels = [(255, 255, 255)] * (width * height)
    for j, row in enumerate(matrix):
        for i, val in enumerate(row):
            if val is None:
                color = (230, 230, 230)
            else:
                r = (val - lo) / (hi - lo)
                color = (int(35 + 200 * r), int(80 + 120 * (1 - abs(r - 0.5) * 2)), int(210 - 170 * r))
            for y in range(top + j * cell, top + (j + 1) * cell - 2):
                for x in range(left + i * cell, left + (i + 1) * cell - 2):
                    pixels[y * width + x] = color
    png_path = FIGURES / f"{name}.png"
    pdf_path = FIGURES / f"{name}.pdf"
    png_write(png_path, width, height, pixels)
    write_simple_pdf(pdf_path, width, height, matrix, labels_x, labels_y)
    generated_figures.extend([rel(png_path), rel(pdf_path)])
    used_sources.add(rel(source))


def write_simple_pdf(path: Path, width: int, height: int, matrix: list[list[float | None]], labels_x: list[str], labels_y: list[str]) -> None:
    cell = 38
    left = 120
    top = 60
    vals = [v for row in matrix for v in row if v is not None]
    lo, hi = (min(vals), max(vals)) if vals else (0, 1)
    if lo == hi:
        lo -= 1
        hi += 1
    cmds = ["0.95 0.95 0.95 rg 0 0 %d %d re f" % (width, height), "0 0 0 rg /F1 8 Tf"]
    for i, label in enumerate(labels_x):
        cmds.append(f"BT 1 0 0 1 {left+i*cell+2} {height-25} Tm ({pdf_escape(label[:10])}) Tj ET")
    for j, label in enumerate(labels_y):
        cmds.append(f"BT 1 0 0 1 8 {height-(top+j*cell+22)} Tm ({pdf_escape(label[:16])}) Tj ET")
    for j, row in enumerate(matrix):
        for i, val in enumerate(row):
            if val is None:
                color = (0.85, 0.85, 0.85)
            else:
                r = (val - lo) / (hi - lo)
                color = (0.15 + 0.75 * r, 0.30 + 0.45 * (1 - abs(r - 0.5) * 2), 0.80 - 0.65 * r)
            x = left + i * cell
            y = height - top - (j + 1) * cell
            cmds.append(f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg {x} {y} {cell-2} {cell-2} re f")
            if val is not None:
                cmds.append(f"0 0 0 rg BT /F1 7 Tf 1 0 0 1 {x+5} {y+16} Tm ({val:.2f}) Tj ET")
    pdf_stream = "\n".join(cmds).encode("latin1", "ignore")
    make_pdf(path, width, height, pdf_stream)


def pdf_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(path: Path, width: int, height: int, stream: bytes) -> None:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    chunks = [b"%PDF-1.4\n"]
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(sum(len(c) for c in chunks))
        chunks.append(f"{i} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = sum(len(c) for c in chunks)
    chunks.append(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
    for off in offsets[1:]:
        chunks.append(f"{off:010d} 00000 n \n".encode())
    chunks.append(f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(b"".join(chunks))


def draw_grouped_bars(name: str, rows: list[dict[str, Any]], label_col: str, value_cols: list[str], source: Path) -> None:
    labels = [r[label_col] for r in rows]
    matrix: list[list[float | None]] = []
    for vc in value_cols:
        matrix.append([])
        for r in rows:
            try:
                matrix[-1].append(float(r.get(vc, "")))
            except Exception:
                matrix[-1].append(None)
    draw_heatmap(name, matrix, labels, value_cols, source)


def parse_main() -> list[dict[str, Any]]:
    path = THESIS_MAIN
    sections: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text(errors="ignore").splitlines(), start=1):
        m = re.search(r"\\(chapter|section|subsection)\*?\{([^}]+)\}", line)
        if m:
            sections.append({"line": i, "level": m.group(1), "title": m.group(2)})
    return sections


def write_discovery_plan(sections: list[dict[str, Any]]) -> None:
    relevant = [s for s in sections if s["line"] >= 699]
    lines = [
        "# Discovery Plan",
        "",
        "Parsed from `main.tex`. Result-bearing sections start at Chapter 4 and continue through appendices.",
        "",
        "## Thesis Sections Needing Results",
    ]
    for s in relevant:
        lines.append(f"- line {s['line']}: {s['level']} `{s['title']}`")
    lines += [
        "",
        "## Expected Tables/Figures",
        "- Experimental setup: task suite, datasets/splits, LoRA/training hyperparameters, evaluation metrics, compute/cost.",
        "- Single-task/interference: base vs single-task, cross-task absolute and recovery matrices, heatmaps, task-vector norms/similarity and layer/component plots.",
        "- Merging: seven-task absolute and recovery tables, mean/min/geometric recovery summaries, per-task recovery plots, scalar sweep/calibration plots, minor TIES/DARE/AdaMerging comparisons where present, SpeechQA/MMSU tables.",
        "- MTL: six- and seven-task MTL vs merge absolute/recovery tables and performance/modularity summary.",
        "- Continual: six-to-seven and five-to-six/seven extension tables, retention/new-task/degradation summaries, curves, sequential FT/reoptimised upper-bound if present.",
        "- Appendix: pairwise sweeps, confusion/training curves, coefficient plots, ASR error analysis, prompt/split details, SpeechTranslation boundary-case results.",
        "",
        "## Expected Tasks",
        ", ".join(CANON_TASK_ORDER) + ". SpeechQA/MMSU is treated as held-out/OOD; SpeechTranslation is appendix/boundary-case.",
        "",
        "## Expected Methods",
        ", ".join(["Base", "SingleTask", "CrossTaskAdapter", "Uniform", "UniformScalar", "TaskWeighted", "LayerWise", "GradientOptimised", "TIES", "DARE", "AdaMerging", "MTL", "SequentialFT", "ContinualMerge", "ContinualMTL", "ReoptimisedUpperBound"]),
        "",
        "## Validation Rules",
        "- Prefer latest machine-readable CSV/JSON/JSONL in `artifacts/` or `analysis/`.",
        "- Use logs only when structured files are unavailable.",
        "- Do not fabricate values; mark missing or partial results explicitly.",
        "- Recovery for higher-is-better metrics: `(model - base) / (single - base)`.",
        "- Recovery for ASR/WER: `(WER_base - WER_model) / (WER_base - WER_single_ASR)`.",
        "- Record conflicting canonical values in `inconsistencies.md`.",
        "",
    ]
    (MANIFESTS / "discovery_plan.md").write_text("\n".join(lines), encoding="utf-8")


def discover_files() -> list[dict[str, Any]]:
    records = []
    roots = [str(ROOT / r) for r in DISCOVERY_ROOTS if (ROOT / r).exists()]
    if not roots:
        return records
    expr: list[str] = []
    for ext in sorted(EXTENSIONS):
        if expr:
            expr.append("-o")
        expr.extend(["-name", f"*{ext}"])
    cmd = ["find", *roots, "-type", "f", "(", *expr, ")"]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    paths = [Path(line) for line in proc.stdout.splitlines() if line]
    for path in paths:
        try:
            stat = path.stat()
            if not path.is_file():
                continue
        except OSError:
            continue
        text = rel(path) + "\n" + read_text_sample(path)
        records.append(
            {
                "path": rel(path),
                "file_type": path.suffix.lower().lstrip("."),
                "size": stat.st_size,
                "modified_time": dt.datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "inferred_tasks": ";".join(norm_task(text)),
                "inferred_methods": ";".join(norm_method(text)),
                "metrics_found": ";".join(infer_metrics(text)),
                "used": "no",
                "reason_if_unused": "not selected as canonical source or auxiliary artifact",
            }
        )
    return records


def update_source_index(records: list[dict[str, Any]]) -> None:
    for r in records:
        if r["path"] in used_sources:
            r["used"] = "yes"
            r["reason_if_unused"] = ""
    fields = ["path", "file_type", "size", "modified_time", "inferred_tasks", "inferred_methods", "metrics_found", "used", "reason_if_unused"]
    write_csv(MANIFESTS / "source_index.csv", records, fields)


def write_alias_map() -> None:
    (MANIFESTS / "alias_map.json").write_text(
        json.dumps({"tasks": TASK_ALIASES, "methods": METHOD_ALIASES}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_all_experiments() -> list[dict[str, str]]:
    p = ROOT / "analysis/results/all_experiments.csv"
    return read_csv(p) if p.exists() else []


def canonical_task(t: str) -> str:
    return TASK_ALIASES.get(t.lower(), t)


def canonical_method(m: str) -> str:
    if m == "weighted_delta_n_stitched":
        return "LayerWise"
    if m.endswith("_merge"):
        m = m[:-6]
    return METHOD_ALIASES.get(m.lower(), METHOD_ALIASES.get(m.lower().replace("_stitched", ""), m))


def generate_setup_tables(rows: list[dict[str, str]]) -> None:
    task_rows = []
    for task in ["asr", "emotion", "intent", "kws", "langid", "speaker_ver", "vocalsound", "speech_qa", "st"]:
        cfg = ROOT / "configs" / f"{task}.yaml"
        info = {"Task": canonical_task(task), "Config": rel(cfg) if cfg.exists() else "", "Dataset/Source": "", "Split Notes": ""}
        if cfg.exists() and yaml:
            try:
                data = yaml.safe_load(cfg.read_text()) or {}
                info["Dataset/Source"] = str(data.get("dataset", data.get("data", data.get("name", ""))))[:120]
                info["Split Notes"] = str(data.get("splits", data.get("split", "")))[:120]
                used_sources.add(rel(cfg))
            except Exception:
                pass
        task_rows.append(info)
    save_table("experimental_task_suite", task_rows, ["Task", "Config", "Dataset/Source", "Split Notes"], [THESIS_MAIN], "Setup table inferred from configs and thesis.")

    hp = []
    for r in rows:
        if r.get("experiment_type") != "single_task" or r.get("is_primary_metric") != "True":
            continue
        hp.append(
            {
                "Task": canonical_task(r["task"]),
                "Learning Rate": r.get("learning_rate", ""),
                "LoRA r": r.get("lora_r", ""),
                "LoRA alpha": r.get("lora_alpha", ""),
                "Epochs": r.get("num_train_epochs", ""),
                "Batch Size": r.get("per_device_train_batch_size", ""),
                "Selection Metric": r.get("selection_metric_name", ""),
            }
        )
    save_table("lora_training_hyperparameters", hp, ["Task", "Learning Rate", "LoRA r", "LoRA alpha", "Epochs", "Batch Size", "Selection Metric"], [ROOT / "analysis/results/all_experiments.csv"])

    metric_rows = [
        {"Task": "ASR", "Primary metric": "WER", "Direction": "lower is better", "Recovery orientation": "base WER - model WER"},
        {"Task": "Emotion", "Primary metric": "Macro-F1", "Direction": "higher is better", "Recovery orientation": "model - base"},
        {"Task": "Intent", "Primary metric": "Accuracy", "Direction": "higher is better", "Recovery orientation": "model - base"},
        {"Task": "KWS", "Primary metric": "Macro-F1", "Direction": "higher is better", "Recovery orientation": "model - base"},
        {"Task": "LangID", "Primary metric": "Accuracy", "Direction": "higher is better", "Recovery orientation": "model - base"},
        {"Task": "SpeakerVer", "Primary metric": "Accuracy", "Direction": "higher is better", "Recovery orientation": "model - base"},
        {"Task": "VocalSound", "Primary metric": "Accuracy", "Direction": "higher is better", "Recovery orientation": "model - base"},
        {"Task": "SpeechQA", "Primary metric": "Accuracy", "Direction": "higher is better", "Recovery orientation": "OOD probe; not trained task"},
        {"Task": "SpeechTranslation", "Primary metric": "BLEU/COMET where available", "Direction": "higher is better", "Recovery orientation": "appendix boundary case"},
    ]
    save_table("evaluation_metrics", metric_rows, ["Task", "Primary metric", "Direction", "Recovery orientation"], [THESIS_MAIN])


def generate_from_existing_csvs() -> None:
    candidates = {
        "seven_task_merge_absolute_metrics": ROOT / "artifacts/merged/comparisons/seven_task_merge_with_baselines_and_interference_20260304.csv",
        "seven_task_merge_recovery_detail": ROOT / "artifacts/merged/comparisons/seven_task_merge_interference_detail_20260304.csv",
        "six_and_seven_task_mtl_merge_summary": ROOT / "artifacts/merged/comparisons/six_and_seven_task_merge_mtl_summary_20260311.csv",
        "mtl_merge_three_setting_summary": ROOT / "artifacts/merged/comparisons/three_setting_merge_mtl_summary_20260319.csv",
        "continual_final_comparison": ROOT / "artifacts/continual_suite/thesis_continual_extension_supermerge_v1/continual_final_comparison.csv",
        "continual_stage_metrics": ROOT / "artifacts/continual_suite/thesis_continual_extension_supermerge_v1/continual_stage_metrics.csv",
        "continual_forgetting_curves": ROOT / "artifacts/continual_suite/thesis_continual_extension_supermerge_v1/continual_forgetting_curves.csv",
        "continual_growth_curves": ROOT / "artifacts/continual_suite/thesis_continual_extension_supermerge_v1/continual_growth_curves.csv",
        "continual_mtl_merge_final_comparison": ROOT / "artifacts/continual_suite/thesis_continual_extension_v1/continual_final_comparison.csv",
        "continual_mtl_merge_stage_metrics": ROOT / "artifacts/continual_suite/thesis_continual_extension_v1/continual_stage_metrics.csv",
        "mmsu_overall_metrics": ROOT / "analysis/results/mmsu_spoken_qa/overall_metrics_table.csv",
        "mmsu_category_rollup": ROOT / "analysis/results/mmsu_spoken_qa/category_rollup_mean_delta_vs_base.csv",
        "speech_translation_metrics": ROOT / "analysis/results/st_multilingual/st_metrics_normalized.csv",
        "speech_translation_group_summary": ROOT / "analysis/results/st_multilingual/st_group_summary.csv",
        "asr_scalar_sweep": ROOT / "analysis/results/asr_scalar_sweep/asr_only_scalar_sweep.csv",
        "asr_magnitude_calibration": ROOT / "analysis/results/asr_mechanism/alphaeff_mechanism_points.csv",
        "asr_error_summary": ROOT / "analysis/results/asr_pairwise/summary.csv",
        "task_vector_norms": ROOT / "analysis/results/chapter5/task_vector_norms_7task.csv",
        "task_vector_cosine_matrix": ROOT / "analysis/results/chapter5/task_vector_cosine_7task.csv",
        "layer_wise_statistics": ROOT / "analysis/results/chapter5/layer_wise_statistics_7task.csv",
        "component_statistics": ROOT / "analysis/results/chapter5/component_statistics_7task.csv",
        "pairwise_weighted_delta_best_lambda": ROOT / "analysis/results/old/weighted_delta_best_lambda_max_min_interference_delta.csv",
        "layerwise_coefficient_summary": ROOT / "artifacts/merged/weighted_delta_n/emotion_intent_kws_langid/runs/run_supermerge_layer_wise_20260215_045019/layerwise_coeff_summary.csv",
        "layerwise_coefficient_analysis": ROOT / "artifacts/merged/weighted_delta_n/emotion_intent_kws_langid/runs/run_supermerge_layer_wise_20260215_045019/layerwise_coeff_analysis.csv",
    }
    for name, src in candidates.items():
        if not src.exists():
            mark_missing(name, "Appendix/main", "canonical source file", "none", "No raw machine-readable source found at expected path.", "Check result generation scripts; do not run expensive evaluation without confirmation.", "medium")
            continue
        rows = read_csv(src)
        save_table(name, rows, None, [src])


def generate_appendix_inventory() -> None:
    confusion_paths = sorted((ROOT / "artifacts").glob("*/metrics/eval/test/**/confusion_matrix*.png"))
    confusion_rows = []
    for p in confusion_paths:
        confusion_rows.append(
            {
                "Path": rel(p),
                "Task": ";".join(norm_task(rel(p))),
                "Method": ";".join(norm_method(rel(p))),
            }
        )
    if confusion_rows:
        save_table("confusion_matrix_inventory", confusion_rows, ["Path", "Task", "Method"], confusion_paths[:1], "Inventory of available confusion matrix figures.")
        # Copy the compact canonical set: base, single-task, and seven-task merge matrices.
        selected = []
        for p in confusion_paths:
            r = rel(p)
            if (
                "confusion_matrix_base_model.png" in r
                or re.search(r"confusion_matrix_(emotion|intent|kws|langid|speaker_ver|vocalsound)_adapter\\.png$", r)
                or "multi_asr_emotion_intent_kws_langid_speaker_ver_vocalsound" in r
            ):
                selected.append(p)
        for p in selected[:80]:
            safe = rel(p).replace("/", "__")
            copy_if_exists(p, f"confusion__{safe}")

    prompt = ROOT / "artifacts/speech_qa/metrics/prompt_sweep/prompt_sweep_mmsu_250.json"
    if prompt.exists():
        try:
            data = json.loads(prompt.read_text())
            rows = []
            for item in data.get("results", []):
                if isinstance(item, dict):
                    row = {k: item.get(k, "") for k in item.keys() if isinstance(item.get(k, ""), (str, int, float, bool, type(None)))}
                    rows.append(row)
            if rows:
                save_table("speechqa_prompt_sweep", rows, None, [prompt], "Prompt sweep results; best template is recorded in source JSON.")
            else:
                save_table(
                    "speechqa_prompt_sweep_summary",
                    [
                        {
                            "Split": data.get("split", ""),
                            "Samples": data.get("num_samples", ""),
                            "Seed": data.get("seed", ""),
                            "Best Template": str(data.get("best_template", ""))[:500],
                            "Source": rel(prompt),
                        }
                    ],
                    ["Split", "Samples", "Seed", "Best Template", "Source"],
                    [prompt],
                )
        except Exception as exc:
            inconsistencies.append(f"- Failed to parse prompt sweep `{rel(prompt)}`: {exc}")


def generate_sequential_ft_from_continual_mtl() -> None:
    final = ROOT / "artifacts/continual_suite/thesis_continual_extension_v1/continual_final_comparison.csv"
    stage = ROOT / "artifacts/continual_suite/thesis_continual_extension_v1/continual_stage_metrics.csv"
    for src, name in [
        (final, "sequential_ft_continual_mtl_final_comparison"),
        (stage, "sequential_ft_continual_mtl_stage_metrics"),
    ]:
        if not src.exists():
            continue
        rows = [r for r in read_csv(src) if r.get("method") == "continual_mtl"]
        for r in rows:
            r["method_normalised"] = "SequentialFT/ContinualMTL"
        if rows:
            fields = ["method_normalised"] + [k for k in rows[0].keys() if k != "method_normalised"]
            save_table(
                name,
                rows,
                fields,
                [src],
                "Continual MTL is treated as the sequential fine-tuning / continual baseline for this thesis.",
            )


def generate_single_task_tables() -> None:
    src = ROOT / "artifacts/merged/comparisons/seven_task_merge_with_baselines_and_interference_20260304.csv"
    if not src.exists():
        return
    rows = read_csv(src)
    base = next((r for r in rows if r.get("method") == "base_model"), {})
    single = next((r for r in rows if r.get("method") == "best_single_task"), {})
    fields = [
        ("ASR", "asr_wer", "WER (lower better)"),
        ("Emotion", "emotion_macro_f1", "Macro-F1"),
        ("Intent", "intent_acc", "Accuracy"),
        ("KWS", "kws_macro_f1", "Macro-F1"),
        ("LangID", "langid_acc", "Accuracy"),
        ("SpeakerVer", "speaker_ver_acc", "Accuracy"),
        ("VocalSound", "vocalsound_acc", "Accuracy"),
    ]
    out = [{"Task": t, "Metric": metric, "Base": base.get(col, ""), "SingleTask": single.get(col, "")} for t, col, metric in fields]
    save_table("base_vs_single_task_adapters", out, ["Task", "Metric", "Base", "SingleTask"], [src])


def generate_minor_method_comparisons(rows: list[dict[str, str]]) -> None:
    subset = []
    for r in rows:
        if r.get("method") not in {"ties", "dare", "adamerging"} or r.get("is_primary_metric") != "True":
            continue
        subset.append(
            {
                "Method": canonical_method(r.get("method", "")),
                "Task Set": r.get("source_tasks_key", ""),
                "Task": canonical_task(r.get("task", "")),
                "Metric": r.get("metric_name", ""),
                "Value": r.get("metric_value", ""),
                "Recovery": r.get("interference_delta", ""),
                "Split": r.get("split", ""),
                "Source": r.get("source_path", ""),
            }
        )
    if subset:
        save_table(
            "minor_ties_dare_adamerging_comparison",
            subset,
            ["Method", "Task Set", "Task", "Metric", "Value", "Recovery", "Split", "Source"],
            [ROOT / "analysis/results/all_experiments.csv"],
            "Minor/subset comparison; no AdaMerging primary rows were present in all_experiments.",
        )


def pivot_matrix(src: Path, name: str) -> None:
    if not src.exists():
        return
    rows = read_csv(src)
    fields = list(rows[0].keys()) if rows else []
    save_table(name, rows, fields, [src])
    if rows and len(fields) > 1:
        matrix = []
        ylabels = []
        xlabels = fields[1:]
        for r in rows:
            ylabels.append(r.get(fields[0], ""))
            row = []
            for f in xlabels:
                try:
                    row.append(float(r.get(f, "")))
                except Exception:
                    row.append(None)
            matrix.append(row)
        draw_heatmap(name + "_heatmap", matrix, xlabels, ylabels, src)


def generate_interference_tables() -> None:
    chapter5 = ROOT / "analysis/results/chapter5"
    pivot_matrix(chapter5 / "cross_task_absolute_matrix_7task.csv", "cross_task_absolute_matrix")
    pivot_matrix(chapter5 / "cross_task_recovery_matrix_7task.csv", "cross_task_recovery_interference_matrix")
    pivot_matrix(chapter5 / "task_vector_cosine_7task.csv", "task_vector_cosine_matrix_heatmap_source")


def generate_summaries_and_figures() -> None:
    seven = ROOT / "artifacts/merged/comparisons/seven_task_merge_with_baselines_and_interference_20260304.csv"
    if seven.exists():
        rows = read_csv(seven)
        summary = []
        for r in rows:
            if r.get("min_interference_delta") or r.get("mean_interference_delta"):
                summary.append(
                    {
                        "Method": canonical_method(r.get("method", "")),
                        "Mean Recovery": r.get("mean_interference_delta", ""),
                        "Min Recovery": r.get("min_interference_delta", ""),
                        "ASR Recovery": "",
                    }
                )
        detail = ROOT / "artifacts/merged/comparisons/seven_task_merge_interference_detail_20260304.csv"
        if detail.exists():
            by_method = defaultdict(dict)
            for r in read_csv(detail):
                by_method[canonical_method(r.get("method", ""))][canonical_task(r.get("task", ""))] = r.get("interference_delta", "")
            for s in summary:
                s["ASR Recovery"] = by_method.get(s["Method"], {}).get("ASR", "")
        save_table("merge_recovery_summary_mean_min", summary, ["Method", "Mean Recovery", "Min Recovery", "ASR Recovery"], [seven, detail])
        draw_grouped_bars("merge_recovery_summary_heatmap", summary, "Method", ["Mean Recovery", "Min Recovery", "ASR Recovery"], seven)
    three = ROOT / "artifacts/merged/comparisons/three_setting_merge_mtl_summary_20260319.csv"
    if three.exists():
        rows = read_csv(three)
        perf = []
        for r in rows:
            if r.get("setting") == "7_task" and r.get("method") not in {"base_model", "best_single_task"}:
                deltas = [float(r[k]) for k in r if k.endswith("_interference_delta") and r[k] not in ("", "nan")]
                cls = [float(r[k]) for k in r if k.endswith("_interference_delta") and not k.startswith("asr_") and r[k] not in ("", "nan")]
                perf.append(
                    {
                        "Method": canonical_method(r.get("method", "")),
                        "Mean Recovery": r.get("mean_interference_delta", ""),
                        "Min Recovery": r.get("min_interference_delta", ""),
                        "ASR Recovery": r.get("asr_wer_interference_delta", ""),
                        "Classification Recovery": sum(cls) / len(cls) if cls else "",
                        "Modularity": "adapter reuse / recomposition" if "merge" in r.get("method", "") else "fixed shared adapter",
                    }
                )
        save_table("performance_modularity_tradeoff", perf, ["Method", "Mean Recovery", "Min Recovery", "ASR Recovery", "Classification Recovery", "Modularity"], [three])
        draw_grouped_bars("mtl_vs_merge_recovery_heatmap", perf, "Method", ["Mean Recovery", "Min Recovery", "ASR Recovery", "Classification Recovery"], three)
    cont = ROOT / "artifacts/continual_suite/thesis_continual_extension_supermerge_v1/continual_final_comparison.csv"
    if cont.exists():
        rows = read_csv(cont)
        comp = []
        for r in rows:
            comp.append(
                {
                    "Path": r.get("path_id", ""),
                    "Added Task": canonical_task(r.get("added_task", "")),
                    "Seen Mean Recovery": r.get("seen_mean_delta", ""),
                    "Seen Min Recovery": r.get("seen_min_delta", ""),
                    "New Task Recovery": r.get("new_task_delta", ""),
                    "Prior Avg Recovery": r.get("prior_avg_delta", ""),
                    "Avg Prior Forgetting": r.get("avg_prior_forgetting", ""),
                }
            )
        save_table("continual_extension_summary", comp, list(comp[0].keys()) if comp else None, [cont])
        draw_grouped_bars("continual_extension_recovery_heatmap", comp, "Path", ["Seen Mean Recovery", "Seen Min Recovery", "New Task Recovery", "Prior Avg Recovery"], cont)


def copy_existing_figures() -> None:
    fig_map = {
        "cross_task_interference_delta_existing.png": ROOT / "analysis/plots/chapter5/cross_task_recovery_matrix_7task_heatmap.png",
        "cross_task_interference_raw_existing.png": ROOT / "analysis/plots/chapter5/cross_task_absolute_matrix_7task_heatmap.png",
        "task_vector_norms_existing.png": ROOT / "analysis/plots/chapter5/task_vector_norms_7task.png",
        "task_vector_cosine_existing.png": ROOT / "analysis/plots/chapter5/task_vector_cosine_7task.png",
        "adapter_magnitude_by_layer_existing.png": ROOT / "analysis/plots/chapter5/adapter_magnitude_by_layer_7task.png",
        "adapter_magnitude_by_module_existing.png": ROOT / "analysis/plots/chapter5/component_statistics_7task.png",
        "layer_task_heatmap_existing.png": ROOT / "analysis/plots/chapter5/adapter_magnitude_by_layer_7task.png",
        "asr_scalar_sweep_existing.png": ROOT / "analysis/results/asr_scalar_sweep_old/asr_only_scalar_sweep_plot.png",
        "asr_magnitude_calibration_existing.png": ROOT / "analysis/results/asr_mechanism/alphaeff_vs_wer_mechanism_plot.png",
        "merge_comparison_by_task_existing.png": ROOT / "analysis/merge_comparison/merge_interference_by_task_method.png",
        "minor_ties_dare_comparison_existing.png": ROOT / "artifacts/merged/interference_comparison_uniform_ties_adamerging_supermerge_dare_emotion_intent_kws_langid_test.png",
    }
    for dst, src in fig_map.items():
        copy_if_exists(src, dst)
    # Copy continual figures if already produced by the suite.
    for src in (ROOT / "artifacts/continual_suite/thesis_continual_extension_supermerge_v1").glob("*.png"):
        copy_if_exists(src, f"continual_{src.name}")


def mark_missing(section: str, thesis_section: str, missing: str, closest: str, raw: str, command: str, expense: str) -> None:
    missing_items.append(
        {
            "Thesis section": thesis_section,
            "Missing task/method/metric": missing,
            "Closest existing files": closest,
            "Raw data appear to exist": raw,
            "Suggested command/script/config": command,
            "Expense category": expense,
        }
    )
    artifact_status.append({"artifact": section, "status": "missing", "sources": closest, "tasks": "", "methods": "", "notes": missing})


def validate_expected() -> None:
    expected_paths = {
        "SpeechQA/MMSU category tables": ROOT / "analysis/results/mmsu_spoken_qa/category_rollup_mean_delta_vs_base.csv",
        "Continual supermerge final comparison": ROOT / "artifacts/continual_suite/thesis_continual_extension_supermerge_v1/continual_final_comparison.csv",
        "Sequential fine-tuning forgetting": ROOT / "thesis_artifacts/tables/sequential_ft_continual_mtl_final_comparison.csv",
        "Reoptimised upper bound": ROOT / "artifacts/continual_suite/thesis_continual_extension_supermerge_v1/reoptimised_upper_bound.csv",
        "Confusion matrices": ROOT / "thesis_artifacts/tables/confusion_matrix_inventory.csv",
        "Coefficient plots": ROOT / "thesis_artifacts/tables/layerwise_coefficient_summary.csv",
        "Prompt templates": ROOT / "thesis_artifacts/tables/speechqa_prompt_sweep.csv",
    }
    for name, path in expected_paths.items():
        if not path.exists():
            mark_missing(
                name,
                "Main/appendix",
                name,
                rel(path.parent) if path.parent.exists() else "none",
                "unknown" if not path.parent.exists() else "possibly",
                "Inspect scripts under scripts/analysis or scripts/merging and run only parse/plot commands where available.",
                "cheap" if "plots" in name.lower() or "templates" in name.lower() else "medium",
            )

    # Detect duplicate/conflicting primary single-task values in all_experiments.
    rows = load_all_experiments()
    vals = defaultdict(set)
    for r in rows:
        if r.get("is_primary_metric") == "True" and r.get("experiment_type") == "single_task":
            vals[(r.get("task"), r.get("metric_name"))].add(r.get("metric_value"))
    for key, v in vals.items():
        if len(v) > 1:
            inconsistencies.append(f"- Single-task primary metric has multiple values for {key}: {sorted(v)}. Prefer curated comparison CSV for thesis tables unless provenance requires otherwise.")


def write_manifests(total_files: int, result_files: int) -> None:
    lines = ["# Results Manifest", ""]
    for a in artifact_status:
        lines += [
            f"## {a['artifact']}",
            f"- status: {a['status']}",
            f"- sources: {a.get('sources', '')}",
            f"- tasks: {a.get('tasks', '')}",
            f"- methods: {a.get('methods', '')}",
            f"- notes: {a.get('notes', '')}",
            "",
        ]
    (MANIFESTS / "results_manifest.md").write_text("\n".join(lines), encoding="utf-8")

    miss_lines = ["# Missing Results", ""]
    if not missing_items:
        miss_lines.append("No missing results detected by the packager.")
    for m in missing_items:
        miss_lines += [
            f"## {m['Missing task/method/metric']}",
            f"- thesis section: {m['Thesis section']}",
            f"- closest existing files: {m['Closest existing files']}",
            f"- raw data appear to exist: {m['Raw data appear to exist']}",
            f"- suggested command/script/config: `{m['Suggested command/script/config']}`",
            f"- expense category: {m['Expense category']}",
            "",
        ]
    (MANIFESTS / "missing_results.md").write_text("\n".join(miss_lines), encoding="utf-8")

    cmd_lines = ["# Commands To Run", "", "Suggested only. These were not run.", ""]
    for m in missing_items:
        cmd_lines.append(f"- {m['Missing task/method/metric']} ({m['Expense category']}): `{m['Suggested command/script/config']}`")
    if not missing_items:
        cmd_lines.append("- None.")
    (MANIFESTS / "commands_to_run.md").write_text("\n".join(cmd_lines) + "\n", encoding="utf-8")

    inc_lines = ["# Inconsistencies", ""]
    inc_lines.extend(inconsistencies or ["No inconsistencies detected by canonical-source checks."])
    (MANIFESTS / "inconsistencies.md").write_text("\n".join(inc_lines) + "\n", encoding="utf-8")

    readme = [
        "# Thesis Artifacts",
        "",
        "Packaged experimental artifacts for `main.tex` from existing repository outputs only.",
        "",
        f"- Files scanned: {total_files}",
        f"- Result files indexed: {result_files}",
        f"- Tables generated: {len(generated_tables) // 2}",
        f"- Figure files generated/copied: {len(generated_figures)}",
        f"- Missing/partial artifacts: {len(missing_items)}",
        f"- Inconsistencies: {len(inconsistencies)}",
        "",
        "## Layout",
        "- `tables/`: CSV and booktabs LaTeX tables.",
        "- `figures/`: copied existing figures plus lightweight generated heatmaps in PNG/PDF where possible.",
        "- `manifests/`: discovery plan, source index, alias map, status, missing results, inconsistencies, and suggested commands.",
        "",
        "## Caveats",
        "- No training, inference, or evaluation was run.",
        "- Existing PNG figures are copied as PNG only when no source plotting dependency is available.",
        "- ASR/WER is lower-is-better in absolute tables; recovery uses the thesis ASR convention.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(readme), encoding="utf-8")


def main() -> None:
    os.chdir(ROOT)
    ensure_dirs()
    sections = parse_main()
    write_discovery_plan(sections)
    write_alias_map()
    rows = load_all_experiments()
    generate_setup_tables(rows)
    generate_from_existing_csvs()
    generate_single_task_tables()
    generate_minor_method_comparisons(rows)
    generate_interference_tables()
    generate_summaries_and_figures()
    generate_appendix_inventory()
    generate_sequential_ft_from_continual_mtl()
    copy_existing_figures()
    validate_expected()
    records = discover_files()
    result_files = sum(1 for r in records if r["metrics_found"] or r["inferred_tasks"] or r["file_type"] in {"csv", "json", "jsonl"})
    update_source_index(records)
    write_manifests(len(records), result_files)
    print("THESIS_ARTIFACTS_SUMMARY")
    print(f"files_scanned={len(records)}")
    print(f"result_files_indexed={result_files}")
    print(f"tables_generated={len(generated_tables)//2}")
    print(f"figures_generated={len(generated_figures)}")
    print(f"missing_artifacts={len(missing_items)}")
    print(f"inconsistencies={len(inconsistencies)}")
    print(f"readme={rel(OUT / 'README.md')}")
    for i, m in enumerate(missing_items[:5], start=1):
        print(f"blocking_missing_{i}={m['Missing task/method/metric']} [{m['Expense category']}]")


if __name__ == "__main__":
    main()
