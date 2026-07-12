#!/usr/bin/env python3
"""Build Chapter 6 merge-method comparison artefacts.

The script intentionally uses explicit source artefacts rather than scanning for
"best" runs. Several merge directories contain exploratory reruns, validation
probes, and OOD-only evaluations; hard-coded sources keep the thesis tables
auditable and stable.
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


ROOT = Path(".")
OUT_TABLE_DIR = ROOT / "thesis" / "tables"
OUT_FIG_DIR = ROOT / "thesis" / "figures" / "chapter6"
OUT_AUDIT_DIR = ROOT / "analysis" / "results" / "chapter6"

CORE_TASKS = ["emotion", "intent", "kws", "langid", "speaker_ver", "asr", "vocalsound"]
FOUR_TASKS = ["emotion", "intent", "kws", "langid"]

TASK_METRICS = {
    "emotion": ("macro_f1", "higher", "Emotion F1"),
    "intent": ("accuracy", "higher", "Intent Acc."),
    "kws": ("macro_f1", "higher", "KWS F1"),
    "langid": ("accuracy", "higher", "LangID Acc."),
    "speaker_ver": ("accuracy", "higher", "SpeakerVer Acc."),
    "asr": ("wer", "lower", "ASR WER"),
    "vocalsound": ("accuracy", "higher", "VocalSound Acc."),
    "speech_qa": ("accuracy", "higher", "SpeechQA Acc."),
}

BASELINE_PATHS = {
    task: ROOT / "artifacts" / task / "metrics" / "eval" / "test" / "base_model.json"
    for task in TASK_METRICS
}
BEST_PATHS = {
    task: ROOT / "artifacts" / task / "metrics" / "eval" / "test" / f"best_{task}_adapter.json"
    for task in CORE_TASKS
}

METHOD_LABELS = {
    "uniform_delta": "Uniform task arithmetic",
    "ties": "TIES",
    "dare": "DARE",
    "uniform_scalar_delta": "Scalar-calibrated task arithmetic",
    "weighted_delta_n": "Supervised layer-weighted merge",
    "weighted": "Pairwise model weighting",
    "weighted_delta": "Pairwise delta weighting",
}


FOUR_TASK_SOURCES = [
    {
        "method": "uniform_delta",
        "path": ROOT
        / "artifacts/merged/uniform_delta/emotion_intent_kws_langid/runs/run_20260209_223613/eval_results_test.json",
    },
    {
        "method": "ties",
        "path": ROOT
        / "artifacts/merged/ties/emotion_intent_kws_langid/eval/test/eval_results_merged_ties_emotion_intent_kws_langid_lambda1_test.json",
    },
    {
        "method": "dare",
        "path": ROOT
        / "artifacts/merged/dare/emotion_intent_kws_langid/runs/run_20260211_205213/eval_results_test.json",
    },
    {
        "method": "uniform_scalar_delta",
        "scale": 0.5741747441703835,
        "path": ROOT
        / "artifacts/merged/uniform_scalar_delta/emotion_intent_kws_langid/runs/run_20260213_171510/eval_results_test__subset_3f15039510.json",
    },
    {
        "method": "weighted_delta_n",
        "path": ROOT
        / "artifacts/merged/weighted_delta_n/emotion_intent_kws_langid/runs/run_supermerge_layer_wise_20260215_045019/eval_results_test.json",
    },
]

SIX_TASK_SOURCES = [
    {
        "method": "uniform_delta",
        "path": ROOT
        / "artifacts/merged/uniform_delta/asr_emotion_intent_kws_langid_speaker_ver/eval/test/eval_results_merged_uniform_delta_emotion_intent_kws_langid_speaker_ver_asr_test.json",
    },
    {
        "method": "uniform_scalar_delta",
        "scale": 0.651053423634474,
        "path": ROOT
        / "artifacts/merged/uniform_scalar_delta/asr_emotion_intent_kws_langid_speaker_ver/runs/run_20260215_230158/eval_results_test.json",
    },
    {
        "method": "weighted_delta_n",
        "path": ROOT
        / "artifacts/merged/weighted_delta_n/asr_emotion_intent_kws_langid_speaker_ver/runs/run_supermerge_layer_wise_20260216_060039/eval_results_test.json",
    },
]

SEVEN_TASK_SOURCES = [
    {
        "method": "uniform_delta",
        "path": ROOT
        / "artifacts/merged/uniform_delta/asr_emotion_intent_kws_langid_speaker_ver_vocalsound/runs/run_20260224_105145/eval_results_test.json",
    },
    {
        "method": "ties",
        "path": ROOT
        / "artifacts/merged/ties/asr_emotion_intent_kws_langid_speaker_ver_vocalsound/eval/test/eval_results_merged_ties_emotion_intent_kws_langid_speaker_ver_asr_vocalsound_lambda1_test.json",
    },
    {
        "method": "dare",
        "path": ROOT
        / "artifacts/merged/dare/asr_emotion_intent_kws_langid_speaker_ver_vocalsound/eval/test/eval_results_merged_dare_emotion_intent_kws_langid_speaker_ver_asr_vocalsound_test.json",
    },
    {
        "method": "uniform_scalar_delta",
        "scale": 0.5853879434207931,
        "path": ROOT
        / "artifacts/merged/uniform_scalar_delta/asr_emotion_intent_kws_langid_speaker_ver_vocalsound/runs/run_20260224_163137/eval_results_test.json",
    },
    {
        "method": "weighted_delta_n",
        "path": ROOT
        / "artifacts/merged/weighted_delta_n/asr_emotion_intent_kws_langid_speaker_ver_vocalsound/eval/test/per_task",
        "summary": ROOT
        / "artifacts/merged/weighted_delta_n/asr_emotion_intent_kws_langid_speaker_ver_vocalsound/runs/run_supermerge_layer_wise_20260310_053857/summary.json",
    },
]

SCALAR_SWEEPS = {
    "4-task": ROOT
    / "artifacts/merged/uniform_scalar_delta/emotion_intent_kws_langid/sweeps/sweep_20260213_161514.json",
    "7-task": ROOT
    / "artifacts/merged/uniform_scalar_delta/asr_emotion_intent_kws_langid_speaker_ver_vocalsound/sweeps/sweep_20260224_110452.json",
}

SCALAR_TEST_SOURCES = {
    "4-task": FOUR_TASK_SOURCES[3]["path"],
}

WEIGHTED_DELTA_N_SUMMARIES = {
    "4-task": ROOT
    / "artifacts/merged/weighted_delta_n/emotion_intent_kws_langid/runs/run_supermerge_layer_wise_20260215_045019/summary.json",
    "7-task": ROOT
    / "artifacts/merged/weighted_delta_n/asr_emotion_intent_kws_langid_speaker_ver_vocalsound/runs/run_supermerge_layer_wise_20260310_053857/summary.json",
}

# R_min and R_mean are over the six in-merge tasks only (omitted task excluded).
# Source paths for the seven completed Scalar leave-one-out runs (7-task suite):
#   VocalSound omit: artifacts/merged/uniform_scalar_delta/asr_emotion_intent_kws_langid_speaker_ver/runs/run_20260215_230158/eval_results_test.json
#   Emotion omitted: artifacts/merged/uniform_scalar_delta/asr_intent_kws_langid_speaker_ver_vocalsound/eval/test/...
#   Intent omitted:  artifacts/merged/uniform_scalar_delta/asr_emotion_kws_langid_speaker_ver_vocalsound/eval/test/...
#   KWS omitted:     artifacts/merged/uniform_scalar_delta/asr_emotion_intent_langid_speaker_ver_vocalsound/eval/test/...
#   LangID omitted:  artifacts/merged/uniform_scalar_delta/asr_emotion_intent_kws_speaker_ver_vocalsound/runs/run_20260513_024914/eval_results_test.json
#   SpeakerVer omit: artifacts/merged/uniform_scalar_delta/asr_emotion_intent_kws_langid_vocalsound/eval/test/...
#   ASR omitted:     artifacts/merged/uniform_scalar_delta/emotion_intent_kws_langid_speaker_ver_vocalsound/eval/test/...
# Source paths for Layer-wise per-task LOO (all tasks including the omitted one are evaluated).
# Tuple: (display_label, loo_rows_label, omitted_task_key, per_task_dir)
# display_label: abbreviated form used in the LaTeX table (matches the hand-written Scalar table).
# loo_rows_label: label used in LEAVE_ONE_OUT_ROWS for cross-checking aggregates.
LAYER_WISE_LOO_SOURCES = [
    ("ASR", "ASR",        "asr",         ROOT / "artifacts/merged/weighted_delta_n/emotion_intent_kws_langid_speaker_ver_vocalsound/eval/test/per_task"),
    ("ER",  "Emotion",    "emotion",     ROOT / "artifacts/merged/weighted_delta_n/asr_intent_kws_langid_speaker_ver_vocalsound/eval/test/per_task"),
    ("IC",  "Intent",     "intent",      ROOT / "artifacts/merged/weighted_delta_n/asr_emotion_kws_langid_speaker_ver_vocalsound/eval/test/per_task"),
    ("KWS", "KWS",        "kws",         ROOT / "artifacts/merged/weighted_delta_n/asr_emotion_intent_langid_speaker_ver_vocalsound/eval/test/per_task"),
    ("LID", "LangID",     "langid",      ROOT / "artifacts/merged/weighted_delta_n/asr_emotion_intent_kws_speaker_ver_vocalsound/eval/test/per_task"),
    ("SV",  "SpeakerVer", "speaker_ver", ROOT / "artifacts/merged/weighted_delta_n/asr_emotion_intent_kws_langid_vocalsound/eval/test/per_task"),
    ("VS",  "VocalSound", "vocalsound",  ROOT / "artifacts/merged/weighted_delta_n/asr_emotion_intent_kws_langid_speaker_ver/eval/test/per_task"),
]
LEAVE_ONE_OUT_ROWS = [
    ("VocalSound", "uniform_scalar_delta", 0.670, 1.029),
    ("Emotion", "uniform_scalar_delta", 0.840, 1.071),
    ("Intent", "uniform_scalar_delta", 0.220, 0.983),
    ("KWS", "uniform_scalar_delta", 0.538, 1.002),
    ("LangID", "uniform_scalar_delta", 0.465, 0.966),
    ("SpeakerVer", "uniform_scalar_delta", 0.402, 1.007),
    ("ASR", "uniform_scalar_delta", 0.461, 0.845),
    ("VocalSound", "weighted_delta_n", 0.661, 0.952),
    ("Emotion", "weighted_delta_n", 0.871, 1.060),
    ("Intent", "weighted_delta_n", 0.214, 0.952),
    ("KWS", "weighted_delta_n", 0.717, 1.010),
    ("LangID", "weighted_delta_n", 0.490, 0.997),
    ("SpeakerVer", "weighted_delta_n", 0.289, 0.920),
    ("ASR", "weighted_delta_n", 0.635, 0.884),
]

PAIRWISE_ROWS = [
    ("ASR+Intent", "weighted", 0.965, 1.029),
    ("ASR+Intent", "weighted_delta", 0.933, 1.341),
    ("ASR+SpeakerVer", "weighted_delta", 0.977, 1.338),
    ("Intent+LangID", "weighted", 0.887, 0.937),
    ("Intent+LangID", "weighted_delta", 0.931, 0.931),
    ("KWS+LangID", "weighted", 0.919, 0.925),
    ("KWS+LangID", "weighted_delta", 0.933, 0.962),
]

OOD_ROWS = [
    ("Base model", ROOT / "artifacts/speech_qa/metrics/eval/test/base_model.json"),
    ("Best SpeakerVer adapter", ROOT / "artifacts/speech_qa/metrics/eval/test/best_speaker_ver_adapter.json"),
    (
        r"\texttt{uniform\_delta} 6-task",
        ROOT
        / "artifacts/merged/uniform_delta/asr_emotion_intent_kws_langid_speaker_ver/eval/test/per_task/speech_qa/speech_qa_merged_uniform_delta_asr_emotion_intent_kws_langid_speaker_ver_metrics.json",
    ),
    (
        r"\texttt{uniform\_scalar\_delta} 7-task",
        ROOT
        / "artifacts/merged/uniform_scalar_delta/asr_emotion_intent_kws_langid_speaker_ver_vocalsound/eval/test/per_task/speech_qa/speech_qa_merged_uniform_scalar_delta_emotion_intent_kws_langid_speaker_ver_asr_vocalsound_metrics.json",
    ),
    (
        r"\texttt{weighted\_delta\_n} 7-task",
        ROOT
        / "artifacts/merged/weighted_delta_n/asr_emotion_intent_kws_langid_speaker_ver_vocalsound/eval/test/per_task/speech_qa/speech_qa_merged_weighted_delta_n_emotion_intent_kws_langid_speaker_ver_asr_vocalsound_metrics.json",
    ),
]

WEIGHTED_HELDOUT_HISTORY = (
    ROOT
    / "artifacts/merged/weighted_delta_n/asr_emotion_intent_kws_langid_speaker_ver_vocalsound/runs/run_supermerge_layer_wise_20260310_053857/heldout_metrics_history.csv"
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def assert_exists(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Chapter 6 source artefacts:\n" + "\n".join(missing))


def nested_results(payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw = payload.get("results", payload)
    out: Dict[str, Dict[str, Any]] = {}
    for task, value in raw.items():
        if task not in TASK_METRICS or not isinstance(value, dict):
            continue
        if any(split in value for split in ("test", "validation", "test_other")):
            for split in ("test", "validation", "test_other"):
                if isinstance(value.get(split), dict):
                    out[task] = value[split]
                    break
        else:
            out[task] = value
    return out


def load_baselines() -> tuple[Dict[str, float], Dict[str, float]]:
    base: Dict[str, float] = {}
    best: Dict[str, float] = {}
    for task, path in BASELINE_PATHS.items():
        metric = TASK_METRICS[task][0]
        if path.exists():
            value = load_json(path).get(metric)
            if isinstance(value, (int, float)):
                base[task] = float(value)
    for task, path in BEST_PATHS.items():
        metric = TASK_METRICS[task][0]
        value = load_json(path).get(metric)
        if isinstance(value, (int, float)):
            best[task] = float(value)
    return base, best


BASE, BEST = load_baselines()


def recovery(task: str, value: Optional[float]) -> Optional[float]:
    if value is None or task not in BASE or task not in BEST:
        return None
    metric, orientation, _ = TASK_METRICS[task]
    del metric
    base = BASE[task]
    best = BEST[task]
    denom = best - base if orientation == "higher" else base - best
    if abs(denom) < 1e-12:
        return None
    return (value - base) / denom if orientation == "higher" else (base - value) / denom


def task_value_and_recovery(path: Path, task: str) -> tuple[Optional[float], Optional[float]]:
    payload = load_json(path)
    results = nested_results(payload)
    row = results.get(task, {})
    metric = TASK_METRICS[task][0]
    value = row.get(metric)
    if value is None:
        value = row.get(f"eval_{metric}")
    value_f = float(value) if isinstance(value, (int, float)) else None
    rec = row.get("interference_delta")
    if rec is None:
        rec = recovery(task, value_f)
    return value_f, float(rec) if isinstance(rec, (int, float)) else None


def source_summary(path: Path, tasks: Sequence[str]) -> Dict[str, Any]:
    recs: List[float] = []
    values: Dict[str, Optional[float]] = {}
    recoveries: Dict[str, Optional[float]] = {}
    for task in tasks:
        if path.is_dir():
            matches = sorted((path / task).glob("*.json"))
            if not matches:
                value, rec = None, None
            else:
                payload = load_json(matches[0])
                metric = TASK_METRICS[task][0]
                value = payload.get(metric)
                value_f = float(value) if isinstance(value, (int, float)) else None
                rec = payload.get("interference_delta")
                if rec is None:
                    rec = recovery(task, value_f)
                value = value_f
        else:
            value, rec = task_value_and_recovery(path, task)
        values[task] = value
        recoveries[task] = rec
        if rec is not None:
            recs.append(rec)
    return {
        "values": values,
        "recoveries": recoveries,
        "min_recovery": min(recs) if recs else None,
        "mean_recovery": mean(recs) if recs else None,
    }


def fmt_float(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def fmt_metric(task: str, value: Optional[float]) -> str:
    if value is None:
        return "--"
    if task == "asr":
        return f"{100.0 * value:.2f}"
    return f"{100.0 * value:.1f}"


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, latex_escape(method))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_latex_table(
    path: Path,
    *,
    label: str,
    caption: str,
    columns: Sequence[str],
    aligns: str,
    rows: Sequence[Sequence[str]],
    notes: Optional[str] = None,
) -> None:
    body: List[str] = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{{aligns}}}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        body.append(" & ".join(row) + r" \\")
    body.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
        ]
    )
    if notes:
        body.append(rf"\vspace{{2pt}}\par\footnotesize{{{notes}}}")
    body.append(r"\end{table}")
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def build_four_task_table() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    latex_rows: List[List[str]] = []
    compact_labels = {
        "uniform_delta": "Uniform arithmetic",
        "ties": "TIES",
        "dare": "DARE",
        "uniform_scalar_delta": "Scalar-calibrated",
        "weighted_delta_n": "Layer-weighted",
    }
    for source in FOUR_TASK_SOURCES:
        summary = source_summary(source["path"], FOUR_TASKS)
        row = {
            "method": source["method"],
            "scale": source.get("scale", ""),
            "min_recovery": summary["min_recovery"],
            "mean_recovery": summary["mean_recovery"],
            "source": source["path"],
        }
        for task in FOUR_TASKS:
            row[f"{task}_value"] = summary["values"][task]
            row[f"{task}_recovery"] = summary["recoveries"][task]
        rows.append(row)
        latex_rows.append(
            [
                compact_labels[source["method"]],
                fmt_metric("emotion", summary["values"]["emotion"]),
                fmt_metric("intent", summary["values"]["intent"]),
                fmt_metric("kws", summary["values"]["kws"]),
                fmt_metric("langid", summary["values"]["langid"]),
                fmt_float(summary["min_recovery"]),
                fmt_float(summary["mean_recovery"]),
            ]
        )
    write_latex_table(
        OUT_TABLE_DIR / "chapter6_four_task_methods.tex",
        label="tab:ch6-four-task-methods",
        caption=(
            "Preliminary four-task diagnostic comparison. Values are task metrics in "
            "percent; recovery is normalised between the base model and the "
            "corresponding single-task adapter."
        ),
        columns=[
            "Method",
            "Emotion",
            "Intent",
            "KWS",
            "LangID",
            "Min rec.",
            "Mean rec.",
        ],
        aligns="lrrrrrr",
        rows=latex_rows,
    )
    return rows


def build_scale_summary() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for source in SEVEN_TASK_SOURCES:
        summary = source_summary(source["path"], CORE_TASKS)
        rows.append(
            {
                "method": source["method"],
                "scale": source.get("scale", ""),
                "min_recovery": summary["min_recovery"],
                "mean_recovery": summary["mean_recovery"],
                "source": source["path"],
            }
        )

    latex_rows = [
        [
            method_label(row["method"]),
            fmt_float(float(row["scale"]), 3) if row["scale"] != "" else "--",
            fmt_float(row["min_recovery"]),
            fmt_float(row["mean_recovery"]),
        ]
        for row in rows
    ]
    write_latex_table(
        OUT_TABLE_DIR / "chapter6_method_scale_summary.tex",
        label="tab:ch6-method-scale-summary",
        caption=(
            "Main seven-task simultaneous merge comparison. Recovery is computed "
            "over the seven trained source tasks."
        ),
        columns=["Method", "Scale", "Min rec.", "Mean rec."],
        aligns="lrrr",
        rows=latex_rows,
    )
    return rows


def build_seven_task_detail() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    latex_rows: List[List[str]] = []
    compact = {
        "uniform_delta": "Uniform",
        "ties": "TIES",
        "dare": "DARE",
        "uniform_scalar_delta": "Scalar",
        "weighted_delta_n": "Layer-wise",
    }
    for source in SEVEN_TASK_SOURCES:
        method = source["method"]
        summary = source_summary(source["path"], CORE_TASKS)
        recs = summary["recoveries"]
        out = {
            "method": method,
            "min_recovery": summary["min_recovery"],
            "mean_recovery": summary["mean_recovery"],
            "source": source["path"],
        }
        for task in CORE_TASKS:
            out[f"{task}_recovery"] = recs[task]
        rows.append(out)
        latex_rows.append(
            [
                compact[method],
                fmt_float(recs["asr"]),
                fmt_float(recs["emotion"]),
                fmt_float(recs["intent"]),
                fmt_float(recs["kws"]),
                fmt_float(recs["langid"]),
                fmt_float(recs["speaker_ver"]),
                fmt_float(recs["vocalsound"]),
                fmt_float(out["min_recovery"]),
                fmt_float(out["mean_recovery"]),
            ]
        )
    write_latex_table(
        OUT_TABLE_DIR / "chapter6_seven_task_recovery.tex",
        label="tab:ch6-seven-task-recovery",
        caption="Per-task recovery in the main seven-task simultaneous merge.",
        columns=[
            "Method",
            "ASR",
            "ER",
            "IC",
            "KWS",
            "LID",
            "SV",
            "VS",
            r"$\boldsymbol{R_{\min}}$",
            r"$\boldsymbol{\bar{R}}$",
        ],
        aligns="lrrrrrrrrr",
        rows=latex_rows,
    )
    return rows


def best_sweep_run(path: Path) -> Dict[str, Any]:
    payload = load_json(path)
    runs = payload.get("runs", [])
    if not runs:
        return {}
    return max(runs, key=lambda row: float(row.get("score", float("-inf"))))


def build_scalar_sweep_table() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    latex_rows: List[List[str]] = []
    for setting, path in SCALAR_SWEEPS.items():
        payload = load_json(path)
        best = best_sweep_run(path)
        post = payload.get("post_sweep_eval", {})
        test_min = post.get("score_details", {}).get("min_interference_delta")
        test_mean = post.get("score_details", {}).get("mean_interference_delta")
        if setting in SCALAR_TEST_SOURCES:
            tasks = FOUR_TASKS if setting == "4-task" else CORE_TASKS
            summary = source_summary(SCALAR_TEST_SOURCES[setting], tasks)
            test_min = summary["min_recovery"]
            test_mean = summary["mean_recovery"]
        row = {
            "setting": setting,
            "budget": payload.get("search", {}).get("budget"),
            "selected_scale": best.get("params", {}).get("scale"),
            "validation_min": best.get("score_details", {}).get("min_interference_delta"),
            "validation_mean": best.get("score_details", {}).get("mean_interference_delta"),
            "test_min": test_min,
            "test_mean": test_mean,
            "source": path,
        }
        rows.append(row)
        latex_rows.append(
            [
                setting,
                str(row["budget"]),
                fmt_float(row["selected_scale"], 3),
                fmt_float(row["validation_min"]),
                fmt_float(row["validation_mean"]),
                fmt_float(row["test_min"]),
                fmt_float(row["test_mean"]),
            ]
        )
    write_latex_table(
        OUT_TABLE_DIR / "chapter6_scalar_sweeps.tex",
        label="tab:ch6-scalar-sweeps",
        caption=(
            "Scalar calibration selected on validation subsets and evaluated on the "
            "corresponding test tasks."
        ),
        columns=["Setting", "Budget", "Scale", "Val. min", "Val. mean", "Test min", "Test mean"],
        aligns="lrrrrrr",
        rows=latex_rows,
    )
    return rows


def build_weighted_coeff_table() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    latex_rows: List[List[str]] = []
    for setting, path in WEIGHTED_DELTA_N_SUMMARIES.items():
        payload = load_json(path)
        provenance = payload.get("params", {}).get("optimizer", {}).get("provenance", {})
        tasks = provenance.get("tasks", [])
        coeffs = provenance.get("final_default_task_coefficients") or provenance.get("final_task_coefficients", [])
        source_tasks = payload.get("source_tasks") or []
        if len(source_tasks) == len(coeffs):
            tasks = source_tasks
        best_step = provenance.get("heldout_best_step")
        best_score = provenance.get("heldout_best_selection_score")
        for task, coeff in zip(tasks, coeffs):
            row = {
                "setting": setting,
                "task": task,
                "coefficient": float(coeff),
                "heldout_best_step": best_step,
                "heldout_best_score": best_score,
                "source": path,
            }
            rows.append(row)
            latex_rows.append(
                [
                    setting,
                    latex_escape(task),
                    fmt_float(row["coefficient"], 3),
                    str(best_step) if best_step is not None else "--",
                    fmt_float(best_score, 3) if best_score is not None else "--",
                ]
            )
    write_latex_table(
        OUT_TABLE_DIR / "chapter6_weighted_coefficients.tex",
        label="tab:ch6-weighted-coefficients",
        caption=(
            "Default task coefficients selected by supervised layer-wise coefficient "
            "optimisation. The actual merge also uses layer-specific coefficients "
            "and a learned per-layer magnitude."
        ),
        columns=["Setting", "Task", "Default coeff.", "Held-out step", "Held-out score"],
        aligns="llrrr",
        rows=latex_rows,
    )
    return rows


def build_leave_one_out_table() -> List[Dict[str, Any]]:
    rows = [
        {
            "omitted_task": omitted,
            "method": method,
            "min_recovery": min_rec,
            "mean_recovery": mean_rec,
        }
        for omitted, method, min_rec, mean_rec in LEAVE_ONE_OUT_ROWS
    ]
    by_task: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(row["omitted_task"], {})[row["method"]] = row
    task_order = ["VocalSound", "Emotion", "Intent", "KWS", "LangID", "SpeakerVer", "ASR"]

    def fmt_best(value: Optional[float], best: Optional[float]) -> str:
        rendered = fmt_float(value)
        if value is not None and best is not None and abs(value - best) < 1e-12:
            return rf"\textbf{{{rendered}}}"
        return rendered

    latex_rows = [
        [
            task,
            fmt_best(scalar["min_recovery"], best_min),
            fmt_float(scalar["mean_recovery"]),
            fmt_best(layer["min_recovery"], best_min),
            fmt_float(layer["mean_recovery"]),
        ]
        for task in task_order
        for scalar in [by_task[task]["uniform_scalar_delta"]]
        for layer in [by_task[task]["weighted_delta_n"]]
        for best_min in [max(scalar["min_recovery"], layer["min_recovery"])]
    ]
    body = [
        r"\begin{table}[H]",
        r"\centering",
        r"\thesistablesetup",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Omitted & Scalar $R_{\min}$ & Scalar $\bar{R}$ & Layer-wise $R_{\min}$ & Layer-wise $\bar{R}$ \\",
        r"\midrule",
    ]
    body.extend(" & ".join(row) + r" \\" for row in latex_rows)
    body.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Scalar versus supervised layer-wise leave-one-out recovery. "
                r"Each row omits one source adapter; $R_{\min}$ and $\bar{R}$ are "
                r"computed over the six in-merge tasks only. Bold marks the better "
                r"worst-task recovery for each omitted-task configuration.}"
            ),
            r"\label{tab:ch6-leave-one-out-comparison}",
            r"\end{table}",
        ]
    )
    (OUT_TABLE_DIR / "chapter6_leave_one_out_comparison.tex").write_text(
        "\n".join(body) + "\n", encoding="utf-8"
    )
    return rows


def build_layer_wise_loo_table() -> List[Dict[str, Any]]:
    """Build full per-task Layer-wise LOO table, parallel to the hand-written Scalar table."""
    task_keys  = ["asr", "emotion", "intent", "kws", "langid", "speaker_ver", "vocalsound"]

    # Hardcoded aggregate reference values from LEAVE_ONE_OUT_ROWS for sanity check.
    # Keys are the loo_rows_label field (e.g. "Emotion", "VocalSound", "ASR"...).
    lw_reference = {
        label: (r_min, r_bar)
        for label, method, r_min, r_bar in LEAVE_ONE_OUT_ROWS
        if method == "weighted_delta_n"
    }

    rows: List[Dict[str, Any]] = []
    body_rows: List[str] = []

    for display_label, loo_rows_label, omitted_task, per_task_dir in LAYER_WISE_LOO_SOURCES:
        in_merge_tasks = [t for t in task_keys if t != omitted_task]
        summary = source_summary(per_task_dir, in_merge_tasks)
        omitted_rec = source_summary(per_task_dir, [omitted_task])["recoveries"].get(omitted_task)

        # Sanity-check against hardcoded reference.
        ref_min, ref_bar = lw_reference[loo_rows_label]
        computed_min = summary["min_recovery"]
        computed_bar = summary["mean_recovery"]
        if computed_min is not None and abs(computed_min - ref_min) > 0.002:
            print(f"  WARNING {display_label} LOO: R_min {computed_min:.3f} vs reference {ref_min:.3f}")
        if computed_bar is not None and abs(computed_bar - ref_bar) > 0.002:
            print(f"  WARNING {display_label} LOO: R_bar {computed_bar:.3f} vs reference {ref_bar:.3f}")

        row: Dict[str, Any] = {
            "omitted_task": display_label,
            "omitted_task_key": omitted_task,
            "min_recovery": computed_min,
            "mean_recovery": computed_bar,
        }
        for t in task_keys:
            row[f"{t}_recovery"] = summary["recoveries"].get(t) if t != omitted_task else omitted_rec
        rows.append(row)

        # LaTeX row: italicise the omitted task's entry.
        cells = [display_label]
        for t in task_keys:
            if t == omitted_task:
                rec = omitted_rec
                rendered = fmt_float(rec) if rec is not None else "--"
                # Negative recoveries: keep minus sign inside \textit
                cells.append(rf"\textit{{{rendered}}}")
            else:
                cells.append(fmt_float(summary["recoveries"].get(t)))
        cells.append(fmt_float(computed_min))
        cells.append(fmt_float(computed_bar))
        body_rows.append(" & ".join(cells) + r" \\")

    # Build 7-task reference row (Layer-wise 7T from SEVEN_TASK_SOURCES).
    lw7_source = next(s for s in SEVEN_TASK_SOURCES if s["method"] == "weighted_delta_n")
    lw7 = source_summary(lw7_source["path"], task_keys)

    ref_header = "Layer-wise (7T)"
    ref_cells = [ref_header]
    for t in task_keys:
        ref_cells.append(fmt_float(lw7["recoveries"].get(t)))
    ref_cells.append(fmt_float(lw7["min_recovery"]))
    ref_cells.append(fmt_float(lw7["mean_recovery"]))

    lines: List[str] = [
        r"\begin{table}[H]",
        r"\centering",
        r"\thesiscompacttablesetup",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccccc@{}}",
        r"\toprule",
        r"\textbf{Omitted} & \textbf{ASR} & \textbf{ER} & \textbf{IC} & \textbf{KWS} & \textbf{LID} & \textbf{SV} & \textbf{VS} & $\boldsymbol{R_{\min}}$ & $\boldsymbol{\bar{R}}$ \\",
        r"\midrule",
        " & ".join(ref_cells) + r" \\",
        r"\midrule",
        r"\multicolumn{10}{l}{\textit{One source adapter omitted}} \\",
        r"\addlinespace[1pt]",
    ]
    lines.extend(body_rows)
    lines.extend([
        r"\bottomrule",
        r"\end{tabular*}",
        (
            r"\caption{Layer-wise leave-one-out recovery. Each row omits one source adapter; "
            r"italicised entries are the omitted task's recovery from the remaining six adapters, "
            r"and $R_{\min}$ and $\bar{R}$ exclude the omitted task.}"
        ),
        r"\label{tab:ch6-leave-one-out-layerwise}",
        r"\end{table}",
    ])
    (OUT_TABLE_DIR / "chapter6_leave_one_out_layerwise.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return rows


def build_pairwise_table() -> List[Dict[str, Any]]:
    rows = [
        {"pair": pair, "method": method, "min_recovery": min_rec, "mean_recovery": mean_rec}
        for pair, method, min_rec, mean_rec in PAIRWISE_ROWS
    ]
    latex_rows = [
        [row["pair"], method_label(row["method"]), fmt_float(row["min_recovery"]), fmt_float(row["mean_recovery"])]
        for row in rows
    ]
    write_latex_table(
        OUT_TABLE_DIR / "chapter6_pairwise_weighting.tex",
        label="tab:ch6-pairwise-weighting",
        caption=(
            "Selected pairwise weighting results. The strong pairwise recoveries show "
            "that coefficient choice is important even before scaling to many tasks."
        ),
        columns=["Task pair", "Method", "Min rec.", "Mean rec."],
        aligns="llrr",
        rows=latex_rows,
    )
    return rows


def build_ood_table() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    latex_rows: List[List[str]] = []
    for label, path in OOD_ROWS:
        payload = load_json(path)
        row = {
            "model": label,
            "accuracy": payload.get("accuracy"),
            "exact_match": payload.get("exact_match"),
            "f1": payload.get("f1"),
            "source": path,
        }
        rows.append(row)
        latex_rows.append(
            [
                label,
                fmt_metric("speech_qa", row["accuracy"]),
                fmt_float(row["exact_match"], 2),
                fmt_float(row["f1"], 2),
            ]
        )
    write_latex_table(
        OUT_TABLE_DIR / "chapter6_speechqa_probe.tex",
        label="tab:ch6-speechqa-probe",
        caption=(
            "SpeechQA/MMSU OOD probe. SpeechQA is not a trained source task in the "
            "merge experiments, so these results test preservation outside the supervised "
            "suite rather than recovery of an eighth adapter."
        ),
        columns=["Model", "Acc.", "Exact match", "F1"],
        aligns="lrrr",
        rows=latex_rows,
    )
    return rows


def build_coverage_table() -> None:
    rows = [
        [
            "Uniform arithmetic",
            r"Fixed average coefficients, \(\lambda_t=1/|\mathcal{S}|\).",
            "Uncalibrated baseline.",
        ],
        [
            "Scalar-calibrated",
            r"Bayesian search over \(\gamma\sum_t\tau_t\).",
            "Tests whether global update magnitude is the main failure mode.",
        ],
        [
            "Layer-weighted",
            "Gradient-optimised layer-wise simplex weights with learned per-layer magnitude.",
            "Main coefficient-optimised merge.",
        ],
        [
            "TIES",
            "Sparse delta-space merge",
            "Diagnostic conflict-aware baseline.",
        ],
        [
            "DARE",
            "Sparse delta-space merge",
            "Diagnostic sparsity baseline.",
        ],
    ]
    write_latex_table(
        OUT_TABLE_DIR / "chapter6_coverage.tex",
        label="tab:ch6-coverage",
        caption="Merge methods and thesis-facing names used in Chapter 6.",
        columns=["Method", "Parameterisation", "Role"],
        aligns=r"p{0.22\textwidth}p{0.39\textwidth}p{0.31\textwidth}",
        rows=rows,
    )


def _svg_to_png(svg: str, dst: Path) -> None:
    mvg_path = dst.with_suffix(".mvg")
    mvg_path.write_text(svg, encoding="utf-8")
    subprocess.run(["convert", f"mvg:{mvg_path}", str(dst)], check=True)
    mvg_path.unlink(missing_ok=True)


def _line_plot_svg(
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    series: Sequence[tuple[str, Sequence[tuple[float, float]], str]],
    selected_x: Optional[float] = None,
    selected_label: Optional[str] = None,
    width: int = 1080,
    height: int = 640,
) -> str:
    margin_l, margin_r, margin_t, margin_b = 95, 35, 70, 85
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    xs = [x for _, points, _ in series for x, _ in points]
    ys = [y for _, points, _ in series for _, y in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys + [0.0]), max(ys + [1.0])
    y_pad = (y_max - y_min) * 0.08 or 0.1
    y_min -= y_pad
    y_max += y_pad

    def sx(x: float) -> float:
        return margin_l + (x - x_min) / (x_max - x_min) * plot_w if x_max != x_min else margin_l

    def sy(y: float) -> float:
        return margin_t + (y_max - y) / (y_max - y_min) * plot_h

    def esc(text: str) -> str:
        return text.replace("\\", "\\\\").replace('"', '\\"')

    out = [
        "push graphic-context",
        f"viewbox 0 0 {width} {height}",
        f"fill #ffffff rectangle 0,0 {width},{height}",
        "font Arial",
        f"fill #111827 font-size 24 text-anchor middle text {width/2:.1f},36 \"{esc(title)}\"",
        f"fill #ffffff stroke #111827 stroke-width 1.4 rectangle {margin_l},{margin_t} {margin_l+plot_w},{margin_t+plot_h}",
    ]
    for i in range(6):
        y = y_min + (y_max - y_min) * i / 5
        py = sy(y)
        out.append(f"stroke #e5e7eb stroke-width 1 line {margin_l},{py:.1f} {margin_l+plot_w},{py:.1f}")
        out.append(f"fill #374151 stroke none font-size 16 text-anchor end text {margin_l-12},{py+5:.1f} \"{y:.2f}\"")
    for i in range(6):
        x = x_min + (x_max - x_min) * i / 5
        px = sx(x)
        out.append(f"stroke #111827 stroke-width 1 line {px:.1f},{margin_t+plot_h} {px:.1f},{margin_t+plot_h+6}")
        out.append(f"fill #374151 stroke none font-size 16 text-anchor middle text {px:.1f},{margin_t+plot_h+30} \"{x:.2f}\"")
    zero_y = sy(0.0)
    out.append(f"stroke #6b7280 stroke-width 1.2 line {margin_l},{zero_y:.1f} {margin_l+plot_w},{zero_y:.1f}")
    for label, points, color in series:
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            out.append(f"stroke {color} stroke-width 4 line {sx(x1):.1f},{sy(y1):.1f} {sx(x2):.1f},{sy(y2):.1f}")
        for x, y in points:
            cx, cy, r = sx(x), sy(y), 5
            out.append(f"fill {color} stroke #ffffff stroke-width 2 circle {cx:.1f},{cy:.1f} {cx+r:.1f},{cy:.1f}")
    if selected_x is not None:
        px = sx(selected_x)
        out.append(f"stroke #111827 stroke-width 1.4 line {px:.1f},{margin_t} {px:.1f},{margin_t+plot_h}")
        if selected_label:
            out.append(f"fill #111827 stroke none font-size 15 text-anchor start text {px+8:.1f},{margin_t+22} \"{esc(selected_label)}\"")
    legend_x = margin_l + 18
    legend_y = margin_t + 22
    for idx, (label, _, color) in enumerate(series):
        y = legend_y + idx * 28
        out.append(f"stroke {color} stroke-width 4 line {legend_x},{y} {legend_x+34},{y}")
        out.append(f"fill #111827 stroke none font-size 16 text-anchor start text {legend_x+44},{y+5} \"{esc(label)}\"")
    out.append(f"fill #111827 stroke none font-size 18 text-anchor middle text {width/2:.1f},{height-24} \"{esc(xlabel)}\"")
    out.append(f"fill #111827 stroke none font-size 18 text-anchor middle text 25,55 \"{esc(ylabel)}\"")
    out.append("pop graphic-context")
    return "\n".join(out)


def generate_figures() -> None:
    colours = {"min": "#1f77b4", "mean": "#d55e00", "geo": "#009e73"}
    for setting, path in SCALAR_SWEEPS.items():
        payload = load_json(path)
        runs = sorted(payload.get("runs", []), key=lambda row: float(row.get("params", {}).get("scale", 0.0)))
        min_points = [
            (float(row["params"]["scale"]), float(row.get("score_details", {}).get("min_interference_delta", 0.0)))
            for row in runs
            if "scale" in row.get("params", {})
        ]
        mean_points = [
            (float(row["params"]["scale"]), float(row.get("score_details", {}).get("mean_interference_delta", 0.0)))
            for row in runs
            if "scale" in row.get("params", {})
        ]
        best = best_sweep_run(path)
        selected = best.get("params", {}).get("scale")
        svg = _line_plot_svg(
            title=f"{setting} Scalar Calibration",
            xlabel="Global scale",
            ylabel="Validation recovery",
            series=[("Worst task", min_points, colours["min"]), ("Mean", mean_points, colours["mean"])],
            selected_x=float(selected) if selected is not None else None,
            selected_label=f"selected {float(selected):.3f}" if selected is not None else None,
        )
        _svg_to_png(svg, OUT_FIG_DIR / f"chapter6_scalar_sweep_{setting.replace('-', '').lower()}.png")

    rows: List[Dict[str, str]] = []
    with WEIGHTED_HELDOUT_HISTORY.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    step = [(float(row["update_step"]), float(row["selection_score"])) for row in rows]
    min_points = [(float(row["update_step"]), float(row["min_interference_delta"])) for row in rows]
    mean_points = [(float(row["update_step"]), float(row["arithmetic_mean_interference_delta"])) for row in rows]
    best_row = max(rows, key=lambda row: float(row["selection_score"]))
    svg = _line_plot_svg(
        title="Seven-task Layer-weighted Selection",
        xlabel="Optimisation update",
        ylabel="Held-out recovery",
        series=[
            ("Geometric mean", step, colours["geo"]),
            ("Worst task", min_points, colours["min"]),
            ("Mean", mean_points, colours["mean"]),
        ],
        selected_x=float(best_row["update_step"]),
        selected_label=f"selected step {int(float(best_row['update_step']))}",
    )
    _svg_to_png(svg, OUT_FIG_DIR / "chapter6_weighted_delta_n_heldout.png")


def write_audit(name: str, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields and key not in {"values", "recoveries"}:
                fields.append(key)
    write_csv(OUT_AUDIT_DIR / f"{name}.csv", rows, fields)


def main() -> None:
    OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    sources = [item["path"] for item in FOUR_TASK_SOURCES + SIX_TASK_SOURCES + SEVEN_TASK_SOURCES]
    sources += [*SCALAR_SWEEPS.values(), *SCALAR_TEST_SOURCES.values(), *WEIGHTED_DELTA_N_SUMMARIES.values(), WEIGHTED_HELDOUT_HISTORY]
    sources += [per_task_dir for _, _, _, per_task_dir in LAYER_WISE_LOO_SOURCES]
    assert_exists(sources)

    build_coverage_table()
    write_audit("four_task_methods", build_four_task_table())
    write_audit("method_scale_summary", build_scale_summary())
    write_audit("seven_task_recovery", build_seven_task_detail())
    write_audit("scalar_sweeps", build_scalar_sweep_table())
    write_audit("weighted_coefficients", build_weighted_coeff_table())
    write_audit("leave_one_out", build_leave_one_out_table())
    write_audit("leave_one_out_layerwise", build_layer_wise_loo_table())
    generate_figures()

    print(f"Wrote Chapter 6 tables to {OUT_TABLE_DIR}")
    print(f"Wrote Chapter 6 audit CSVs to {OUT_AUDIT_DIR}")
    print(f"Wrote Chapter 6 figures to {OUT_FIG_DIR}")


if __name__ == "__main__":
    main()
