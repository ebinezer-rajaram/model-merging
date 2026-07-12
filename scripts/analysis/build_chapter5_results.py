#!/usr/bin/env python3
"""Build canonical seven-task Chapter 5 interference and adapter-analysis artifacts.

The script is intentionally analysis-only: it reads existing evaluation JSONs and
LoRA adapter weights, then writes Chapter 5 tables and figures. It excludes the
older speaker_id task and SpeechQA so the outputs match the thesis seven-task
trained suite.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

from _repo import find_repo_root

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
from safetensors import safe_open


ROOT = find_repo_root(__file__)
ANALYSIS_RESULTS = ROOT / "analysis" / "results" / "chapter5"
ANALYSIS_PLOTS = ROOT / "analysis" / "plots" / "chapter5"
THESIS_TABLES = ROOT / "thesis_artifacts" / "tables"
THESIS_FIGURES = ROOT / "thesis_artifacts" / "figures"
THESIS_CH5_FIGURES = ROOT / "thesis" / "figures" / "chapter5"


@dataclass(frozen=True)
class TaskSpec:
    key: str
    label: str
    metric_key: str
    metric_label: str
    higher_is_better: bool
    adapter_name: str


TASKS = [
    TaskSpec("asr", "ASR", "wer", "WER", False, "qwen2_5_omni_lora_asr_100h"),
    TaskSpec("emotion", "ER", "macro_f1", "Macro-F1", True, "qwen2_5_omni_lora_emotion_audio_v2"),
    TaskSpec("intent", "IC", "accuracy", "Accuracy", True, "qwen2_5_omni_lora_intent"),
    TaskSpec("kws", "KWS", "macro_f1", "Macro-F1", True, "qwen2_5_omni_lora_kws"),
    TaskSpec("langid", "LID", "accuracy", "Accuracy", True, "qwen2_5_omni_lora_langid"),
    TaskSpec("speaker_ver", "SV", "accuracy", "Accuracy", True, "qwen2_5_omni_lora_speaker_ver"),
    TaskSpec("vocalsound", "VS", "accuracy", "Accuracy", True, "qwen2_5_omni_lora_vocalsound"),
]

EPS = 1e-8
TASK_LABELS = [task.label for task in TASKS]
ACOUSTIC_TASKS = {"ASR", "KWS", "LID", "SV", "VS"}
BLOCKS = ["global", "audio_tower", "early", "mid", "late", "decoder_all"]
KEY_DIAGNOSTIC_PAIRS = [
    ("IC", "LID"),
    ("KWS", "LID"),
    ("VS", "IC"),
    ("IC", "ASR"),
    ("SV", "ASR"),
    ("VS", "ASR"),
]
PALETTE = {
    "blue": "#2F5F8F",
    "pale_blue": "#D8E8F5",
    "orange": "#B65A2A",
    "pale_orange": "#F3D6C6",
    "green": "#2F7D5C",
    "purple": "#6F5A8E",
    "grey": "#5F6368",
    "light_grey": "#F3F4F6",
}
TASK_FAMILIES = {
    "ASR": "Transcription",
    "ER": "Paralinguistic",
    "IC": "Semantic",
    "KWS": "Lexical",
    "LID": "Lexical",
    "SV": "Speaker",
    "VS": "Event",
}
FAMILY_COLOURS = {
    "Transcription": PALETTE["blue"],
    "Paralinguistic": PALETTE["purple"],
    "Semantic": PALETTE["orange"],
    "Lexical": PALETTE["green"],
    "Speaker": "#4B8C8A",
    "Event": PALETTE["grey"],
}
RECOVERY_CMAP = LinearSegmentedColormap.from_list(
    "recovery_muted",
    [PALETTE["orange"], PALETTE["pale_orange"], "#FFFFFF", PALETTE["pale_blue"], PALETTE["blue"]],
)
RAW_CMAP = LinearSegmentedColormap.from_list(
    "raw_metric_green_red",
    ["#B94A48", "#F2D6D0", "#FFFFFF", "#D9EAD3", "#4F7F5F"],
)

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def ensure_dirs() -> None:
    for path in (ANALYSIS_RESULTS, ANALYSIS_PLOTS, THESIS_TABLES, THESIS_FIGURES, THESIS_CH5_FIGURES):
        path.mkdir(parents=True, exist_ok=True)


def metric_path(eval_task: TaskSpec, adapter_key: str | None) -> Path:
    eval_dir = ROOT / "artifacts" / eval_task.key / "metrics" / "eval" / "test"
    if adapter_key is None:
        return eval_dir / "base_model.json"
    return eval_dir / f"best_{adapter_key}_adapter.json"


def adapter_path(task: TaskSpec) -> Path:
    return ROOT / "artifacts" / task.key / "adapters" / task.adapter_name / "best"


def read_metric(path: Path, metric_key: str) -> float:
    data = json.loads(path.read_text())
    if metric_key not in data:
        raise KeyError(f"{metric_key} missing from {path}")
    return float(data[metric_key])


def oriented(task: TaskSpec, value: float) -> float:
    return value if task.higher_is_better else -value


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def tex_escape(text: object) -> str:
    out = str(text)
    for old, new in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]:
        out = out.replace(old, new)
    return out


def fmt_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if abs(value) >= 10:
            return f"{value:.2f}"
        return f"{value:.3f}"
    text = str(value)
    try:
        number = float(text)
    except ValueError:
        return text
    if math.isnan(number):
        return ""
    if abs(number) >= 10:
        return f"{number:.2f}"
    return f"{number:.3f}"


def fmt_percent(value: float) -> str:
    pct = value * 100 + 1e-9
    if abs(pct) < 10:
        return f"{pct:.2f}%"
    return f"{pct:.1f}%"


def fmt_points(value: float) -> str:
    return f"{value * 100:.3g} pp"


def tex_cell(value: object) -> str:
    text = tex_escape(fmt_cell(value))
    return text.replace("UPARROW", r"$\uparrow$").replace("DOWNARROW", r"$\downarrow$")


def write_tex_table(path: Path, df: pd.DataFrame, caption: str, label: str, landscape: bool = False) -> None:
    align = "l" + "r" * (len(df.columns) - 1)
    resize = label in {"tab:ch5-outgoing-summary", "tab:ch5-incoming-summary"}
    header_map = {
        "Mean off-diagonal recovery": "Mean rec.",
        "Worst off-diagonal recovery": "Worst rec.",
        "Mean outgoing disruption": "Mean disrupt.",
        "Worst outgoing disruption": "Worst disrupt.",
        "Mean incoming recovery": "Mean rec.",
        "Worst incoming recovery": "Worst rec.",
        "Mean incoming disruption": "Mean disrupt.",
        "Worst incoming disruption": "Worst disrupt.",
        "Below-base cells": "Below base",
        "Positive-transfer cells": "Positive",
        "Worst affected task": "Worst task",
        "Worst incoming adapter": "Worst adapter",
        "Spearman rho": "Spearman rho",
    }
    lines = []
    if landscape:
        lines.append(r"\begin{landscape}")
    lines.extend(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{3pt}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\resizebox{\textwidth}{!}{%" if resize else "",
            rf"\begin{{tabular}}{{{align}}}",
            r"\toprule",
            " & ".join(tex_cell(header_map.get(c, c)) for c in df.columns) + r" \\",
            r"\midrule",
        ]
    )
    for _, row in df.iterrows():
        lines.append(" & ".join(tex_cell(row[c]) for c in df.columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if resize:
        lines.append(r"}")
    lines.append(r"\end{table}")
    if landscape:
        lines.append(r"\end{landscape}")
    path.write_text("\n".join(lines) + "\n")


def build_metric_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_rows = []
    single_rows = []
    recovery_rows = []

    for eval_task in TASKS:
        base_value = read_metric(metric_path(eval_task, None), eval_task.metric_key)
        self_value = read_metric(metric_path(eval_task, eval_task.key), eval_task.metric_key)
        base_score = oriented(eval_task, base_value)
        self_score = oriented(eval_task, self_value)
        denom = self_score - base_score
        if abs(denom) < EPS:
            raise ValueError(f"Non-positive recovery denominator for {eval_task.key}")

        single_rows.append(
            {
                "Task": eval_task.label,
                "Metric": eval_task.metric_label + (" DOWNARROW" if not eval_task.higher_is_better else " UPARROW"),
                "Base": fmt_percent(base_value),
                "SingleTask": fmt_percent(self_value),
                "Improvement": fmt_points(self_score - base_score),
            }
        )

        raw_row = {"Task": eval_task.label, "Metric": eval_task.metric_label, "Base": base_value}
        recovery_row = {"Task": eval_task.label, "Base": np.nan}
        for adapter_task in TASKS:
            value = read_metric(metric_path(eval_task, adapter_task.key), eval_task.metric_key)
            raw_row[adapter_task.label] = value
            recovery_row[adapter_task.label] = (oriented(eval_task, value) - base_score) / denom
        raw_rows.append(raw_row)
        recovery_rows.append(recovery_row)

    return pd.DataFrame(single_rows), pd.DataFrame(raw_rows), pd.DataFrame(recovery_rows)


def build_top_effects(recovery_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in recovery_df.iterrows():
        eval_task = row["Task"]
        for adapter_task in TASKS:
            if adapter_task.label == eval_task:
                continue
            value = float(row[adapter_task.label])
            rows.append({"Evaluation Task": eval_task, "Adapter": adapter_task.label, "Recovery": value})
    ranked = pd.DataFrame(rows)
    ranked["Abs Recovery"] = ranked["Recovery"].abs()
    return pd.concat(
        [
            ranked.sort_values("Recovery", ascending=False).head(10).assign(Group="Largest positive off-diagonal"),
            ranked.sort_values("Recovery", ascending=True).head(10).assign(Group="Largest negative off-diagonal"),
        ],
        ignore_index=True,
    )[["Group", "Evaluation Task", "Adapter", "Recovery"]]


def offdiag_recovery_long(recovery_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in recovery_df.iterrows():
        eval_task = row["Task"]
        for adapter_task in TASKS:
            adapter = adapter_task.label
            if adapter == eval_task:
                continue
            recovery = float(row[adapter])
            rows.append(
                {
                    "Adapter": adapter,
                    "Evaluation Task": eval_task,
                    "Recovery": recovery,
                    "Below Base": recovery < 0.0,
                    "Positive Transfer": recovery > 0.0,
                    "Disruption": max(0.0, -recovery),
                }
            )
    return pd.DataFrame(rows)


def build_interference_summary(recovery_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    long = offdiag_recovery_long(recovery_df)

    outgoing = (
        long.groupby("Adapter", sort=False)
        .agg(
            **{
                "Mean off-diagonal recovery": ("Recovery", "mean"),
                "Worst off-diagonal recovery": ("Recovery", "min"),
                "Mean outgoing disruption": ("Disruption", "mean"),
                "Worst outgoing disruption": ("Disruption", "max"),
                "Below-base cells": ("Below Base", "sum"),
                "Positive-transfer cells": ("Positive Transfer", "sum"),
            }
        )
        .reset_index()
    )
    outgoing["Worst affected task"] = outgoing["Adapter"].map(
        long.loc[long.groupby("Adapter")["Recovery"].idxmin()].set_index("Adapter")["Evaluation Task"]
    )
    outgoing["Adapter"] = pd.Categorical(outgoing["Adapter"], categories=TASK_LABELS, ordered=True)
    outgoing = outgoing.sort_values("Adapter").reset_index(drop=True)
    outgoing["Adapter"] = outgoing["Adapter"].astype(str)

    incoming = (
        long.groupby("Evaluation Task", sort=False)
        .agg(
            **{
                "Mean incoming recovery": ("Recovery", "mean"),
                "Worst incoming recovery": ("Recovery", "min"),
                "Mean incoming disruption": ("Disruption", "mean"),
                "Worst incoming disruption": ("Disruption", "max"),
                "Below-base cells": ("Below Base", "sum"),
                "Positive-transfer cells": ("Positive Transfer", "sum"),
            }
        )
        .reset_index()
        .rename(columns={"Evaluation Task": "Task"})
    )
    incoming["Worst incoming adapter"] = incoming["Task"].map(
        long.loc[long.groupby("Evaluation Task")["Recovery"].idxmin()].set_index("Evaluation Task")["Adapter"]
    )
    incoming["Task"] = pd.Categorical(incoming["Task"], categories=TASK_LABELS, ordered=True)
    incoming = incoming.sort_values("Task").reset_index(drop=True)
    incoming["Task"] = incoming["Task"].astype(str)

    return outgoing, incoming


def build_task_compatibility_map(outgoing_df: pd.DataFrame, incoming_df: pd.DataFrame) -> pd.DataFrame:
    outgoing = outgoing_df[
        [
            "Adapter",
            "Mean outgoing disruption",
            "Worst outgoing disruption",
            "Positive-transfer cells",
            "Below-base cells",
            "Worst affected task",
        ]
    ].rename(
        columns={
            "Adapter": "Task",
            "Mean outgoing disruption": "Outgoing disruption",
            "Worst outgoing disruption": "Worst outgoing disruption",
            "Positive-transfer cells": "Positive-transfer count",
            "Below-base cells": "Outgoing below-base count",
        }
    )
    incoming = incoming_df[
        [
            "Task",
            "Mean incoming disruption",
            "Worst incoming disruption",
            "Below-base cells",
            "Worst incoming adapter",
        ]
    ].rename(
        columns={
            "Mean incoming disruption": "Incoming vulnerability",
            "Worst incoming disruption": "Worst incoming disruption",
            "Below-base cells": "Incoming below-base count",
        }
    )
    compatibility = outgoing.merge(incoming, on="Task", how="inner")
    compatibility["Task family"] = compatibility["Task"].map(TASK_FAMILIES)
    compatibility["Task"] = pd.Categorical(compatibility["Task"], categories=TASK_LABELS, ordered=True)
    compatibility = compatibility.sort_values("Task").reset_index(drop=True)
    compatibility["Task"] = compatibility["Task"].astype(str)
    return compatibility


def save_metric_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    single_df, raw_df, recovery_df = build_metric_tables()
    top_df = build_top_effects(recovery_df)
    outgoing_df, incoming_df = build_interference_summary(recovery_df)
    compatibility_df = build_task_compatibility_map(outgoing_df, incoming_df)

    outputs = [
        ("base_vs_single_task_adapters_7task", single_df, "Base and single-task adapter performance for the seven trained tasks.", "tab:ch5-base-single", False),
        ("cross_task_absolute_matrix_7task", raw_df, "Absolute cross-task metrics for each evaluation task and active adapter.", "tab:ch5-cross-task-absolute", True),
        ("cross_task_recovery_matrix_7task", recovery_df, "Headroom-normalised cross-task recovery for each evaluation task and active adapter.", "tab:ch5-cross-task-recovery", True),
        ("cross_task_top_effects_7task", top_df, "Largest positive and negative off-diagonal cross-task recovery effects.", "tab:ch5-top-effects", False),
        ("cross_task_outgoing_summary_7task", outgoing_df, "Outgoing off-diagonal recovery and disruption summary by active adapter.", "tab:ch5-outgoing-summary", False),
        ("cross_task_incoming_summary_7task", incoming_df, "Incoming off-diagonal recovery and disruption summary by evaluated task.", "tab:ch5-incoming-summary", False),
    ]
    for name, df, caption, label, landscape in outputs:
        write_csv(ANALYSIS_RESULTS / f"{name}.csv", df)
        write_csv(THESIS_TABLES / f"{name}.csv", df)
        write_tex_table(THESIS_TABLES / f"{name}.tex", df, caption, label, landscape)
    write_csv(ANALYSIS_RESULTS / "task_compatibility_map_7task.csv", compatibility_df)

    return single_df, raw_df, recovery_df, top_df


def plot_matrix(df: pd.DataFrame, value_columns: list[str], title: str, out_name: str, diverging: bool) -> None:
    values = df[value_columns].to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    fig_width = 10.5 if len(value_columns) > 6 else 8.5
    fig, ax = plt.subplots(figsize=(fig_width, 6.2))
    if diverging:
        im = ax.imshow(masked, cmap="RdBu_r", vmin=-2.0, vmax=2.0, aspect="auto")
    else:
        row_min = np.nanmin(values, axis=1, keepdims=True)
        row_max = np.nanmax(values, axis=1, keepdims=True)
        normed = (values - row_min) / np.maximum(row_max - row_min, EPS)
        masked = np.ma.masked_invalid(normed)
        im = ax.imshow(masked, cmap="viridis", aspect="auto")

    ax.set_xticks(np.arange(len(value_columns)))
    ax.set_yticks(np.arange(len(df)))
    ax.set_xticklabels(value_columns, rotation=40, ha="right")
    ax.set_yticklabels(df["Task"].tolist())
    ax.set_xlabel("Active adapter")
    ax.set_ylabel("Evaluation task")
    ax.set_title(title)
    ax.set_xticks(np.arange(-0.5, len(value_columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(df), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if not np.isfinite(value):
                continue
            text = f"{value:.2f}" if diverging else f"{value:.3f}"
            color_value = value if diverging else masked[i, j]
            text_color = "white" if abs(float(color_value)) > (1.0 if diverging else 0.6) else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8.5, color=text_color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Recovery" if diverging else "Row-normalised metric")
    fig.tight_layout()
    for root in (ANALYSIS_PLOTS, THESIS_FIGURES, THESIS_CH5_FIGURES):
        fig.savefig(root / f"{out_name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(root / f"{out_name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_single_task_performance(single_df: pd.DataFrame) -> None:
    def parse_percent_cell(value: object) -> float:
        if isinstance(value, str) and value.endswith("%"):
            return float(value[:-1]) / 100.0
        return float(value)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.15, 5.85]})

    asr = single_df[single_df["Task"] == "ASR"].iloc[0]
    asr_values = [parse_percent_cell(asr["Base"]), parse_percent_cell(asr["SingleTask"])]
    axes[0].bar(["Base", "ASR"], asr_values, color=[PALETTE["grey"], PALETTE["blue"]], width=0.65)
    axes[0].set_title("ASR")
    axes[0].set_ylabel("WER (lower is better)")
    axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.7)
    for i, v in enumerate(asr_values):
        axes[0].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    cls = single_df[single_df["Task"] != "ASR"].copy()
    cls_base = cls["Base"].map(parse_percent_cell)
    cls_single = cls["SingleTask"].map(parse_percent_cell)
    x = np.arange(len(cls))
    width = 0.36
    axes[1].bar(x - width / 2, cls_base, width, label="Base", color=PALETTE["grey"])
    axes[1].bar(x + width / 2, cls_single, width, label="Single-task adapter", color=PALETTE["blue"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cls["Task"], rotation=30, ha="right")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Accuracy or macro-F1")
    axes[1].set_title("Classification-style tasks")
    axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.7)
    axes[1].legend(frameon=False, loc="lower right")

    fig.suptitle("Base model versus selected single-task adapters", y=1.02)
    fig.tight_layout()
    save_figure(fig, "single_task_performance_7task")


def plot_cross_task_recovery(recovery_df: pd.DataFrame) -> None:
    matrix = recovery_df.set_index("Task")[TASK_LABELS].T
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    im = ax.imshow(values, cmap=RECOVERY_CMAP, vmin=-2.0, vmax=2.0, aspect="auto")
    ax.set_xticks(np.arange(len(TASK_LABELS)))
    ax.set_yticks(np.arange(len(TASK_LABELS)))
    ax.set_xticklabels(TASK_LABELS, rotation=35, ha="right")
    ax.set_yticklabels(TASK_LABELS)
    ax.set_xlabel("Evaluation task")
    ax.set_ylabel("Active single-task adapter")
    ax.set_title("")
    ax.set_xticks(np.arange(-0.5, len(TASK_LABELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(TASK_LABELS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if i == j:
                label = "1.00"
                bbox = {"boxstyle": "round,pad=0.18", "fc": "white", "ec": "#333333", "lw": 0.6, "alpha": 0.85}
            else:
                label = f"{value:.2f}"
                bbox = None
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(value) > 1.35 and bbox is None else "#111111",
                bbox=bbox,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.set_label("Headroom-normalised recovery")
    fig.tight_layout()
    save_figure(fig, "cross_task_recovery_matrix_7task_heatmap")


def plot_cross_task_absolute(raw_df: pd.DataFrame) -> None:
    matrix = raw_df.set_index("Task")[["Base"] + TASK_LABELS].T
    values = matrix.to_numpy(dtype=float)
    col_min = np.nanmin(values, axis=0, keepdims=True)
    col_max = np.nanmax(values, axis=0, keepdims=True)
    # Each task column has its own metric scale; colour only shows within-task rank.
    normed = (values - col_min) / np.maximum(col_max - col_min, EPS)
    # Reverse ASR because lower WER is better.
    asr_col = matrix.columns.get_loc("ASR")
    normed[:, asr_col] = 1.0 - normed[:, asr_col]

    fig, ax = plt.subplots(figsize=(9.8, 6.8))
    im = ax.imshow(normed, cmap=RAW_CMAP, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Evaluation task")
    ax.set_ylabel("Active adapter")
    ax.set_title("")
    ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if matrix.columns[j] == "ASR":
                label = f"{values[i, j] * 100:.2f}"
            else:
                label = f"{values[i, j] * 100:.1f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=7.8, color="#111111")

    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.set_label("Within-task oriented performance")
    fig.tight_layout()
    save_figure(fig, "cross_task_absolute_matrix_7task_heatmap")


def _signed_bar_colours(values: pd.Series) -> list[str]:
    return [PALETTE["blue"] if v >= 0 else PALETTE["orange"] for v in values]


def plot_interference_summaries(outgoing_df: pd.DataFrame, incoming_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), sharex=False)

    out_mean = outgoing_df.sort_values("Mean off-diagonal recovery")
    axes[0, 0].barh(out_mean["Adapter"], out_mean["Mean off-diagonal recovery"], color=_signed_bar_colours(out_mean["Mean off-diagonal recovery"]))
    axes[0, 0].axvline(0, color="#222222", linewidth=0.8)
    axes[0, 0].set_title("Mean outgoing recovery")

    out_worst = outgoing_df.sort_values("Worst off-diagonal recovery")
    axes[0, 1].barh(out_worst["Adapter"], out_worst["Worst off-diagonal recovery"], color=_signed_bar_colours(out_worst["Worst off-diagonal recovery"]))
    axes[0, 1].axvline(0, color="#222222", linewidth=0.8)
    axes[0, 1].set_title("Worst outgoing recovery")

    in_mean = incoming_df.sort_values("Mean incoming recovery")
    axes[1, 0].barh(in_mean["Task"], in_mean["Mean incoming recovery"], color=_signed_bar_colours(in_mean["Mean incoming recovery"]))
    axes[1, 0].axvline(0, color="#222222", linewidth=0.8)
    axes[1, 0].set_title("Mean incoming recovery")

    in_worst = incoming_df.sort_values("Worst incoming recovery")
    axes[1, 1].barh(in_worst["Task"], in_worst["Worst incoming recovery"], color=_signed_bar_colours(in_worst["Worst incoming recovery"]))
    axes[1, 1].axvline(0, color="#222222", linewidth=0.8)
    axes[1, 1].set_title("Worst incoming recovery")

    for ax in axes.ravel():
        ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
        ax.set_xlabel("Off-diagonal recovery")

    fig.tight_layout()
    save_figure(fig, "incoming_outgoing_recovery_summary_7task")


def plot_task_compatibility_map(compatibility_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    x = compatibility_df["Outgoing disruption"].to_numpy(dtype=float)
    y = compatibility_df["Incoming vulnerability"].to_numpy(dtype=float)

    x_cut = float(np.nanmedian(x))
    y_cut = float(np.nanmedian(y))
    x_max = float(np.nanmax(x)) * 1.12
    y_max = float(np.nanmax(y)) * 1.18

    ax.axvspan(-0.02, x_cut, ymin=0, ymax=y_cut / y_max, color=PALETTE["green"], alpha=0.075, linewidth=0)
    ax.axvspan(x_cut, x_max, ymin=0, ymax=y_cut / y_max, color=PALETTE["orange"], alpha=0.06, linewidth=0)
    ax.axhspan(y_cut, y_max, xmin=0, xmax=x_cut / x_max, color=PALETTE["purple"], alpha=0.055, linewidth=0)
    ax.axvline(x_cut, color="#AEB7C2", linewidth=0.8, linestyle="--", zorder=0)
    ax.axhline(y_cut, color="#AEB7C2", linewidth=0.8, linestyle="--", zorder=0)

    for family, family_df in compatibility_df.groupby("Task family", sort=False):
        ax.scatter(
            family_df["Outgoing disruption"],
            family_df["Incoming vulnerability"],
            s=190,
            color=FAMILY_COLOURS.get(str(family), PALETTE["blue"]),
            alpha=0.88,
            edgecolor="#202124",
            linewidth=0.8,
            zorder=3,
        )

    offsets = {
        "ASR": (8, 7),
        "ER": (8, -12),
        "IC": (8, 8),
        "KWS": (8, 7),
        "LID": (8, 7),
        "SV": (-9, 9),
        "VS": (12, 12),
    }
    for _, row in compatibility_df.iterrows():
        task_label = str(row["Task"])
        dx, dy = offsets.get(str(row["Task"]), (7, 7))
        ax.annotate(
            task_label,
            (float(row["Outgoing disruption"]), float(row["Incoming vulnerability"])),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.8,
            weight="bold",
            color="#111111",
            ha="right" if task_label == "SV" else "left",
            bbox={"boxstyle": "round,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.72},
            zorder=4,
        )

    ax.set_xlim(-0.02, x_max)
    ax.set_ylim(-0.02, y_max)
    ax.set_xlabel("Outgoing disruption")
    ax.set_ylabel("Incoming vulnerability")
    ax.grid(color="#E5E7EB", linewidth=0.65)
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    family_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=FAMILY_COLOURS[family],
            markeredgecolor="#202124",
            markersize=5.8,
            label=family,
        )
        for family in ["Transcription", "Paralinguistic", "Semantic", "Lexical", "Speaker", "Event"]
    ]
    family_legend = ax.legend(
        handles=family_handles,
        title="Family",
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.01, 1.01),
        borderpad=0.2,
        labelspacing=0.22,
        handletextpad=0.45,
        fontsize=7.2,
        title_fontsize=8.0,
    )
    ax.add_artist(family_legend)

    fig.tight_layout()
    save_figure(fig, "task_compatibility_map_7task")


def iter_lora_base_keys(adapter: Path) -> list[str]:
    with safe_open(adapter / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    bases = sorted(k.replace(".lora_A.weight", "") for k in keys if k.endswith(".lora_A.weight"))
    return bases


def load_pair(handle: safe_open, base_key: str) -> tuple[torch.Tensor, torch.Tensor]:
    a = handle.get_tensor(base_key + ".lora_A.weight").float()
    b = handle.get_tensor(base_key + ".lora_B.weight").float()
    return a, b


def lora_scaling(adapter: Path) -> float:
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    return float(cfg["lora_alpha"]) / float(cfg["r"])


def parse_layer_component(base_key: str) -> tuple[int, str]:
    if "audio_tower" in base_key:
        return -1, "audio_tower"
    parts = base_key.split(".")
    layer = -2
    for idx, part in enumerate(parts):
        if part == "layers" and idx + 1 < len(parts):
            try:
                layer = int(parts[idx + 1])
                break
            except ValueError:
                pass
    if any(token in base_key for token in ("q_proj", "k_proj", "v_proj", "o_proj")):
        return layer, "attention"
    if any(token in base_key for token in ("gate_proj", "up_proj", "down_proj")):
        return layer, "mlp"
    return layer, "other"


def block_names_for_base_key(base_key: str) -> list[str]:
    layer, _ = parse_layer_component(base_key)
    blocks = ["global"]
    if layer == -1:
        blocks.append("audio_tower")
    elif layer >= 0:
        blocks.append("decoder_all")
        if layer <= 11:
            blocks.append("early")
        elif layer <= 23:
            blocks.append("mid")
        else:
            blocks.append("late")
    return blocks


def low_rank_norm_sq(a: torch.Tensor, b: torch.Tensor, scale: float) -> float:
    aa = a @ a.T
    bb = b.T @ b
    return float((aa * bb).sum().item() * scale * scale)


def low_rank_dot(a1: torch.Tensor, b1: torch.Tensor, s1: float, a2: torch.Tensor, b2: torch.Tensor, s2: float) -> float:
    aa = a1 @ a2.T
    bb = b1.T @ b2
    return float((aa * bb).sum().item() * s1 * s2)


def build_task_vector_stats() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global_rows = []
    component_acc: dict[tuple[str, str], float] = {}
    component_elements: dict[tuple[str, str], int] = {}
    layer_acc: dict[tuple[str, int], float] = {}
    norms_sq: dict[str, float] = {}

    for task in TASKS:
        path = adapter_path(task)
        scale = lora_scaling(path)
        total_sq = 0.0
        bases = iter_lora_base_keys(path)
        with safe_open(path / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
            for base_key in bases:
                a, b = load_pair(handle, base_key)
                norm_sq = max(low_rank_norm_sq(a, b, scale), 0.0)
                total_sq += norm_sq
                layer, component = parse_layer_component(base_key)
                component_acc[(task.label, component)] = component_acc.get((task.label, component), 0.0) + norm_sq
                component_elements[(task.label, component)] = (
                    component_elements.get((task.label, component), 0) + int(b.shape[0] * a.shape[1])
                )
                layer_acc[(task.label, layer)] = layer_acc.get((task.label, layer), 0.0) + norm_sq

        norms_sq[task.label] = total_sq
        total_elements = sum(v for (name, _), v in component_elements.items() if name == task.label)
        global_rows.append(
            {
                "Task": task.label,
                "Global L2 Norm": math.sqrt(total_sq),
                "RMS Norm": math.sqrt(total_sq / total_elements) if total_elements else np.nan,
                "Delta Elements": total_elements,
                "LoRA Modules": len(bases),
            }
        )

    component_rows = [
        {
            "Task": task,
            "Component": component,
            "L2 Norm": math.sqrt(value),
            "Delta Elements": component_elements[(task, component)],
            "RMS Norm": math.sqrt(value / component_elements[(task, component)]),
        }
        for (task, component), value in sorted(component_acc.items())
    ]
    layer_rows = [
        {"Task": task, "Layer": layer, "L2 Norm": math.sqrt(value)}
        for (task, layer), value in sorted(layer_acc.items(), key=lambda item: (item[0][0], item[0][1]))
    ]

    cosine = pd.DataFrame(index=[task.label for task in TASKS], columns=[task.label for task in TASKS], dtype=float)
    for task_i in TASKS:
        for task_j in TASKS:
            if task_i.label == task_j.label:
                cosine.loc[task_i.label, task_j.label] = 1.0
                continue
            path_i = adapter_path(task_i)
            path_j = adapter_path(task_j)
            scale_i = lora_scaling(path_i)
            scale_j = lora_scaling(path_j)
            bases = sorted(set(iter_lora_base_keys(path_i)).intersection(iter_lora_base_keys(path_j)))
            dot = 0.0
            with safe_open(path_i / "adapter_model.safetensors", framework="pt", device="cpu") as hi:
                with safe_open(path_j / "adapter_model.safetensors", framework="pt", device="cpu") as hj:
                    for base_key in bases:
                        a1, b1 = load_pair(hi, base_key)
                        a2, b2 = load_pair(hj, base_key)
                        dot += low_rank_dot(a1, b1, scale_i, a2, b2, scale_j)
            denom = math.sqrt(norms_sq[task_i.label]) * math.sqrt(norms_sq[task_j.label])
            cosine.loc[task_i.label, task_j.label] = dot / denom if denom > EPS else np.nan

    global_df = pd.DataFrame(global_rows)
    component_df = pd.DataFrame(component_rows)
    layer_df = pd.DataFrame(layer_rows)
    cosine_df = cosine.reset_index().rename(columns={"index": "Task"})
    return global_df, component_df, layer_df, cosine_df


def build_block_cosine_diagnostics(recovery_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task_paths = {task.label: adapter_path(task) for task in TASKS}
    task_scales = {task.label: lora_scaling(task_paths[task.label]) for task in TASKS}
    task_bases = {task.label: iter_lora_base_keys(task_paths[task.label]) for task in TASKS}
    norm_sq: dict[tuple[str, str], float] = {(block, task.label): 0.0 for block in BLOCKS for task in TASKS}

    for task in TASKS:
        path = task_paths[task.label]
        scale = task_scales[task.label]
        with safe_open(path / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
            for base_key in task_bases[task.label]:
                a, b = load_pair(handle, base_key)
                value = max(low_rank_norm_sq(a, b, scale), 0.0)
                for block in block_names_for_base_key(base_key):
                    norm_sq[(block, task.label)] += value

    rows = []
    for block in BLOCKS:
        for task_i in TASKS:
            for task_j in TASKS:
                if task_i.label == task_j.label:
                    cosine = 1.0 if norm_sq[(block, task_i.label)] > EPS else np.nan
                    dot = norm_sq[(block, task_i.label)]
                else:
                    path_i = task_paths[task_i.label]
                    path_j = task_paths[task_j.label]
                    scale_i = task_scales[task_i.label]
                    scale_j = task_scales[task_j.label]
                    bases = sorted(set(task_bases[task_i.label]).intersection(task_bases[task_j.label]))
                    dot = 0.0
                    with safe_open(path_i / "adapter_model.safetensors", framework="pt", device="cpu") as hi:
                        with safe_open(path_j / "adapter_model.safetensors", framework="pt", device="cpu") as hj:
                            for base_key in bases:
                                if block not in block_names_for_base_key(base_key):
                                    continue
                                a1, b1 = load_pair(hi, base_key)
                                a2, b2 = load_pair(hj, base_key)
                                dot += low_rank_dot(a1, b1, scale_i, a2, b2, scale_j)
                    denom = math.sqrt(norm_sq[(block, task_i.label)]) * math.sqrt(norm_sq[(block, task_j.label)])
                    cosine = dot / denom if denom > EPS else np.nan
                rows.append(
                    {
                        "Block": block,
                        "Task_i": task_i.label,
                        "Task_j": task_j.label,
                        "Cosine": cosine,
                        "Dot": dot,
                        "Norm_i": math.sqrt(norm_sq[(block, task_i.label)]),
                        "Norm_j": math.sqrt(norm_sq[(block, task_j.label)]),
                    }
                )

    block_cosine_df = pd.DataFrame(rows)
    _validate_block_cosines(block_cosine_df)
    summary_df = summarise_block_cosines(block_cosine_df)
    key_pairs_df = build_block_key_pair_summary(block_cosine_df, recovery_df)
    correlations_df = build_block_cosine_correlations(block_cosine_df, recovery_df)
    return block_cosine_df, summary_df, key_pairs_df, correlations_df


def _validate_block_cosines(block_cosine_df: pd.DataFrame) -> None:
    for block in BLOCKS:
        matrix = block_cosine_df[block_cosine_df["Block"] == block].pivot(index="Task_i", columns="Task_j", values="Cosine")
        matrix = matrix.reindex(index=TASK_LABELS, columns=TASK_LABELS)
        diag = np.diag(matrix.to_numpy(dtype=float))
        if np.nanmax(np.abs(diag - 1.0)) > 1e-6:
            raise AssertionError(f"Unexpected non-unit diagonal for block {block}: {diag}")
        delta = matrix.to_numpy(dtype=float) - matrix.to_numpy(dtype=float).T
        if np.nanmax(np.abs(delta)) > 1e-6:
            raise AssertionError(f"Block cosine matrix is not symmetric for {block}")


def summarise_block_cosines(block_cosine_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for block in BLOCKS:
        df = block_cosine_df[(block_cosine_df["Block"] == block) & (block_cosine_df["Task_i"] != block_cosine_df["Task_j"])]
        values = pd.to_numeric(df["Cosine"], errors="coerce").dropna()
        rows.append(
            {
                "Block": block,
                "N": int(values.shape[0]),
                "Min cosine": float(values.min()),
                "Max cosine": float(values.max()),
                "Mean cosine": float(values.mean()),
                "Mean abs cosine": float(values.abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def recovery_lookup(recovery_df: pd.DataFrame) -> dict[tuple[str, str], float]:
    lookup = {}
    for _, row in recovery_df.iterrows():
        eval_task = row["Task"]
        for adapter in TASK_LABELS:
            if adapter == eval_task:
                continue
            lookup[(adapter, eval_task)] = float(row[adapter])
    return lookup


def build_block_key_pair_summary(block_cosine_df: pd.DataFrame, recovery_df: pd.DataFrame) -> pd.DataFrame:
    rec = recovery_lookup(recovery_df)
    rows = []
    cos = block_cosine_df.set_index(["Block", "Task_i", "Task_j"])["Cosine"]
    for task_a, task_b in KEY_DIAGNOSTIC_PAIRS:
        r_ab = rec[(task_a, task_b)]
        r_ba = rec[(task_b, task_a)]
        for block in BLOCKS:
            rows.append(
                {
                    "Pair": f"{task_a}-{task_b}",
                    "Task A": task_a,
                    "Task B": task_b,
                    "Block": block,
                    "Cosine": float(cos.loc[(block, task_a, task_b)]),
                    "A->B recovery": r_ab,
                    "B->A recovery": r_ba,
                    "Worst bidirectional disruption": max(0.0, -r_ab, -r_ba),
                    "Bidirectional asymmetry": abs(r_ab - r_ba),
                }
            )
    return pd.DataFrame(rows)


def build_block_cosine_correlations(block_cosine_df: pd.DataFrame, recovery_df: pd.DataFrame) -> pd.DataFrame:
    directed_recovery = offdiag_recovery_long(recovery_df)
    block_lookup = block_cosine_df.set_index(["Block", "Task_i", "Task_j"])["Cosine"]
    directed_rows = []
    for _, row in directed_recovery.iterrows():
        for block in BLOCKS:
            directed_rows.append(
                {
                    "Block": block,
                    "Cosine": float(block_lookup.loc[(block, row["Adapter"], row["Evaluation Task"])]),
                    "Recovery": float(row["Recovery"]),
                    "Disruption": float(row["Disruption"]),
                }
            )
    directed_df = pd.DataFrame(directed_rows)

    rec = recovery_lookup(recovery_df)
    unordered_rows = []
    for i, task_i in enumerate(TASK_LABELS):
        for task_j in TASK_LABELS[i + 1 :]:
            r_ij = rec[(task_i, task_j)]
            r_ji = rec[(task_j, task_i)]
            for block in BLOCKS:
                unordered_rows.append(
                    {
                        "Block": block,
                        "Cosine": float(block_lookup.loc[(block, task_i, task_j)]),
                        "Worst bidirectional disruption": max(0.0, -r_ij, -r_ji),
                        "Bidirectional asymmetry": abs(r_ij - r_ji),
                    }
                )
    unordered_df = pd.DataFrame(unordered_rows)

    rows = []
    for block in BLOCKS:
        directed_block = directed_df[directed_df["Block"] == block]
        unordered_block = unordered_df[unordered_df["Block"] == block]
        for label, x_col, y_col, df in [
            ("Directed block cosine vs off-diagonal recovery", "Cosine", "Recovery", directed_block),
            ("Directed block cosine vs off-diagonal disruption", "Cosine", "Disruption", directed_block),
            ("Unordered block cosine vs worst bidirectional disruption", "Cosine", "Worst bidirectional disruption", unordered_block),
            ("Unordered block cosine vs bidirectional asymmetry", "Cosine", "Bidirectional asymmetry", unordered_block),
        ]:
            rows.append(
                {
                    "Diagnostic": label,
                    "Block": block,
                    "N": int((pd.to_numeric(df[x_col], errors="coerce").notna() & pd.to_numeric(df[y_col], errors="coerce").notna()).sum()),
                    "Pearson r": _pearson(df[x_col], df[y_col]),
                    "Spearman rho": _spearman(df[x_col], df[y_col]),
                }
            )
    return pd.DataFrame(rows)


def _pearson(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _spearman(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return np.nan
    return _pearson(x[mask].rank(method="average"), y[mask].rank(method="average"))


def build_task_vector_diagnostics(
    global_df: pd.DataFrame,
    component_df: pd.DataFrame,
    layer_df: pd.DataFrame,
    cosine_df: pd.DataFrame,
    recovery_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long = offdiag_recovery_long(recovery_df)
    outgoing = (
        long.groupby("Adapter", sort=False)
        .agg(
            mean_recovery=("Recovery", "mean"),
            worst_recovery=("Recovery", "min"),
            mean_disruption=("Disruption", "mean"),
            worst_disruption=("Disruption", "max"),
        )
        .reset_index()
    )
    acoustic = long[long["Evaluation Task"].isin(ACOUSTIC_TASKS)]
    acoustic_transfer = acoustic.groupby("Adapter", sort=False)["Recovery"].mean().rename("acoustic_task_recovery").reset_index()

    audio_rms = (
        component_df[component_df["Component"] == "audio_tower"][["Task", "RMS Norm", "L2 Norm"]]
        .rename(columns={"Task": "Adapter", "RMS Norm": "audio_rms_norm", "L2 Norm": "audio_l2_norm"})
    )
    merged = (
        global_df.rename(columns={"Task": "Adapter", "Global L2 Norm": "global_l2_norm", "RMS Norm": "global_rms_norm"})
        .merge(outgoing, on="Adapter")
        .merge(acoustic_transfer, on="Adapter")
        .merge(audio_rms, on="Adapter")
    )

    layer = layer_df.copy()
    layer["Layer"] = pd.to_numeric(layer["Layer"])
    layer["sq"] = layer["L2 Norm"] ** 2
    decoder = layer[layer["Layer"] >= 0]
    upper = decoder[decoder["Layer"] >= 24].groupby("Task")["sq"].sum()
    total = decoder.groupby("Task")["sq"].sum()
    upper_conc = (upper / total).rename("upper_layer_concentration").reset_index().rename(columns={"Task": "Adapter"})
    merged = merged.merge(upper_conc, on="Adapter")

    pairs = []
    cos = cosine_df.set_index("Task")[TASK_LABELS]
    for _, row in long.iterrows():
        pairs.append(
            {
                "Adapter": row["Adapter"],
                "Evaluation Task": row["Evaluation Task"],
                "Recovery": row["Recovery"],
                "Disruption": row["Disruption"],
                "Cosine": float(cos.loc[row["Adapter"], row["Evaluation Task"]]),
            }
        )
    pair_df = pd.DataFrame(pairs)

    specs = [
        ("Global L2 norm vs mean outgoing disruption", "global_l2_norm", "mean_disruption", merged),
        ("Global L2 norm vs worst outgoing disruption", "global_l2_norm", "worst_disruption", merged),
        ("Global RMS norm vs mean outgoing disruption", "global_rms_norm", "mean_disruption", merged),
        ("Audio-projection RMS norm vs acoustic-task recovery", "audio_rms_norm", "acoustic_task_recovery", merged),
        ("Upper-layer concentration vs mean outgoing disruption", "upper_layer_concentration", "mean_disruption", merged),
        ("Upper-layer concentration vs worst outgoing disruption", "upper_layer_concentration", "worst_disruption", merged),
        ("Global cosine vs off-diagonal recovery", "Cosine", "Recovery", pair_df),
        ("Global cosine vs off-diagonal disruption", "Cosine", "Disruption", pair_df),
    ]
    rows = []
    for label, x_col, y_col, df in specs:
        rows.append(
            {
                "Diagnostic": label,
                "N": int((pd.to_numeric(df[x_col], errors="coerce").notna() & pd.to_numeric(df[y_col], errors="coerce").notna()).sum()),
                "Pearson r": _pearson(df[x_col], df[y_col]),
                "Spearman rho": _spearman(df[x_col], df[y_col]),
            }
        )

    return pd.DataFrame(rows), merged


def plot_task_vectors(global_df: pd.DataFrame, component_df: pd.DataFrame, layer_df: pd.DataFrame, cosine_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(global_df["Task"], global_df["Global L2 Norm"], color=PALETTE["blue"])
    ax.set_ylabel("Effective LoRA delta L2 norm")
    ax.set_title("Task-vector magnitude by adapter")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    fig.tight_layout()
    save_figure(fig, "task_vector_norms_7task")

    pivot_l2 = component_df.pivot(index="Task", columns="Component", values="L2 Norm").fillna(0.0)
    pivot_rms = component_df.pivot(index="Task", columns="Component", values="RMS Norm").fillna(0.0)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    bottom = np.zeros(len(pivot_l2))
    colors = {"attention": PALETTE["blue"], "mlp": PALETTE["pale_orange"], "audio_tower": PALETTE["green"], "other": PALETTE["purple"]}
    for component in [c for c in ("attention", "mlp", "audio_tower", "other") if c in pivot_l2.columns]:
        axes[0].bar(
            pivot_l2.index,
            pivot_l2[component],
            bottom=bottom,
            label=component.replace("_", " ").title(),
            color=colors.get(component),
        )
        bottom += pivot_l2[component].to_numpy()
    axes[0].set_ylabel("L2 norm")
    axes[0].set_title("Raw component norms")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.7)

    components = [c for c in ("attention", "mlp", "audio_tower") if c in pivot_rms.columns]
    x = np.arange(len(pivot_rms.index))
    width = 0.25
    for offset, component in enumerate(components):
        axes[1].bar(
            x + (offset - (len(components) - 1) / 2) * width,
            pivot_rms[component],
            width,
            label=component.replace("_", " ").title(),
            color=colors.get(component),
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(pivot_rms.index, rotation=35, ha="right")
    axes[1].set_ylabel("RMS norm per delta element")
    axes[1].set_title("Parameter-count-normalised norms")
    axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.7)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "component_statistics_7task")

    layer_pivot = layer_df.pivot(index="Task", columns="Layer", values="L2 Norm").fillna(0.0)
    layer_pivot = layer_pivot.reindex([task.label for task in TASKS])
    fig, ax = plt.subplots(figsize=(11, 5.6))
    matrix = layer_pivot.to_numpy(dtype=float)
    row_max = np.maximum(matrix.max(axis=1, keepdims=True), EPS)
    im = ax.imshow(matrix / row_max, cmap="magma", aspect="auto")
    ax.set_yticks(np.arange(len(layer_pivot.index)))
    ax.set_yticklabels(layer_pivot.index)
    ax.set_xticks(np.arange(len(layer_pivot.columns)))
    ax.set_xticklabels([str(c) for c in layer_pivot.columns], rotation=90, fontsize=7)
    ax.set_xlabel("Layer (-1 is audio projection)")
    ax.set_ylabel("Adapter")
    ax.set_title("Layer-wise task-vector magnitude (row-normalised)")
    fig.colorbar(im, ax=ax, label="Row-normalised L2 norm")
    fig.tight_layout()
    save_figure(fig, "adapter_magnitude_by_layer_7task")

    cos_values = cosine_df[[task.label for task in TASKS]].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(cos_values, cmap="RdBu", vmin=-0.08, vmax=0.08)
    ax.set_xticks(np.arange(len(TASKS)))
    ax.set_yticks(np.arange(len(TASKS)))
    ax.set_xticklabels([task.label for task in TASKS], rotation=40, ha="right")
    ax.set_yticklabels([task.label for task in TASKS])
    ax.set_title("Pairwise effective task-vector cosine similarity")
    for i in range(cos_values.shape[0]):
        for j in range(cos_values.shape[1]):
            ax.text(j, i, f"{cos_values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Cosine similarity")
    fig.tight_layout()
    save_figure(fig, "task_vector_cosine_7task")


def plot_task_vector_diagnostics(global_behaviour_df: pd.DataFrame, cosine_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7))
    axes[0].scatter(global_behaviour_df["global_l2_norm"], global_behaviour_df["mean_disruption"], color=PALETTE["orange"])
    for _, row in global_behaviour_df.iterrows():
        axes[0].annotate(row["Adapter"], (row["global_l2_norm"], row["mean_disruption"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[0].set_xlabel("Global L2 norm")
    axes[0].set_ylabel("Mean outgoing disruption")
    axes[0].set_title("Magnitude vs disruption")

    axes[1].scatter(global_behaviour_df["audio_rms_norm"], global_behaviour_df["acoustic_task_recovery"], color=PALETTE["green"])
    for _, row in global_behaviour_df.iterrows():
        axes[1].annotate(row["Adapter"], (row["audio_rms_norm"], row["acoustic_task_recovery"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[1].axhline(0, color="#222222", linewidth=0.8)
    axes[1].set_xlabel("Audio-projection RMS norm")
    axes[1].set_ylabel("Mean acoustic-task recovery")
    axes[1].set_title("Audio norm vs acoustic transfer")

    values = cosine_df.set_index("Task")[TASK_LABELS].to_numpy(dtype=float)
    offdiag = values[~np.eye(values.shape[0], dtype=bool)]
    axes[2].hist(offdiag, bins=12, color=PALETTE["blue"], edgecolor="white")
    axes[2].axvline(0, color="#222222", linewidth=0.8)
    axes[2].set_xlabel("Off-diagonal cosine similarity")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Global cosine distribution")

    for ax in axes:
        ax.grid(color="#E5E7EB", linewidth=0.7)
    fig.tight_layout()
    save_figure(fig, "task_vector_diagnostics_7task")


def save_figure(fig: plt.Figure, name: str) -> None:
    for root in (ANALYSIS_PLOTS, THESIS_FIGURES, THESIS_CH5_FIGURES):
        fig.savefig(root / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(root / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_task_vector_outputs(recovery_df: pd.DataFrame) -> None:
    global_df, component_df, layer_df, cosine_df = build_task_vector_stats()
    diagnostics_df, global_behaviour_df = build_task_vector_diagnostics(global_df, component_df, layer_df, cosine_df, recovery_df)
    block_cosine_df, block_summary_df, block_key_pairs_df, block_correlations_df = build_block_cosine_diagnostics(recovery_df)
    outputs = [
        ("task_vector_norms_7task", global_df, "Effective LoRA task-vector norms for the seven trained adapters.", "tab:ch5-task-vector-norms", False),
        ("component_statistics_7task", component_df, "Effective LoRA task-vector norms by broad model component.", "tab:ch5-component-statistics", False),
        ("layer_wise_statistics_7task", layer_df, "Effective LoRA task-vector norms by adapter and layer.", "tab:ch5-layer-statistics", True),
        ("task_vector_cosine_7task", cosine_df, "Pairwise cosine similarity between effective LoRA task vectors.", "tab:ch5-task-vector-cosine", True),
        ("task_vector_interference_correlations_7task", diagnostics_df, "Correlations between task-vector diagnostics and cross-task behaviour.", "tab:ch5-vector-correlations", False),
        ("task_vector_behaviour_summary_7task", global_behaviour_df, "Task-vector magnitude and outgoing behaviour diagnostics by adapter.", "tab:ch5-vector-behaviour", False),
    ]
    for name, df, caption, label, landscape in outputs:
        write_csv(ANALYSIS_RESULTS / f"{name}.csv", df)
        write_csv(THESIS_TABLES / f"{name}.csv", df)
        write_tex_table(THESIS_TABLES / f"{name}.tex", df, caption, label, landscape)
    for name, df in [
        ("task_vector_block_cosine_7task", block_cosine_df),
        ("task_vector_block_cosine_summary_7task", block_summary_df),
        ("task_vector_block_cosine_key_pairs_7task", block_key_pairs_df),
        ("task_vector_block_cosine_correlations_7task", block_correlations_df),
    ]:
        write_csv(ANALYSIS_RESULTS / f"{name}.csv", df)
    plot_task_vectors(global_df, component_df, layer_df, cosine_df)
    plot_task_vector_diagnostics(global_behaviour_df, cosine_df)


def validate_outputs(raw_df: pd.DataFrame, recovery_df: pd.DataFrame) -> None:
    missing = []
    for eval_task in TASKS:
        if not metric_path(eval_task, None).exists():
            missing.append(str(metric_path(eval_task, None)))
        for adapter_task in TASKS:
            if not metric_path(eval_task, adapter_task.key).exists():
                missing.append(str(metric_path(eval_task, adapter_task.key)))
    if missing:
        raise FileNotFoundError("Missing metric files:\n" + "\n".join(missing))

    for task in TASKS:
        diag = float(recovery_df.loc[recovery_df["Task"] == task.label, task.label].iloc[0])
        if abs(diag - 1.0) > 1e-6:
            raise AssertionError(f"Diagonal recovery for {task.label} is {diag}, expected 1.0")

    if raw_df.shape != (7, 10):
        raise AssertionError(f"Unexpected raw matrix shape {raw_df.shape}")
    if recovery_df.shape != (7, 9):
        raise AssertionError(f"Unexpected recovery matrix shape {recovery_df.shape}")


def main() -> None:
    ensure_dirs()
    single_df, raw_df, recovery_df, _ = save_metric_outputs()
    outgoing_df, incoming_df = build_interference_summary(recovery_df)
    compatibility_df = build_task_compatibility_map(outgoing_df, incoming_df)
    plot_single_task_performance(single_df)
    plot_cross_task_absolute(raw_df)
    plot_cross_task_recovery(recovery_df)
    plot_interference_summaries(outgoing_df, incoming_df)
    plot_task_compatibility_map(compatibility_df)
    save_task_vector_outputs(recovery_df)
    validate_outputs(raw_df, recovery_df)

    # Keep the historical analysis/results filenames available for quick comparison,
    # while making the Chapter 5 versions the canonical thesis artifacts.
    for name in [
        "cross_task_absolute_matrix_7task",
        "cross_task_recovery_matrix_7task",
        "task_vector_cosine_7task",
    ]:
        src = ANALYSIS_RESULTS / f"{name}.csv"
        if src.exists():
            shutil.copy2(src, ANALYSIS_RESULTS / f"{name.replace('_7task', '')}.csv")

    print(f"Wrote Chapter 5 tables to {THESIS_TABLES}")
    print(f"Wrote Chapter 5 figures to {THESIS_CH5_FIGURES}")


if __name__ == "__main__":
    main()
