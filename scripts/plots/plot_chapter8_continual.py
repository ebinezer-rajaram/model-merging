#!/usr/bin/env python3
"""Generate Chapter 8 continual-composition tables and figures.

The script reads existing continual-suite artefacts and writes thesis-ready
tables/figures. It does not modify experiment outputs.

Three methods compared:
  - continual_supermerge_lw  : Layer-Wise Continual Merge (main method)
  - continual_supermerge      : Two-Scalar Merge (scalar gradient baseline)
  - continual_mtl             : Sequential MTL (practical reference)
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_style import OKABE_ITO, apply_thesis_style, save_thesis_figure, style_axes


ROOT = Path(__file__).resolve().parents[2]
FIXED_DIR = ROOT / "artifacts" / "continual_suite" / "thesis_continual_extension_v1"
TWO_SCALAR_DIR = ROOT / "artifacts" / "continual_suite" / "thesis_continual_extension_supermerge_v1"
LAYER_WISE_DIR = ROOT / "artifacts" / "continual_suite" / "thesis_continual_extension_supermerge_layer_wise_v1"
FIG_DIR = ROOT / "thesis" / "figures" / "chapter8"
TABLE_DIR = ROOT / "thesis" / "tables"

PATH_ORDER = [
    "compatible_acoustic",
    "conflicting_stress",
    "heldout_asr_probe",
    "semantic_to_speaker_5task",
    "speaker_to_semantic_5task",
]

PATH_LABELS = {
    "compatible_acoustic": "Acoustic-first",
    "conflicting_stress": "Format-shift",
    "heldout_asr_probe": "ASR-held-out",
    "semantic_to_speaker_5task": "Semantic-first",
    "speaker_to_semantic_5task": "Speaker-first",
}

PATH_TASKS = {
    "compatible_acoustic": ["SV", "LID", "KWS", "ASR"],
    "conflicting_stress": ["ASR", "IC", "ER", "SV"],
    "heldout_asr_probe": ["SV", "LID", "KWS", "VS"],
    "semantic_to_speaker_5task": ["IC", "ER", "ASR", "VS", "SV"],
    "speaker_to_semantic_5task": ["SV", "VS", "ASR", "ER", "IC"],
}

TASK_LABELS = {
    "asr": "ASR",
    "emotion": "ER",
    "intent": "IC",
    "kws": "KWS",
    "langid": "LID",
    "speaker_ver": "SV",
    "vocalsound": "VS",
}

TASK_TABLE_LABELS = {
    **TASK_LABELS,
}

DISPLAY_TASK_TABLE_LABELS = {
    "ASR": "ASR",
    "ER": "ER",
    "IC": "IC",
    "KWS": "KWS",
    "LID": "LID",
    "SV": "SV",
    "VS": "VS",
}

TASK_ORDER = ["ASR", "ER", "IC", "KWS", "LID", "SV", "VS"]

# Main method is Layer-wise continual; Two-scalar is the scalar gradient baseline; Sequential MTL is reference.
METHOD_ORDER = ["continual_supermerge_lw", "continual_supermerge", "continual_mtl"]
METHOD_LABELS = {
    "continual_supermerge_lw": "Layer-wise continual",
    "continual_supermerge": "Two-scalar",
    "continual_mtl": "Sequential MTL",
}

# Table Method column row labels.
METHOD_TABLE_LABELS = {
    "continual_supermerge_lw": "Layer-wise continual",
    "continual_supermerge": "Two-scalar",
    "continual_mtl": "Sequential MTL",
}

METHOD_STYLE = {
    "continual_supermerge_lw": {"colour": OKABE_ITO["orange"], "marker": "D", "linestyle": "-"},
    "continual_supermerge": {"colour": OKABE_ITO["bluish_green"], "marker": "s", "linestyle": "--"},
    "continual_mtl": {"colour": OKABE_ITO["vermillion"], "marker": "^", "linestyle": ":"},
}

TASK_PLOT_LABELS = {
    "ASR": "ASR",
    "ER": "ER",
    "IC": "IC",
    "KWS": "KWS",
    "LID": "LID",
    "SV": "SV",
    "VS": "VS",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save(fig: plt.Figure, name: str) -> None:
    save_thesis_figure(fig, FIG_DIR / name, png=True)


def five_panel_figure(*, figsize: tuple[float, float]) -> tuple[plt.Figure, list[plt.Axes]]:
    """Return a balanced three-over-two layout without an empty sixth panel."""
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(
        2,
        6,
        left=0.085,
        right=0.995,
        bottom=0.13,
        top=0.86,
        wspace=0.8,
        hspace=0.58,
    )
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 1:3]),
        fig.add_subplot(grid[1, 3:5]),
    ]
    return fig, axes


def method_handles(*, include_lines: bool) -> list[Line2D]:
    handles: list[Line2D] = []
    for method in METHOD_ORDER:
        style = METHOD_STYLE[method]
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["colour"] if include_lines else "none",
                linestyle=style["linestyle"] if include_lines else "None",
                linewidth=1.65 if include_lines else 0.0,
                marker=style["marker"],
                markersize=5.4,
                markerfacecolor=style["colour"],
                markeredgecolor="white",
                markeredgewidth=0.6,
            )
        )
    return handles


def add_shared_legend(fig: plt.Figure, handles: list[Line2D]) -> None:
    fig.legend(
        handles=handles,
        labels=[METHOD_LABELS[method] for method in METHOD_ORDER],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        borderaxespad=0.0,
    )


def hide_inner_y_ticklabels(axes: list[plt.Axes]) -> None:
    for idx, ax in enumerate(axes):
        if idx not in {0, 3}:
            ax.tick_params(axis="y", labelleft=False)


def fmt(value: float, places: int = 3) -> str:
    if math.isnan(value):
        return "--"
    return f"{value:.{places}f}"


def f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return math.nan if value == "" else float(value)


def load_final_rows() -> list[dict[str, str]]:
    """Load final-stage rows for the three reported methods.

    Layer-Wise rows come from LAYER_WISE_DIR (method='continual_supermerge' in source,
    renamed to 'continual_supermerge_lw'). Two-Scalar rows come from TWO_SCALAR_DIR.
    Sequential MTL rows come from FIXED_DIR.
    """
    # Sequential MTL only from FIXED_DIR
    fixed_rows = read_csv(FIXED_DIR / "continual_final_comparison.csv")
    mtl_rows = [r for r in fixed_rows if r["method"] == "continual_mtl" and r["path_id"] in PATH_ORDER]

    # Two-Scalar from TWO_SCALAR_DIR
    two_scalar_rows_raw = read_csv(TWO_SCALAR_DIR / "continual_final_comparison.csv")
    two_scalar_rows = [r for r in two_scalar_rows_raw if r["method"] == "continual_supermerge" and r["path_id"] in PATH_ORDER]

    # Layer-Wise from LAYER_WISE_DIR — rename method key so it doesn't clash with Two-Scalar
    lw_rows_raw = read_csv(LAYER_WISE_DIR / "continual_final_comparison.csv")
    lw_rows: list[dict[str, str]] = []
    for r in lw_rows_raw:
        if r["method"] == "continual_supermerge" and r["path_id"] in PATH_ORDER:
            row = dict(r)
            row["method"] = "continual_supermerge_lw"
            lw_rows.append(row)

    rows = lw_rows + two_scalar_rows + mtl_rows
    rows.sort(key=lambda r: (PATH_ORDER.index(r["path_id"]), METHOD_ORDER.index(r["method"])))
    return rows


def worst_retained_task(row: dict[str, str]) -> tuple[str, float]:
    deltas = json.loads(row["per_task_delta_json"])
    seen = [task for task in row["seen_tasks"].split(",") if task]
    retained = [task for task in seen if task != row["added_task"]]
    if not retained:
        return "--", math.nan
    worst = min(retained, key=lambda task: float(deltas[task]))
    return TASK_LABELS[worst], float(deltas[worst])


def retained_min(row: dict[str, str]) -> float:
    """Minimum recovery over retained (prior) tasks, excluding the newly added task."""
    _, value = worst_retained_task(row)
    return value


def generate_final_table() -> None:
    rows = load_final_rows()
    by_path_method = {(row["path_id"], row["method"]): row for row in rows}

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\scriptsize",
        "\\renewcommand{\\arraystretch}{1.08}%",
        "\\setlength{\\tabcolsep}{2pt}%",
        "\\begin{tabular}{",
        "    >{\\raggedright\\arraybackslash}p{0.16\\textwidth}",
        "    >{\\raggedright\\arraybackslash}p{0.11\\textwidth}",
        "    >{\\raggedright\\arraybackslash}p{0.16\\textwidth}",
        "    S[table-format=1.3]",
        "    S[table-format=-1.3]",
        "    >{\\raggedright\\arraybackslash}p{0.11\\textwidth}",
        "    S[table-format=1.2]",
        "}",
        "\\toprule",
        "Path & Final task & Method & {Acquisition} & {Retained-task min} & Casualty & {ASR WER} \\\\",
        "\\midrule",
    ]
    for path in PATH_ORDER:
        path_rows = [by_path_method[(path, method)] for method in METHOD_ORDER]
        for idx, row in enumerate(path_rows):
            casualty, retained_value = worst_retained_task(row)
            added = TASK_LABELS[row["added_task"]]
            lines.append(
                f"{PATH_LABELS[path]} & "
                f"{added} & "
                f"{METHOD_TABLE_LABELS[row['method']]} & "
                f"{fmt(f(row, 'new_task_delta'))} & "
                f"{fmt(retained_value)} & "
                f"{casualty} & "
                f"{fmt(100.0 * f(row, 'asr__primary_metric_value'), 2)} \\\\"
            )
        if path != PATH_ORDER[-1]:
            lines.append("\\addlinespace[2pt]")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Final-stage acquisition--retention operating points. "
            "Acquisition is recovery on the final newly added task. "
            "Retained-task min is the minimum recovery over tasks present before that final addition. "
            "Casualty names the retained task with the lowest retained recovery. "
            "ASR WER is raw word error rate (\\%) where ASR is evaluated. "
            "Higher acquisition and retained-task min are better; lower WER is better.}",
            "\\label{tab:ch8-final-results}",
            "\\end{table}",
            "",
        ]
    )
    write(TABLE_DIR / "chapter8_final_stage_results.tex", "\n".join(lines))


def plot_acquisition_retention() -> None:
    rows = load_final_rows()
    fig, axes = five_panel_figure(figsize=(6.85, 4.45))
    for ax, path in zip(axes, PATH_ORDER):
        path_rows = [row for row in rows if row["path_id"] == path]
        for row in path_rows:
            style = METHOD_STYLE[row["method"]]
            ax.scatter(
                f(row, "new_task_delta"),
                retained_min(row),
                s=44,
                marker=style["marker"],
                color=style["colour"],
                edgecolor="white",
                linewidth=0.6,
                label=METHOD_LABELS[row["method"]],
                zorder=3,
            )
            casualty, _ = worst_retained_task(row)
            ax.annotate(
                TASK_PLOT_LABELS.get(casualty, casualty),
                xy=(f(row, "new_task_delta"), retained_min(row)),
                xytext=(3.0, 3.0),
                textcoords="offset points",
                fontsize=6.2,
                color="#333333",
                zorder=4,
            )
        ax.axhline(0.5, color="#9A9A9A", linewidth=0.75, linestyle=(0, (1.4, 2.2)), zorder=0)
        ax.axhline(0.0, color="#555555", linewidth=0.75, zorder=0)
        ax.axvline(0.8, color="#9A9A9A", linewidth=0.75, linestyle=(0, (1.4, 2.2)), zorder=0)
        ax.set_title(PATH_LABELS[path], pad=4)
        # Acoustic-first has LW acquisition=1.657; use 1.80 upper bound.
        ax.set_xlim(-0.05, 1.80)
        ax.set_ylim(-0.55, 1.05)
        style_axes(ax)
    hide_inner_y_ticklabels(axes)
    add_shared_legend(fig, method_handles(include_lines=False))
    fig.supxlabel("New-task recovery", x=0.54, y=0.035, fontsize=9.0)
    fig.supylabel("Retained-task minimum recovery", x=0.012, y=0.49, fontsize=9.0)
    save(fig, "chapter8_acquisition_retention_facets")


def plot_dumbbell_chart() -> None:
    rows = load_final_rows()
    fig, ax = plt.subplots(figsize=(6.85, 4.0))
    fig.subplots_adjust(top=0.84, bottom=0.20, left=0.09, right=0.97)

    method_x_offsets = {
        "continual_supermerge_lw": -0.22,
        "continual_supermerge": 0.0,
        "continual_mtl": 0.22,
    }

    for path_idx, path in enumerate(PATH_ORDER):
        for method in METHOD_ORDER:
            row = next(
                (r for r in rows if r["path_id"] == path and r["method"] == method),
                None,
            )
            if row is None:
                continue
            style = METHOD_STYLE[method]
            colour = style["colour"]
            marker = style["marker"]
            acq = f(row, "new_task_delta")
            rtm = retained_min(row)
            x = path_idx + method_x_offsets[method]

            # Connecting line between retention and acquisition
            ax.plot([x, x], [rtm, acq], color=colour, linewidth=1.5, alpha=0.65, zorder=2)

            # Acquisition dot (filled)
            ax.scatter(
                x, acq, s=44, marker=marker, color=colour,
                edgecolor="white", linewidth=0.6, zorder=4,
            )

            # Retained-task min dot (open)
            ax.scatter(
                x, rtm, s=44, marker=marker,
                facecolor="white", edgecolor=colour, linewidth=1.2, zorder=4,
            )

            # Annotate sub-0.5 retained-task min points
            if rtm < 0.5:
                casualty, _ = worst_retained_task(row)
                if casualty == "ASR":
                    wer_val = f(row, "asr__primary_metric_value") * 100
                    label = f"ASR\n{wer_val:.2f}%"
                else:
                    label = casualty
                # Place annotation to the right; below marker when RTM is negative
                xytext = (5, -14) if rtm < 0 else (5, 3)
                va = "top" if rtm < 0 else "bottom"
                ax.annotate(
                    label,
                    xy=(x, rtm),
                    xytext=xytext,
                    textcoords="offset points",
                    fontsize=6.2,
                    color=colour,
                    va=va,
                    zorder=5,
                )

    # Reference lines
    ax.axhline(0.0, color="#555555", linewidth=0.75, zorder=0)
    ax.axhline(0.5, color="#9A9A9A", linewidth=0.75, linestyle=(0, (1.4, 2.2)), zorder=0)

    # Axes
    ax.set_xlim(-0.55, len(PATH_ORDER) - 0.45)
    ax.set_ylim(-0.65, 1.85)
    ax.set_yticks([-0.5, 0.0, 0.5, 1.0, 1.5])
    ax.set_xticks(range(len(PATH_ORDER)))
    ax.set_xticklabels([PATH_LABELS[p] for p in PATH_ORDER], rotation=15, ha="right")
    ax.set_ylabel("Recovery", fontsize=8.5)

    # Legend: interleaved so column-major fill (ncol=3) gives row1=methods, row2=types
    # col-major with 5 items, ncol=3: col1=items[0,1], col2=items[2,3], col3=items[4]
    # row1 = items[0], items[2], items[4] → LW, 2S, Seq
    # row2 = items[1], items[3]           → Acq, RTM
    m_handles = method_handles(include_lines=True)
    lw_h, ts_h, seq_h = m_handles
    acq_proxy = Line2D(
        [0], [0], marker="o", color="none",
        markerfacecolor="#666666", markeredgecolor="white",
        markeredgewidth=0.5, markersize=5.4,
    )
    rtm_proxy = Line2D(
        [0], [0], marker="o", color="none",
        markerfacecolor="white", markeredgecolor="#666666",
        markeredgewidth=1.2, markersize=5.4,
    )
    fig.legend(
        handles=[lw_h, acq_proxy, ts_h, rtm_proxy, seq_h],
        labels=["Layer-wise continual", "Acquisition", "Two-scalar", "Retained-task min", "Sequential MTL"],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        borderaxespad=0.0,
        fontsize=7.5,
    )

    style_axes(ax)
    save(fig, "chapter8_dumbbell")


def load_stage_rows() -> list[dict[str, str]]:
    """Load per-stage rows for the three reported methods."""
    fixed_rows = read_csv(FIXED_DIR / "continual_stage_metrics.csv")
    mtl_rows = [r for r in fixed_rows if r["method"] == "continual_mtl" and r["path_id"] in PATH_ORDER]

    two_scalar_rows_raw = read_csv(TWO_SCALAR_DIR / "continual_stage_metrics.csv")
    two_scalar_rows = [r for r in two_scalar_rows_raw if r["method"] == "continual_supermerge" and r["path_id"] in PATH_ORDER]

    lw_rows_raw = read_csv(LAYER_WISE_DIR / "continual_stage_metrics.csv")
    lw_rows: list[dict[str, str]] = []
    for r in lw_rows_raw:
        if r["method"] == "continual_supermerge" and r["path_id"] in PATH_ORDER:
            row = dict(r)
            row["method"] = "continual_supermerge_lw"
            lw_rows.append(row)

    rows = lw_rows + two_scalar_rows + mtl_rows
    rows.sort(key=lambda r: (PATH_ORDER.index(r["path_id"]), int(r["stage_index"]), METHOD_ORDER.index(r["method"])))
    return rows


def retained_min_for_stage(row: dict[str, str]) -> float:
    deltas = json.loads(row["per_task_delta_json"])
    prior = [task for task in row["prior_tasks"].split(",") if task]
    if not prior:
        return math.nan
    return min(float(deltas[task]) for task in prior)


def retained_min_task_for_stage(row: dict[str, str]) -> tuple[str, float]:
    deltas = json.loads(row["per_task_delta_json"])
    prior = [task for task in row["prior_tasks"].split(",") if task]
    if not prior:
        return "--", math.nan
    worst = min(prior, key=lambda task: float(deltas[task]))
    return TASK_LABELS[worst], float(deltas[worst])


def acquisition_for_stage(row: dict[str, str]) -> float:
    deltas = json.loads(row["per_task_delta_json"])
    added = row["added_task"]
    val = deltas.get(added)
    return math.nan if val is None else float(val)


def plot_retained_min_trajectory() -> None:
    rows = load_stage_rows()
    fig, axes = five_panel_figure(figsize=(6.85, 4.75))
    for ax, path in zip(axes, PATH_ORDER):
        path_rows = [row for row in rows if row["path_id"] == path]
        stages = sorted({int(row["stage_index"]) for row in path_rows})
        labels = []
        for stage in stages:
            stage_row = next(row for row in path_rows if int(row["stage_index"]) == stage)
            labels.append(f"+{TASK_PLOT_LABELS[TASK_LABELS[stage_row['added_task']]]}")
        x_by_stage = {stage: idx for idx, stage in enumerate(stages)}
        for method in METHOD_ORDER:
            method_rows = [row for row in path_rows if row["method"] == method]
            if not method_rows:
                continue
            style = METHOD_STYLE[method]
            colour = style["colour"]
            # Retention trajectory (solid line + open markers — consistent with Fig 8.1 where open=retention)
            ax.plot(
                [x_by_stage[int(row["stage_index"])] for row in method_rows],
                [retained_min_for_stage(row) for row in method_rows],
                marker=style["marker"],
                markersize=5.0,
                linewidth=1.6,
                linestyle=style["linestyle"],
                color=colour,
                markerfacecolor="white",
                markeredgecolor=colour,
                markeredgewidth=1.2,
                label=METHOD_LABELS[method],
                zorder=3,
            )
            # Acquisition markers (filled) + dotted connectors at each stage
            for row in method_rows:
                x = x_by_stage[int(row["stage_index"])]
                acq = acquisition_for_stage(row)
                ret = retained_min_for_stage(row)
                if math.isnan(acq) or math.isnan(ret):
                    continue
                # Thin dotted connector
                ax.plot(
                    [x, x], [acq, ret],
                    color=colour, linewidth=0.8, alpha=0.45, linestyle=":", zorder=2,
                )
                # Filled acquisition marker
                ax.scatter(
                    x, acq, s=32, marker=style["marker"],
                    color=colour, edgecolor="white", linewidth=0.6, zorder=4,
                )
            # Annotate sub-0.5 retention points (retention only, not acquisition)
            for row in method_rows:
                weakest_task, value = retained_min_task_for_stage(row)
                if math.isnan(value) or value >= 0.5:
                    continue
                ax.annotate(
                    TASK_PLOT_LABELS.get(weakest_task, weakest_task),
                    xy=(x_by_stage[int(row["stage_index"])], value),
                    xytext=(2.5, 3.0),
                    textcoords="offset points",
                    fontsize=6.0,
                    color=colour,
                    zorder=5,
                )
        ax.axhline(0.5, color="#9A9A9A", linewidth=0.75, linestyle=(0, (1.4, 2.2)), zorder=0)
        ax.axhline(0.0, color="#555555", linewidth=0.75, zorder=0)
        ax.set_title(PATH_LABELS[path], pad=4)
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels(labels, rotation=0, ha="center")
        ax.set_ylim(-0.65, 1.95)
        ax.set_yticks([-0.5, 0.0, 0.5, 1.0, 1.5])
        style_axes(ax)
    hide_inner_y_ticklabels(axes)
    # Legend: interleaved for ncol=5 so all 5 items appear in one row
    # Filled = acquisition, open = retained-task min (consistent with Fig 8.1)
    m_handles = method_handles(include_lines=True)
    acq_proxy = Line2D(
        [0], [0], marker="o", color="none",
        markerfacecolor="#666666", markeredgecolor="white",
        markeredgewidth=0.5, markersize=5.4,
    )
    ret_proxy = Line2D(
        [0], [0], marker="o", color="none",
        markerfacecolor="white", markeredgecolor="#666666",
        markeredgewidth=1.2, markersize=5.4,
    )
    fig.legend(
        handles=m_handles + [acq_proxy, ret_proxy],
        labels=[METHOD_LABELS[m] for m in METHOD_ORDER] + ["Acquisition (filled)", "Retained-task min (open)"],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=5,
        borderaxespad=0.0,
        fontsize=7.2,
    )
    fig.supxlabel("Added task, in continual order", x=0.54, y=0.028, fontsize=9.0)
    fig.supylabel("Recovery", x=0.012, y=0.49, fontsize=9.0)
    save(fig, "chapter8_retained_min_trajectories")


def main() -> None:
    apply_thesis_style()
    plot_dumbbell_chart()
    plot_retained_min_trajectory()


if __name__ == "__main__":
    main()
