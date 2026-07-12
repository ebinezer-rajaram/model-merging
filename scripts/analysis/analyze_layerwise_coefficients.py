#!/usr/bin/env python3
"""Layer-wise coefficient interpretability analysis for the 7-task merge.

Four analyses from saved coefficients (no model re-running required):
  A. Layer-block task contribution (early / mid / late)
  B. Dominant-task count per layer
  C. Shannon entropy by layer (line plot)
  D. Coefficient mass vs. task recovery (scatter)

Run from repo root:
  venv/bin/python scripts/analysis/analyze_layerwise_coefficients.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.plots.plot_style import apply_thesis_style, save_thesis_figure, style_axes, OKABE_ITO


# ---------------------------------------------------------------------------
# Canonical paths
# ---------------------------------------------------------------------------
ROOT = Path(".")
SUMMARY_PATH = (
    ROOT
    / "artifacts/merged/weighted_delta_n"
    / "asr_emotion_intent_kws_langid_speaker_ver_vocalsound"
    / "runs/run_supermerge_layer_wise_20260310_053857/summary.json"
)
PER_TASK_DIR = (
    ROOT
    / "artifacts/merged/weighted_delta_n"
    / "asr_emotion_intent_kws_langid_speaker_ver_vocalsound"
    / "eval/test/per_task"
)

OUT_FIG = ROOT / "thesis" / "figures" / "chapter6"
OUT_TAB = ROOT / "thesis" / "tables"

# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------
TASK_METRICS = {
    "emotion":     ("macro_f1",  "higher"),
    "intent":      ("accuracy",  "higher"),
    "kws":         ("macro_f1",  "higher"),
    "langid":      ("accuracy",  "higher"),
    "speaker_ver": ("accuracy",  "higher"),
    "asr":         ("wer",       "lower"),
    "vocalsound":  ("accuracy",  "higher"),
}

DISPLAY_NAME = {
    "asr":         "ASR",
    "emotion":     "ER",
    "intent":      "IC",
    "kws":         "KWS",
    "langid":      "LID",
    "speaker_ver": "SV",
    "vocalsound":  "VS",
}

DISPLAY_ORDER = ["asr", "emotion", "intent", "kws", "langid", "speaker_ver", "vocalsound"]

TASK_COLORS = {
    "asr":         OKABE_ITO["blue"],
    "emotion":     OKABE_ITO["vermillion"],
    "intent":      OKABE_ITO["bluish_green"],
    "kws":         OKABE_ITO["orange"],
    "langid":      OKABE_ITO["reddish_purple"],
    "speaker_ver": OKABE_ITO["sky_blue"],
    "vocalsound":  OKABE_ITO["black"],
}

N_LAYERS = 36
EARLY = slice(0, 12)   # layers 1–12 (0-indexed)
MID   = slice(12, 24)  # layers 13–24
LATE  = slice(24, 36)  # layers 25–36


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_coefficients(summary_path: Path) -> tuple[list[str], np.ndarray]:
    """Return (source_tasks, C) where C has shape (36, 7)."""
    with open(summary_path) as f:
        d = json.load(f)
    source_tasks = d["source_tasks"]
    raw = d["params"]["layer_task_coefficients"]
    C = np.array([raw[str(l)] for l in range(N_LAYERS)], dtype=float)  # (36, 7)
    return source_tasks, C


def load_recovery(source_tasks: list[str]) -> dict[str, float]:
    """Compute interference-delta (recovery) for each task from eval files."""
    recovery: dict[str, float] = {}
    for task in source_tasks:
        metric, direction = TASK_METRICS[task]
        # Merged value
        task_dir = PER_TASK_DIR / task
        merged_files = list(task_dir.glob("*.json"))
        if not merged_files:
            print(f"  WARNING: no eval file for {task}", file=sys.stderr)
            continue
        with open(merged_files[0]) as f:
            merged_val = json.load(f)[metric]

        base_path = ROOT / f"artifacts/{task}/metrics/eval/test/base_model.json"
        best_path = ROOT / f"artifacts/{task}/metrics/eval/test/best_{task}_adapter.json"
        with open(base_path) as f:
            base_val = json.load(f)[metric]
        with open(best_path) as f:
            best_val = json.load(f)[metric]

        if direction == "higher":
            r = (merged_val - base_val) / (best_val - base_val)
        else:
            r = (base_val - merged_val) / (base_val - best_val)
        recovery[task] = r

    return recovery


# ---------------------------------------------------------------------------
# Analysis A: layer-block task contribution
# ---------------------------------------------------------------------------

def analysis_a(source_tasks: list[str], C: np.ndarray) -> list[dict]:
    """Mean effective coefficient per block per task."""
    rows = []
    for task in DISPLAY_ORDER:
        t_idx = source_tasks.index(task)
        col = C[:, t_idx]
        rows.append({
            "task":    task,
            "display": DISPLAY_NAME[task],
            "early":   float(col[EARLY].mean()),
            "mid":     float(col[MID].mean()),
            "late":    float(col[LATE].mean()),
            "overall": float(col.mean()),
        })
    return rows


def print_table_a(rows: list[dict]) -> None:
    print("\n% === Table A: Layer-block task contribution ===")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Task & Early (1--12) & Middle (13--24) & Late (25--36) & Overall \\")
    print(r"\midrule")
    for r in rows:
        print(
            f"{r['display']} & {r['early']:.3f} & {r['mid']:.3f} & {r['late']:.3f} & {r['overall']:.3f} \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")


def save_csv_a(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "layerwise_block_contribution.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "display", "early", "mid", "late", "overall"])
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Analysis B: dominant-task count
# ---------------------------------------------------------------------------

def analysis_b(source_tasks: list[str], C: np.ndarray) -> list[dict]:
    """For each layer find dominant task; aggregate counts and margins per task."""
    dominant_counts = {t: 0 for t in source_tasks}
    dominant_margins: dict[str, list[float]] = {t: [] for t in source_tasks}

    for l in range(N_LAYERS):
        row = C[l]
        top2 = np.argsort(row)[-2:][::-1]
        dom_idx = top2[0]
        dom_task = source_tasks[dom_idx]
        margin = float(row[top2[0]] - row[top2[1]])
        dominant_counts[dom_task] += 1
        dominant_margins[dom_task].append(margin)

    result = []
    for task in DISPLAY_ORDER:
        count = dominant_counts[task]
        margins = dominant_margins[task]
        result.append({
            "task":           task,
            "display":        DISPLAY_NAME[task],
            "dominant_layers": count,
            "share_pct":      100.0 * count / N_LAYERS,
            "mean_margin":    float(np.mean(margins)) if margins else 0.0,
        })
    return result


def print_table_b(rows: list[dict]) -> None:
    print("\n% === Table B: Dominant-task count ===")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Task & Dominant layers & Share (\%) & Mean margin \\")
    print(r"\midrule")
    for r in rows:
        print(
            f"{r['display']} & {r['dominant_layers']} & {r['share_pct']:.1f} & {r['mean_margin']:.3f} \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")


def save_csv_b(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "layerwise_dominant_task.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "display", "dominant_layers", "share_pct", "mean_margin"])
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Analysis C: Shannon entropy by layer
# ---------------------------------------------------------------------------

def compute_entropy(C: np.ndarray) -> np.ndarray:
    """Compute per-layer Shannon entropy (nats) of the normalised coefficient distribution."""
    row_sums = C.sum(axis=1, keepdims=True)
    P = C / row_sums  # normalise to simplex
    # Clip to avoid log(0)
    P = np.clip(P, 1e-12, 1.0)
    H = -(P * np.log(P)).sum(axis=1)
    return H


def plot_entropy(H: np.ndarray, out_stem: Path) -> None:
    apply_thesis_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.5))

    layers = np.arange(1, N_LAYERS + 1)
    H_max = math.log(7)

    # Shade blocks
    block_alpha = 0.07
    ax.axvspan(1,  12, color="gray", alpha=block_alpha, lw=0)
    ax.axvspan(13, 24, color="white", alpha=0.0, lw=0)
    ax.axvspan(25, 36, color="gray", alpha=block_alpha, lw=0)

    # Block labels
    for mid_x, label in [(6.5, "Early"), (18.5, "Mid"), (30.5, "Late")]:
        ax.text(mid_x, H_max * 1.01, label, ha="center", va="bottom", fontsize=6.5, color="#555555")

    # Reference line
    ax.axhline(H_max, color="#AAAAAA", lw=0.8, ls="--", label=r"$H_{\max}=\ln 7$")

    # Entropy curve
    ax.plot(layers, H, color=OKABE_ITO["blue"], lw=1.2, zorder=3)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Entropy (nats)")
    ax.set_xlim(1, 36)
    ax.set_ylim(0, H_max * 1.12)
    ax.set_xticks([1, 6, 12, 18, 24, 30, 36])
    ax.legend(loc="lower right", fontsize=7)
    style_axes(ax)

    fig.tight_layout(pad=0.3)
    save_thesis_figure(fig, out_stem)
    print(f"  Saved entropy plot → {out_stem}.pdf")


# ---------------------------------------------------------------------------
# Analysis D: coefficient mass vs task recovery
# ---------------------------------------------------------------------------

def analysis_d(source_tasks: list[str], C: np.ndarray, recovery: dict[str, float]) -> list[dict]:
    rows = []
    for task in DISPLAY_ORDER:
        t_idx = source_tasks.index(task)
        mass = float(C[:, t_idx].mean())
        rec = recovery.get(task, float("nan"))
        rows.append({
            "task":     task,
            "display":  DISPLAY_NAME[task],
            "mass":     mass,
            "recovery": rec,
        })
    return rows


def plot_mass_recovery(rows: list[dict], out_stem: Path) -> None:
    apply_thesis_style()
    fig, ax = plt.subplots(figsize=(3.8, 3.2))

    masses    = [r["mass"]     for r in rows]
    recoveries = [r["recovery"] for r in rows]
    labels    = [r["display"]  for r in rows]
    tasks     = [r["task"]     for r in rows]

    median_mass = float(np.median(masses))

    # Reference lines
    ax.axhline(1.0,         color="#CCCCCC", lw=0.7, ls="--", zorder=1)
    ax.axvline(median_mass, color="#CCCCCC", lw=0.7, ls=":",  zorder=1)

    # Scatter points
    for r in rows:
        ax.scatter(
            r["mass"], r["recovery"],
            color=TASK_COLORS[r["task"]],
            s=40, zorder=4, edgecolors="white", linewidths=0.4,
        )

    # Labels — offset to avoid overlap; positive dy = above point
    offsets = {
        "ASR": ( 0.005,  0.03),   # top-centre outlier — label above
        "ER":  ( 0.007,  0.025),  # bottom-right — nudge above-right to clear axis
        "IC":  ( 0.005, -0.035),  # rightmost — label below
        "KWS": ( 0.005,  0.025),  # bottom-left cluster — label above
        "LID": (-0.005, -0.035),  # label below-left
        "SV":  ( 0.005,  0.025),  # mid — label above
        "VS":  (-0.005, -0.035),  # bottom-left — label below-left
    }
    for r in rows:
        dx, dy = offsets.get(r["display"], (0.005, 0.025))
        ax.annotate(
            r["display"],
            xy=(r["mass"], r["recovery"]),
            xytext=(r["mass"] + dx, r["recovery"] + dy),
            fontsize=7, color="#2F2F2F",
            arrowprops=None,
            va="center",
        )

    ax.set_xlabel("Mean coefficient mass")
    ax.set_ylabel("Task recovery")

    # Add headroom so bottom labels aren't clipped
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo - 0.04, yhi + 0.04)

    style_axes(ax)
    fig.tight_layout(pad=0.3)
    save_thesis_figure(fig, out_stem)
    print(f"  Saved mass-recovery scatter → {out_stem}.pdf")


def save_csv_d(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "layerwise_coeff_recovery.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "display", "mass", "recovery"])
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Combined compact table (A + B + D merged) for thesis half-page layout
# ---------------------------------------------------------------------------

def print_combined_table(
    rows_a: list[dict],
    rows_b: list[dict],
    rows_d: list[dict],
) -> None:
    """Single compact table: Early/Mid/Late | Dom. layers (share%) | Recovery."""
    # Index rows_b and rows_d by task
    b = {r["task"]: r for r in rows_b}
    d = {r["task"]: r for r in rows_d}

    print("\n% === Combined compact table for thesis (half-page layout) ===")
    print(r"% Place inside \begin{table}[t] ... \end{table}")
    print(r"\setlength{\tabcolsep}{4pt}")
    print(r"\begin{tabular}{@{}lcccccc@{}}")
    print(r"\toprule")
    print(r"Task & Early & Mid & Late & Dom.\ layers & Dom.\ share & Recovery \\")
    print(r"\midrule")
    for r in rows_a:
        task = r["task"]
        dom = b[task]["dominant_layers"]
        share = b[task]["share_pct"]
        rec = d[task]["recovery"]
        share_str = f"{share:.0f}\\%"
        dom_str = str(dom) if dom > 0 else "0"
        rec_str = f"{rec:.3f}"
        print(
            f"{r['display']} & {r['early']:.3f} & {r['mid']:.3f} & {r['late']:.3f}"
            f" & {dom_str} & {share_str} & {rec_str} \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print()
    print(r"% Suggested caption (adjust as needed):")
    print(
        r"% \caption{Layer-wise coefficient analysis for the 7-task merge."
        r" \emph{Early/Mid/Late}: mean effective coefficient in layers 1--12, 13--24, 25--36."
        r" \emph{Dom.\ layers}: number of the 36 layers at which that task holds the largest"
        r" coefficient. \emph{Recovery}: interference-delta relative to best single-task adapter"
        r" (1.0 = full recovery; ASR recovery $>1$ reflects small absolute WER headroom).}"
    )
    print()
    print(r"% Suggested LaTeX float (scatter left, table right, single figure environment):")
    print(r"% \begin{figure}[t]")
    print(r"% \centering")
    print(r"% \begin{minipage}[t]{0.44\linewidth}")
    print(r"% \centering")
    print(r"% \includegraphics[width=\linewidth]{figures/chapter6/chapter6_layerwise_mass_recovery}")
    print(r"% \end{minipage}\hfill")
    print(r"% \begin{minipage}[t]{0.52\linewidth}")
    print(r"% \centering\footnotesize\setlength{\tabcolsep}{3.5pt}")
    print(r"% [table here]")
    print(r"% \end{minipage}")
    print(r"% \caption{...}")
    print(r"% \end{figure}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading coefficients...")
    source_tasks, C = load_coefficients(SUMMARY_PATH)
    print(f"  source_tasks = {source_tasks}")
    print(f"  C shape = {C.shape}")

    print("Loading task recovery...")
    recovery = load_recovery(source_tasks)
    for task in DISPLAY_ORDER:
        print(f"  {DISPLAY_NAME[task]:3s}: recovery = {recovery.get(task, float('nan')):.3f}")

    # --- A ---
    rows_a = analysis_a(source_tasks, C)
    print_table_a(rows_a)
    save_csv_a(rows_a, OUT_TAB)
    print(f"\n  Saved → {OUT_TAB / 'layerwise_block_contribution.csv'}")

    # --- B ---
    rows_b = analysis_b(source_tasks, C)
    print_table_b(rows_b)
    save_csv_b(rows_b, OUT_TAB)
    print(f"  Saved → {OUT_TAB / 'layerwise_dominant_task.csv'}")

    # --- C ---
    H = compute_entropy(C)
    print(f"\nEntropy stats: min={H.min():.3f} max={H.max():.3f} mean={H.mean():.3f}  H_max={math.log(7):.3f}")
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    plot_entropy(H, OUT_FIG / "chapter6_layerwise_entropy")

    # --- D ---
    rows_d = analysis_d(source_tasks, C, recovery)
    save_csv_d(rows_d, OUT_TAB)
    print(f"  Saved → {OUT_TAB / 'layerwise_coeff_recovery.csv'}")
    plot_mass_recovery(rows_d, OUT_FIG / "chapter6_layerwise_mass_recovery")

    # --- Combined table ---
    print_combined_table(rows_a, rows_b, rows_d)

    print("\nDone.")


if __name__ == "__main__":
    main()
