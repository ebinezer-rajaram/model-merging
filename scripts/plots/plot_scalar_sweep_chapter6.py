#!/usr/bin/env python3
"""Generate combined calibration-search figure for Chapter 6.

Left panel:  Scalar BO sweep — worst-task and mean recovery vs gamma (7-task).
Right panel: Layer-wise gradient search — worst-task and mean validation
             recovery over training steps, with selected checkpoint marked.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_style import OKABE_ITO, apply_thesis_style, save_thesis_figure, style_axes

ROOT = Path(__file__).resolve().parents[2]

SWEEP_7TASK = ROOT / "artifacts/merged/uniform_scalar_delta/asr_emotion_intent_kws_langid_speaker_ver_vocalsound/sweeps/sweep_20260224_110452.json"
LAYERWISE_CSV = ROOT / "artifacts/merged/weighted_delta_n/asr_emotion_intent_kws_langid_speaker_ver_vocalsound/runs/run_supermerge_layer_wise_20260310_053857/heldout_metrics_history.csv"

OUT_DIR = ROOT / "thesis/figures/chapter6"

_COL_MEAN  = OKABE_ITO["blue"]
_COL_WORST = OKABE_ITO["vermillion"]
_COL_SEL   = "#4D4D4D"


def _load_scalar_sweep(path: Path) -> tuple[list[float], list[float], list[float], float]:
    with path.open() as f:
        d = json.load(f)
    runs = d["runs"]
    best_scale = float(runs[d["best_index"]]["params"]["scale"])
    triples = []
    for run in runs:
        sc = run["params"].get("scale", run["params"].get("scale_alpha"))
        det = run.get("score_details", {})
        mn = det.get("min_interference_delta")
        me = det.get("mean_interference_delta")
        if sc is None or mn is None or me is None:
            continue
        triples.append((float(sc), float(mn), float(me)))
    triples.sort(key=lambda t: t[0])
    xs      = [t[0] for t in triples]
    ys_min  = [t[1] for t in triples]
    ys_mean = [t[2] for t in triples]
    return xs, ys_min, ys_mean, best_scale


def _load_layerwise(path: Path) -> tuple[list[int], list[float], list[float], int]:
    steps, ys_min, ys_mean = [0], [0.0], [0.0]
    best_step, best_score = 0, -1.0
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["update_step"])
            mn   = float(row["min_interference_delta"])
            me   = float(row["arithmetic_mean_interference_delta"])
            sel  = row["selection_score"]
            steps.append(step)
            ys_min.append(mn)
            ys_mean.append(me)
            if sel and float(sel) > best_score:
                best_score = float(sel)
                best_step  = step
    return steps, ys_min, ys_mean, best_step


def main() -> None:
    xs7, ys_min7, ys_mean7, best_gamma = _load_scalar_sweep(SWEEP_7TASK)
    steps, lw_min, lw_mean, best_step  = _load_layerwise(LAYERWISE_CSV)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_thesis_style()

    fig, axes = plt.subplots(
        1, 2,
        figsize=(6.85, 2.65),
        gridspec_kw={"left": 0.08, "right": 0.995, "bottom": 0.19,
                     "top": 0.78, "wspace": 0.32},
    )

    # --- Left: Scalar BO sweep ---
    ax = axes[0]
    h_mean,  = ax.plot(xs7, ys_mean7, color=_COL_MEAN,  marker="o", markersize=4.5,
                       linewidth=1.5, label="Mean recovery")
    h_worst, = ax.plot(xs7, ys_min7,  color=_COL_WORST, marker="s", markersize=4.5,
                       linewidth=1.5, linestyle="--", label="Worst-task recovery")
    h_sel    = ax.axvline(best_gamma, color=_COL_SEL, linewidth=1.1,
                          linestyle=(0, (1.2, 2.0)), label="Selected")
    ax.axvline(1.0, color="#8B8B8B", linewidth=0.9,
               linestyle=(0, (3.5, 2.0)), label=r"$\gamma=1$ (raw sum)")
    ax.axvline(1.0 / 7, color="#BBBBBB", linewidth=0.9,
               linestyle=(0, (1.0, 1.8)), label=r"$\gamma=1/|S|$ (Uniform)")
    ax.axhline(1.0, color="#CCCCCC", linewidth=0.7,
               linestyle=(0, (1.4, 2.2)), zorder=0)
    ax.set_title("Scalar (BO sweep)", pad=4)
    ax.set_xlabel(r"Scale $\gamma$")
    ax.set_ylabel("Headroom-normalised recovery")
    ax.set_xlim(-0.04, 1.57)
    style_axes(ax)

    # --- Right: Layer-wise gradient search ---
    ax = axes[1]
    ax.plot(steps, lw_mean, color=_COL_MEAN,  marker="o", markersize=4.5,
            linewidth=1.5)
    ax.plot(steps, lw_min,  color=_COL_WORST, marker="s", markersize=4.5,
            linewidth=1.5, linestyle="--")
    ax.axvline(best_step, color=_COL_SEL, linewidth=1.1,
               linestyle=(0, (1.2, 2.0)))
    ax.axhline(1.0, color="#CCCCCC", linewidth=0.7,
               linestyle=(0, (1.4, 2.2)), zorder=0)
    ax.set_title("Layer-wise (gradient search)", pad=4)
    ax.set_xlabel("Training step")
    ax.set_xlim(-40, 1900)
    style_axes(ax)

    fig.legend(
        [h_mean, h_worst, h_sel], ["Mean recovery", "Worst-task recovery", "Selected"],
        frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3,
    )

    save_thesis_figure(fig, OUT_DIR / "chapter6_scalar_sweeps_combined", png=True)
    print(f"Saved {OUT_DIR / 'chapter6_scalar_sweeps_combined.pdf'}")


if __name__ == "__main__":
    main()
