"""Shared modern plotting style for thesis figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
}


def apply_thesis_style() -> None:
    """Apply a clean, compact, print-oriented Matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.2,
            "axes.titleweight": "semibold",
            "axes.labelweight": "regular",
            "legend.fontsize": 8.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.6,
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "xtick.color": "#2F2F2F",
            "ytick.color": "#2F2F2F",
            "axes.edgecolor": "#2F2F2F",
            "axes.labelcolor": "#1F1F1F",
            "text.color": "#1F1F1F",
            "grid.color": "#E8E8E8",
            "grid.linewidth": 0.55,
            "grid.linestyle": "-",
            "legend.handlelength": 1.8,
            "legend.handletextpad": 0.45,
            "legend.columnspacing": 1.0,
            "legend.borderaxespad": 0.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )


def style_axes(ax: plt.Axes, *, y_grid: bool = True) -> None:
    """Apply consistent axis cleanup after plotting."""
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="both", which="major", width=0.55, length=2.8, pad=2.0)
    ax.grid(False)
    if y_grid:
        ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def save_thesis_figure(fig: plt.Figure, out_stem: Path, png: bool = True) -> None:
    """Save vector outputs, plus PNG when useful for review or artefact parity."""
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".pdf"), format="pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_stem.with_suffix(".svg"), format="svg", bbox_inches="tight", pad_inches=0.04)
    if png:
        fig.savefig(out_stem.with_suffix(".png"), format="png", bbox_inches="tight", pad_inches=0.04, dpi=300)
    plt.close(fig)
