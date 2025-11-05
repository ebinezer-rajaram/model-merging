"""Plotting utilities for training metrics."""

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _parse_float(value: Optional[str]) -> Optional[float]:
    """Safely parse floats that may be missing."""
    if value is None or value == "" or str(value).lower() == "none":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_step(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _collect_series(csv_path: Path) -> Tuple[List[Optional[int]], Dict[str, List[Optional[float]]]]:
    """Load series data from the training history CSV."""
    steps: List[Optional[int]] = []
    metric_series: Dict[str, List[Optional[float]]] = {}

    with csv_path.open("r") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            return steps, metric_series

        metric_fields = [field for field in reader.fieldnames if field != "step"]
        for field in metric_fields:
            metric_series[field] = []

        for row in reader:
            steps.append(_parse_step(row.get("step")))
            for field in metric_fields:
                metric_series[field].append(_parse_float(row.get(field)))

    return steps, metric_series


def plot_loss_and_wer(csv_path: Path, plot_path: Path) -> None:
    """Render training metrics against steps."""
    if plt is None:
        print("⚠️ Matplotlib not available; skipping plot generation.")
        return

    if not os.path.isfile(csv_path):
        print(f"⚠️ Metrics CSV {csv_path} not found; skipping plot generation.")
        return

    steps, metric_series = _collect_series(csv_path)
    if not metric_series:
        print("ℹ️ No metric columns available to plot.")
        return

    loss_metrics = {
        name: values for name, values in metric_series.items() if "loss" in name.lower()
    }
    remaining_metrics = {
        name: values
        for name, values in metric_series.items()
        if name not in loss_metrics and name.lower() != "epoch"
    }

    if loss_metrics:
        primary_metrics = loss_metrics
        secondary_metrics = remaining_metrics
    else:
        primary_metrics = remaining_metrics
        secondary_metrics = {}

    fig, ax_primary = plt.subplots(figsize=(8, 5))
    plotted = False

    def _plot_series(ax, series_dict, *, color_cycle=None):
        nonlocal plotted
        for index, (name, values) in enumerate(sorted(series_dict.items())):
            points = [(s, v) for s, v in zip(steps, values) if s is not None and v is not None]
            if not points:
                continue
            xs, ys = zip(*points)
            label = name.replace("eval_", "Eval ").replace("train_", "Train ").replace("_", " ").title()
            if color_cycle and index < len(color_cycle):
                ax.plot(xs, ys, label=label, color=color_cycle[index])
            else:
                ax.plot(xs, ys, label=label)
            plotted = True

    _plot_series(ax_primary, primary_metrics, color_cycle=["tab:blue", "tab:orange", "tab:red"])

    axis_label = "Loss" if loss_metrics else "Metric Value"
    ax_primary.set_ylabel(axis_label)

    ax_secondary = None
    if secondary_metrics:
        ax_secondary = ax_primary.twinx()
        _plot_series(ax_secondary, secondary_metrics, color_cycle=["tab:green", "tab:purple", "tab:brown"])
        ax_secondary.set_ylabel("Secondary Metrics")

    if not plotted:
        print("ℹ️ Not enough data to render plots.")
        plt.close(fig)
        return

    ax_primary.set_xlabel("Step")

    primary_handles, primary_labels = ax_primary.get_legend_handles_labels()
    if primary_handles:
        ax_primary.legend(loc="upper left")

    if ax_secondary is not None:
        secondary_handles, secondary_labels = ax_secondary.get_legend_handles_labels()
        if secondary_handles:
            ax_secondary.legend(loc="upper right")

    plt.title("Training Metrics Over Steps")
    plt.tight_layout()

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    plt.close(fig)

    print(f"🖼️ Saved training metrics plot to {plot_path}")
