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
    """Render training metrics against steps with proper subplot organization."""
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

    # Categorize metrics into losses and task-specific metrics
    loss_metrics = {
        name: values for name, values in metric_series.items() if "loss" in name.lower()
    }

    # Task metrics: accuracy, f1, macro_f1, wer, exact_match
    # Exclude: num_samples, recognized_rate, epoch, learning_rate
    task_metrics = {
        name: values
        for name, values in metric_series.items()
        if name not in loss_metrics
        and name.lower() != "epoch"
        and "num_samples" not in name.lower()
        and "recognized_rate" not in name.lower()
        and "learning_rate" not in name.lower()
        and any(
            keyword in name.lower()
            for keyword in ["accuracy", "f1", "macro_f1", "wer", "exact_match"]
        )
    }

    if not loss_metrics and not task_metrics:
        print("ℹ️ No plottable metrics found (losses or task metrics).")
        return

    # Determine number of subplots needed
    num_plots = sum([bool(loss_metrics), bool(task_metrics)])
    if num_plots == 0:
        print("ℹ️ No plottable metrics found.")
        return

    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots))
    if num_plots == 1:
        axes = [axes]

    current_ax_idx = 0

    def _plot_series(ax, series_dict, ylabel: str, is_wer: bool = False):
        """Plot a set of metrics on the given axis."""
        plotted = False
        all_y_values = []  # Collect all y values for custom scaling

        colors = plt.cm.tab10.colors
        color_idx = 0

        # Plot all metrics in sorted order, each with its own color
        for name, values in sorted(series_dict.items()):
            points = [(s, v) for s, v in zip(steps, values) if s is not None and v is not None]
            if not points:
                continue
            xs, ys = zip(*points)
            all_y_values.extend(ys)

            # Create clean label
            if name.startswith("train_"):
                label = "Train " + name.replace("train_", "").replace("_", " ").title()
            elif name.startswith("eval_"):
                label = "Eval " + name.replace("eval_", "").replace("_", " ").title()
            else:
                label = name.replace("_", " ").title()

            ax.plot(xs, ys, label=label, linewidth=2,
                   color=colors[color_idx % len(colors)], alpha=0.9)
            plotted = True
            color_idx += 1

        if plotted:
            ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
            ax.set_xlabel("Training Step", fontsize=11, fontweight='bold')

            # Enhanced grid - major and minor
            ax.grid(True, which='major', alpha=0.4, linestyle='-', linewidth=0.8, color='gray')
            ax.grid(True, which='minor', alpha=0.2, linestyle=':', linewidth=0.5, color='gray')
            ax.minorticks_on()

            ax.legend(loc="best", framealpha=0.95, fontsize=9)

            # Custom y-axis scaling for WER plots - add/subtract 20% of value for better visibility
            if is_wer and all_y_values:
                y_min, y_max = min(all_y_values), max(all_y_values)
                # Add 20% to max, subtract 20% from min
                y_min_scaled = y_min - (y_min * 0.20)
                y_max_scaled = y_max + (y_max * 0.20)
                ax.set_ylim(y_min_scaled, y_max_scaled)
            else:
                # Default padding for non-WER plots
                ax.margins(y=0.1)

        return plotted

    # Plot losses
    if loss_metrics:
        if _plot_series(axes[current_ax_idx], loss_metrics, "Loss"):
            axes[current_ax_idx].set_title("Training Loss", fontsize=12, fontweight='bold', pad=10)
            current_ax_idx += 1

    # Plot task metrics
    if task_metrics:
        # Determine the appropriate y-axis label based on which metrics are present
        metric_names = list(task_metrics.keys())

        if any("wer" in name.lower() for name in metric_names):
            ylabel = "WER"
            is_wer = True
        elif any("exact_match" in name.lower() for name in metric_names):
            ylabel = "Score"
            is_wer = False
        elif any("accuracy" in name.lower() for name in metric_names) and any("f1" in name.lower() for name in metric_names):
            ylabel = "Accuracy / F1"
            is_wer = False
        elif any("accuracy" in name.lower() for name in metric_names):
            ylabel = "Accuracy"
            is_wer = False
        elif any("f1" in name.lower() for name in metric_names):
            ylabel = "F1 Score"
            is_wer = False
        else:
            ylabel = "Metric Value"
            is_wer = False

        if _plot_series(axes[current_ax_idx], task_metrics, ylabel, is_wer=is_wer):
            axes[current_ax_idx].set_title("Task Metrics", fontsize=12, fontweight='bold', pad=10)

    plt.suptitle("Training Progress", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"🖼️ Saved training metrics plot to {plot_path}")
