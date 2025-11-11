"""Plotting utilities for training metrics."""

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    plt = None
    np = None


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

            # Deduplicate points with the same x-value (step) by keeping the first occurrence
            # This prevents vertical lines when validation and test metrics share the same step
            seen_steps = {}
            for s, v in points:
                if s not in seen_steps:  # First value wins for each step
                    seen_steps[s] = v
            points = sorted(seen_steps.items())  # Sort by step

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


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: Sequence[str],
    plot_path: Path,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> None:
    """Plot a confusion matrix for classification results.

    Args:
        y_true: True labels (as integer indices)
        y_pred: Predicted labels (as integer indices)
        label_names: Names of the emotion labels
        plot_path: Path to save the plot
        title: Title for the plot
        normalize: If True, normalize the confusion matrix by row (true label)
    """
    if plt is None or np is None:
        print("⚠️ Matplotlib/NumPy not available; skipping confusion matrix generation.")
        return

    # Filter out invalid labels (-1 means unrecognized)
    valid_indices = [i for i in range(len(y_true)) if y_true[i] >= 0]
    if not valid_indices:
        print("⚠️ No valid predictions found; skipping confusion matrix generation.")
        return

    y_true_filtered = [y_true[i] for i in valid_indices]
    y_pred_filtered = [y_pred[i] for i in valid_indices]

    num_classes = len(label_names)

    # Create confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true_filtered, y_pred_filtered):
        if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
            cm[true_label, pred_label] += 1

    # Normalize if requested
    if normalize:
        # Normalize by row (each true label sums to 1)
        row_sums = cm.sum(axis=1, keepdims=True)
        # Avoid division by zero
        cm_normalized = np.divide(
            cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0
        )
        cm_display = cm_normalized
        fmt = ".2f"
        vmin, vmax = 0.0, 1.0
    else:
        cm_display = cm
        fmt = "d"
        vmin, vmax = None, None

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    im = ax.imshow(cm_display, interpolation='nearest', cmap=plt.cm.Blues, vmin=vmin, vmax=vmax)
    ax.figure.colorbar(im, ax=ax)

    # Set ticks and labels
    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=label_names,
        yticklabels=label_names,
        ylabel='True Label',
        xlabel='Predicted Label',
        title=title
    )

    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm_display.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            value = cm_display[i, j]
            if normalize:
                # Show percentage
                text = f"{value:.1%}"
            else:
                # Show count
                text = f"{int(value)}"

            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if value > thresh else "black",
                fontsize=9
            )

    plt.tight_layout()

    # Save plot
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"🖼️ Saved confusion matrix to {plot_path}")
