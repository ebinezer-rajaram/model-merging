"""Plotting utilities for training metrics."""

import csv
import os
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _parse_float(value: Optional[str]) -> Optional[float]:
    """Safely parse floats that may be missing."""
    if value is None or value == "" or str(value).lower() == "none":
        return None
    return float(value)


def _collect_series(csv_path: Path) -> Tuple[List[Optional[int]], List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """Load series data from the training history CSV."""
    steps: List[Optional[int]] = []
    train_loss: List[Optional[float]] = []
    eval_loss: List[Optional[float]] = []
    wers: List[Optional[float]] = []

    with csv_path.open("r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            step_value = row.get("step")
            steps.append(int(float(step_value)) if step_value else None)
            train_loss.append(_parse_float(row.get("loss")))
            eval_loss.append(_parse_float(row.get("eval_loss")))
            wers.append(_parse_float(row.get("wer")))

    return steps, train_loss, eval_loss, wers


def plot_loss_and_wer(csv_path: Path, plot_path: Path) -> None:
    """Render loss and WER curves."""
    if plt is None:
        print("⚠️ Matplotlib not available; skipping plot generation.")
        return

    if not os.path.isfile(csv_path):
        print(f"⚠️ Metrics CSV {csv_path} not found; skipping plot generation.")
        return

    steps, train_loss, eval_loss, wers = _collect_series(csv_path)

    fig, ax_loss = plt.subplots(figsize=(8, 5))
    plotted = False

    train_points = [(s, l) for s, l in zip(steps, train_loss) if s is not None and l is not None]
    if train_points:
        train_steps, train_losses = zip(*train_points)
        ax_loss.plot(train_steps, train_losses, label="Train Loss", color="tab:blue")
        plotted = True

    eval_points = [(s, l) for s, l in zip(steps, eval_loss) if s is not None and l is not None]
    if eval_points:
        eval_steps, eval_losses = zip(*eval_points)
        ax_loss.plot(eval_steps, eval_losses, label="Eval Loss", color="tab:orange")
        plotted = True

    ax_wer = ax_loss.twinx()
    wer_points = [(s, w) for s, w in zip(steps, wers) if s is not None and w is not None]
    if wer_points:
        wer_steps, wer_values = zip(*wer_points)
        ax_wer.plot(wer_steps, wer_values, label="WER", color="tab:green")
        ax_wer.set_ylabel("Word Error Rate")
        plotted = True

    if not plotted:
        print("ℹ️ Not enough data to render plots.")
        plt.close(fig)
        return

    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(loc="upper left")
    ax_wer.legend(loc="upper right")
    plt.title("Training and Evaluation Metrics")
    plt.tight_layout()

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    plt.close(fig)

    print(f"🖼️ Saved loss/WER plot to {plot_path}")
