#!/usr/bin/env python3
"""Plot per-task interference_delta comparison for two merge methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_interference_by_task(path: Path) -> Tuple[List[str], Dict[str, float]]:
    data = _load_json(path)
    results = data.get("results")
    if not isinstance(results, dict):
        raise ValueError(f"Missing 'results' dict in {path}")

    tasks: List[str] = []
    values: Dict[str, float] = {}
    for task, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        value = metrics.get("interference_delta")
        if isinstance(value, (int, float)):
            task_name = str(task)
            tasks.append(task_name)
            values[task_name] = float(value)

    if not values:
        raise ValueError(f"No per-task interference_delta values found in {path}")
    return tasks, values


def plot_interference_comparison(
    *,
    method_a_path: Path,
    method_b_path: Path,
    method_a_label: str,
    method_b_label: str,
    output_path: Path,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Matplotlib/NumPy are required to render this plot.") from exc

    tasks, method_a_values = _extract_interference_by_task(method_a_path)
    _, method_b_values = _extract_interference_by_task(method_b_path)

    for task in tasks:
        if task not in method_b_values:
            raise ValueError(f"Task '{task}' missing from {method_b_path}")

    x = np.arange(len(tasks), dtype=float)
    width = 0.36
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars_a = ax.bar(
        x - (width / 2),
        [method_a_values[t] for t in tasks],
        width=width,
        label=method_a_label,
        color=colors[0],
        alpha=0.9,
    )
    bars_b = ax.bar(
        x + (width / 2),
        [method_b_values[t] for t in tasks],
        width=width,
        label=method_b_label,
        color=colors[1],
        alpha=0.9,
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Task", fontsize=11, fontweight="bold")
    ax.set_ylabel("Interference delta", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in tasks], fontsize=10)
    ax.set_axisbelow(True)

    ax.grid(True, which="major", axis="y", alpha=0.4, linestyle="-", linewidth=0.8, color="gray")
    ax.grid(True, which="minor", axis="y", alpha=0.2, linestyle=":", linewidth=0.5, color="gray")
    ax.minorticks_on()

    all_values = [method_a_values[t] for t in tasks] + [method_b_values[t] for t in tasks]
    ymin = min(0.0, min(all_values) - 0.05)
    ymax = max(all_values) + 0.08
    ax.set_ylim(ymin, ymax)

    for bars in (bars_a, bars_b):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.legend(loc="upper left", framealpha=0.95, fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-task interference_delta comparison bars for two methods.")
    parser.add_argument("--method-a-path", type=Path, required=True)
    parser.add_argument("--method-b-path", type=Path, required=True)
    parser.add_argument("--method-a-label", type=str, default="Uniform (1/T)")
    parser.add_argument("--method-b-label", type=str, default="Uniform scalar (alpha=0.651)")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(
            "artifacts/merged/"
            "interference_comparison_uniform_delta_uniform_scalar_delta_"
            "asr_emotion_intent_kws_langid_speaker_ver_test.png"
        ),
    )
    parser.add_argument("--title", type=str, default="Interference Delta Comparison by Task")
    args = parser.parse_args()

    plot_interference_comparison(
        method_a_path=args.method_a_path,
        method_b_path=args.method_b_path,
        method_a_label=args.method_a_label,
        method_b_label=args.method_b_label,
        output_path=args.output_path,
        title=args.title,
    )
    print(f"Wrote plot: {args.output_path}")


if __name__ == "__main__":
    main()
