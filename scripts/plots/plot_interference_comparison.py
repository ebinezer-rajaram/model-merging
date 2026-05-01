#!/usr/bin/env python3
"""Plot per-task interference_delta comparison across merge methods."""

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
    uniform_path: Path,
    uniform_scalar_path: Path,
    supermerge_path: Path,
    output_path: Path,
    title: str,
    uniform_scalar_lambda_label: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Matplotlib/NumPy are required to render this plot.") from exc

    _, uniform = _extract_interference_by_task(uniform_path)
    _, uniform_scalar = _extract_interference_by_task(uniform_scalar_path)
    tasks, supermerge = _extract_interference_by_task(supermerge_path)

    # Keep task ordering from the reference file and require all methods to cover the same tasks.
    for task in tasks:
        if task not in uniform or task not in uniform_scalar:
            raise ValueError(f"Task '{task}' missing from one or more input files.")

    x = np.arange(len(tasks), dtype=float)
    width = 0.24
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars_uniform = ax.bar(
        x - width,
        [uniform[t] for t in tasks],
        width=width,
        label="Uniform (1/T)",
        color=colors[0],
        alpha=0.9,
    )
    bars_uniform_scalar = ax.bar(
        x,
        [uniform_scalar[t] for t in tasks],
        width=width,
        label=f"Uniform scalar (lambda={uniform_scalar_lambda_label:.3f})",
        color=colors[1],
        alpha=0.9,
    )
    bars_supermerge = ax.bar(
        x + width,
        [supermerge[t] for t in tasks],
        width=width,
        label="SuperMerge layer-wise (scalar+simplex)",
        color=colors[2],
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

    all_values = (
        [uniform[t] for t in tasks]
        + [uniform_scalar[t] for t in tasks]
        + [supermerge[t] for t in tasks]
    )
    ymin = min(0.0, min(all_values) - 0.05)
    ymax = max(all_values) + 0.08
    ax.set_ylim(ymin, ymax)

    for bars in (bars_uniform, bars_uniform_scalar, bars_supermerge):
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
    parser = argparse.ArgumentParser(description="Plot per-task interference_delta comparison bars.")
    parser.add_argument(
        "--uniform-path",
        type=Path,
        default=Path(
            "artifacts/merged/uniform_delta/emotion_intent_kws_langid/eval/test/"
            "eval_results_merged_uniform_delta_emotion_intent_kws_langid_test.json"
        ),
    )
    parser.add_argument(
        "--uniform-scalar-path",
        type=Path,
        default=Path(
            "artifacts/merged/uniform_scalar_delta/emotion_intent_kws_langid/eval/test/"
            "eval_results_merged_uniform_scalar_delta_emotion_intent_kws_langid_test.json"
        ),
    )
    parser.add_argument(
        "--supermerge-path",
        type=Path,
        default=Path(
            "artifacts/merged/weighted_delta_n/emotion_intent_kws_langid/eval/test/"
            "eval_results_merged_weighted_delta_n_emotion_intent_kws_langid_test.json"
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(
            "artifacts/merged/"
            "interference_comparison_uniform_delta_uniform_scalar_delta_weighted_delta_n_"
            "emotion_intent_kws_langid_test.png"
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Interference Delta Comparison by Task",
    )
    parser.add_argument(
        "--uniform-scalar-lambda-label",
        type=float,
        default=0.577,
        help="Legend label value for the uniform scalar lambda.",
    )
    args = parser.parse_args()

    plot_interference_comparison(
        uniform_path=args.uniform_path,
        uniform_scalar_path=args.uniform_scalar_path,
        supermerge_path=args.supermerge_path,
        output_path=args.output_path,
        title=args.title,
        uniform_scalar_lambda_label=args.uniform_scalar_lambda_label,
    )
    print(f"Wrote plot: {args.output_path}")


if __name__ == "__main__":
    main()
