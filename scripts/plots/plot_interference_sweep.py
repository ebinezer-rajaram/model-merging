#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot min/mean interference delta vs scale alpha from a sweep JSON."
    )
    parser.add_argument("input_json", type=Path, help="Path to sweep_*.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: <input_stem>_interference_plot.png)",
    )
    return parser.parse_args()


def extract_xy(runs: list[dict]) -> tuple[list[float], list[float], list[float]]:
    xs: list[float] = []
    ys_min: list[float] = []
    ys_mean: list[float] = []

    for run in runs:
        params = run.get("params", {})
        score_details = run.get("score_details", {})

        # Support both naming variants in case future sweeps use scale_alpha.
        x = params.get("scale_alpha", params.get("scale"))
        y_min = score_details.get("min_interference_delta")
        y_mean = score_details.get("mean_interference_delta")

        if x is None or y_min is None or y_mean is None:
            continue

        xs.append(float(x))
        ys_min.append(float(y_min))
        ys_mean.append(float(y_mean))

    triples = sorted(zip(xs, ys_min, ys_mean), key=lambda t: t[0])
    if not triples:
        raise ValueError("No valid runs found with x + min/mean interference delta.")
    xs, ys_min, ys_mean = map(list, zip(*triples))
    return xs, ys_min, ys_mean


def main() -> None:
    args = parse_args()
    with args.input_json.open() as f:
        sweep = json.load(f)

    runs = sweep.get("runs", [])
    xs, ys_min, ys_mean = extract_xy(runs)

    best_alpha = None
    best_score = None
    best_index = sweep.get("best_index")
    if isinstance(best_index, int) and 0 <= best_index < len(runs):
        best_run = runs[best_index]
        best_alpha = best_run.get("params", {}).get(
            "scale_alpha", best_run.get("params", {}).get("scale")
        )
        best_score = best_run.get("score")
    elif ys_mean:
        # Fallback: use max mean interference delta.
        i = max(range(len(xs)), key=lambda j: ys_mean[j])
        best_alpha = xs[i]
        best_score = ys_mean[i]

    output = args.output
    if output is None:
        output = args.input_json.with_name(f"{args.input_json.stem}_interference_plot.png")

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys_min, marker="o", linewidth=2, label="Min interference delta")
    plt.plot(xs, ys_mean, marker="s", linewidth=2, label="Mean interference delta")
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Scale alpha")
    plt.ylabel("Interference delta")
    plt.title("Sweep: scale alpha vs interference delta across tasks")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if best_alpha is not None:
        label = f"Best alpha: {best_alpha:.6g}"
        if best_score is not None:
            label += f" (score={best_score:.6g})"
        plt.gcf().text(
            0.02,
            0.02,
            label,
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
        )
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    if best_alpha is not None:
        print(f"best_alpha={best_alpha} best_score={best_score}")
    print(output)


if __name__ == "__main__":
    main()
