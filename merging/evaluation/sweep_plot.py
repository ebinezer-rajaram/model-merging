"""Shared sweep plot data types and rendering for live plot regeneration during sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SweepPoint:
    lambda_value: float
    min_interference_delta: float | None
    mean_interference_delta: float | None
    per_task_deltas: dict[str, float] | None = None


def extract_points(sweep: dict) -> list[SweepPoint]:
    """Parse sweep JSON runs into a sorted list of SweepPoint."""
    points: list[SweepPoint] = []
    for run in sweep.get("runs", []):
        params = run.get("params") or {}
        lambda_value = params.get("lambda")
        if lambda_value is None:
            lambda_value = params.get("scale")
        if lambda_value is None:
            continue

        score_details = run.get("score_details") or {}
        min_delta = score_details.get("min_interference_delta")
        mean_delta = score_details.get("mean_interference_delta")

        # Fallbacks for older sweeps.
        if min_delta is None and isinstance(run.get("score"), (int, float)):
            min_delta = float(run["score"])

        # Derive mean from per-task results if not stored explicitly.
        if mean_delta is None:
            results = run.get("results") or {}
            if isinstance(results, dict):
                per_task_list = [
                    float(m["interference_delta"])
                    for m in results.values()
                    if isinstance(m, dict) and isinstance(m.get("interference_delta"), (int, float))
                ]
                if per_task_list:
                    mean_delta = sum(per_task_list) / len(per_task_list)

        per_task_deltas: dict[str, float] | None = None
        results = run.get("results") or {}
        if isinstance(results, dict):
            per_task = {
                str(task): float(m["interference_delta"])
                for task, m in results.items()
                if isinstance(m, dict) and isinstance(m.get("interference_delta"), (int, float))
            }
            if per_task:
                per_task_deltas = per_task

        points.append(
            SweepPoint(
                lambda_value=float(lambda_value),
                min_interference_delta=None if min_delta is None else float(min_delta),
                mean_interference_delta=None if mean_delta is None else float(mean_delta),
                per_task_deltas=per_task_deltas,
            )
        )

    return sorted(points, key=lambda p: p.lambda_value)


def plot_sweep(points: list[SweepPoint], *, title: str, output_path: Path) -> None:
    """Render a sweep interference plot to a PNG. Silently skips on any error.

    Plots solid lines for min/mean interference delta and dashed per-task lines.
    Used for live plot regeneration during sweeps; does not support gnuplot or
    reference-lambda guide lines (those remain in the CLI analysis script).
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except Exception:
        return

    if not points:
        return

    try:
        xs = [p.lambda_value for p in points]
        ys_min = [p.min_interference_delta for p in points]
        ys_mean = [p.mean_interference_delta for p in points]

        if not any(v is not None for v in ys_min) and not any(v is not None for v in ys_mean):
            return

        # Best lambda marker
        scored = [(p.lambda_value, p.min_interference_delta) for p in points if p.min_interference_delta is not None]
        best_x = max(scored, key=lambda item: item[1])[0] if scored else None
        best_y = next((p.min_interference_delta for p in points if p.lambda_value == best_x), None) if best_x is not None else None

        fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        legend_handles = []
        legend_labels = []

        line_min = ax.plot(xs, ys_min, marker="o", markersize=4, linewidth=2.2,
                           color="#1f77b4", alpha=0.95)[0]
        legend_handles.append(line_min)
        legend_labels.append("Min interference delta")

        if any(v is not None for v in ys_mean):
            line_mean = ax.plot(xs, ys_mean, marker="o", markersize=4, linewidth=2.2,
                                color="#ff7f0e", alpha=0.95)[0]
            legend_handles.append(line_mean)
            legend_labels.append("Mean interference delta")

        if best_x is not None and best_y is not None:
            best_scatter = ax.scatter([best_x], [best_y], marker="D", s=55,
                                      color="#2ca02c", alpha=0.95, zorder=5)
            legend_handles.append(best_scatter)
            legend_labels.append(f"Best λ={best_x:.3g}")

        # Per-task dashed lines
        all_tasks = sorted({t for p in points if p.per_task_deltas for t in p.per_task_deltas})
        try:
            tab20 = plt.cm.tab20.colors
        except Exception:
            tab20 = [
                "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
                "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
            ]
        for i, task in enumerate(all_tasks):
            task_ys = [p.per_task_deltas.get(task) if p.per_task_deltas else None for p in points]
            if not any(v is not None for v in task_ys):
                continue
            line_task = ax.plot(xs, task_ys, marker=".", markersize=3, linewidth=1.3,
                                linestyle="--", color=tab20[(i + 4) % len(tab20)], alpha=0.70)[0]
            legend_handles.append(line_task)
            legend_labels.append(f"{task} Δ")

        ax.set_xlabel("λ")
        ax.set_ylabel("Interference delta")
        ax.set_ylim(0.0, 1.0)
        x_min, x_max = min(xs), max(xs)
        ax.set_xlim(x_min, x_max)
        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.55, color="#bdbdbd")
        ax.grid(True, which="minor", linestyle=":", linewidth=0.6, alpha=0.35, color="#d9d9d9")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))

        n_legend_cols = min(len(legend_labels), 4)
        fig.suptitle(title, y=0.98)
        fig.legend(legend_handles, legend_labels, loc="upper center",
                   ncol=n_legend_cols, frameon=False, fontsize=8,
                   bbox_to_anchor=(0.5, 0.93))
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass


__all__ = ["SweepPoint", "extract_points", "plot_sweep"]
