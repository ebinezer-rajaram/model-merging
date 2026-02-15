"""Export held-out tracking metrics to CSV and plots."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


_PLOT_METRICS: Sequence[tuple[str, str, str]] = (
    ("min_interference_delta", "Min Interference Delta", "heldout_min_interference_delta_over_time.png"),
    (
        "arithmetic_mean_interference_delta",
        "Arithmetic Mean Interference Delta",
        "heldout_arithmetic_mean_interference_delta_over_time.png",
    ),
    (
        "geometric_mean_interference_delta",
        "Geometric Mean Interference Delta",
        "heldout_geometric_mean_interference_delta_over_time.png",
    ),
    ("hypervolume", "Pareto Hypervolume", "heldout_hypervolume_over_time.png"),
    ("l2_shortfall_score", "L2 Shortfall Score", "heldout_l2_shortfall_over_time.png"),
)


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    return None


def _derive_vector_metrics(entry: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    minimum = _safe_float(entry.get("min_interference_delta"))
    arithmetic_mean = _safe_float(entry.get("arithmetic_mean_interference_delta"))
    geometric_mean = _safe_float(entry.get("geometric_mean_interference_delta"))
    if minimum is not None and arithmetic_mean is not None and geometric_mean is not None:
        return {
            "min_interference_delta": minimum,
            "arithmetic_mean_interference_delta": arithmetic_mean,
            "geometric_mean_interference_delta": geometric_mean,
        }

    vector_map = entry.get("interference_by_task")
    values: List[float] = []
    if isinstance(vector_map, Mapping):
        for value in vector_map.values():
            val = _safe_float(value)
            if val is not None:
                values.append(val)
    if not values:
        return {
            "min_interference_delta": minimum,
            "arithmetic_mean_interference_delta": arithmetic_mean,
            "geometric_mean_interference_delta": geometric_mean,
        }

    if minimum is None:
        minimum = float(min(values))
    if arithmetic_mean is None:
        arithmetic_mean = float(sum(values) / float(len(values)))
    if geometric_mean is None:
        if any(v < 0.0 for v in values):
            geometric_mean = None
        elif any(v == 0.0 for v in values):
            geometric_mean = 0.0
        else:
            geometric_mean = float(math.exp(sum(math.log(v) for v in values) / float(len(values))))
    return {
        "min_interference_delta": minimum,
        "arithmetic_mean_interference_delta": arithmetic_mean,
        "geometric_mean_interference_delta": geometric_mean,
    }


def _resolve_heldout_history(metadata: Mapping[str, Any]) -> List[Dict[str, Any]]:
    params = metadata.get("params")
    if isinstance(params, Mapping):
        optimizer = params.get("optimizer")
        if isinstance(optimizer, Mapping):
            provenance = optimizer.get("provenance")
            if isinstance(provenance, Mapping):
                history = provenance.get("heldout_eval_history")
                if isinstance(history, list):
                    return [dict(x) for x in history if isinstance(x, Mapping)]

    optimizer = metadata.get("optimizer")
    if isinstance(optimizer, Mapping):
        provenance = optimizer.get("provenance")
        if isinstance(provenance, Mapping):
            history = provenance.get("heldout_eval_history")
            if isinstance(history, list):
                return [dict(x) for x in history if isinstance(x, Mapping)]
    return []


def export_heldout_tracking_artifacts(
    *,
    metadata: Mapping[str, Any],
    output_dir: Path,
    show_summary: bool = True,
) -> Optional[Dict[str, Any]]:
    """Export held-out tracking CSV and plots when history exists."""
    history = _resolve_heldout_history(metadata)
    if not history:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "heldout_metrics_history.csv"

    rows: List[Dict[str, Any]] = []
    for i, entry in enumerate(history, start=1):
        vector_metrics = _derive_vector_metrics(entry)
        rows.append(
            {
                "eval_count": int(entry.get("eval_count", i)),
                "update_step": int(entry.get("update_step", i)),
                "selection_criterion": str(entry.get("selection_criterion", "")),
                "selection_score": _safe_float(entry.get("selection_score")),
                "delta_selection_score": _safe_float(entry.get("delta_selection_score")),
                "best_selection_score": _safe_float(entry.get("best_selection_score")),
                "no_improve_evals": int(entry.get("no_improve_evals", 0)),
                "frontier_size": int(entry.get("frontier_size", 0)),
                "is_nondominated": bool(entry.get("is_nondominated", False)),
                "should_stop": bool(entry.get("should_stop", False)),
                "min_interference_delta": vector_metrics["min_interference_delta"],
                "arithmetic_mean_interference_delta": vector_metrics["arithmetic_mean_interference_delta"],
                "geometric_mean_interference_delta": vector_metrics["geometric_mean_interference_delta"],
                "hypervolume": _safe_float(entry.get("hypervolume")),
                "delta_hypervolume": _safe_float(entry.get("delta_hypervolume")),
                "best_hypervolume": _safe_float(entry.get("best_hypervolume")),
                "l2_shortfall_score": _safe_float(entry.get("l2_shortfall_score")),
            }
        )

    fieldnames = [
        "eval_count",
        "update_step",
        "selection_criterion",
        "selection_score",
        "delta_selection_score",
        "best_selection_score",
        "no_improve_evals",
        "frontier_size",
        "is_nondominated",
        "should_stop",
        "min_interference_delta",
        "arithmetic_mean_interference_delta",
        "geometric_mean_interference_delta",
        "hypervolume",
        "delta_hypervolume",
        "best_hypervolume",
        "l2_shortfall_score",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    plot_paths: List[Path] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x_vals = [int(row["update_step"]) for row in rows]
        for metric_key, title, filename in _PLOT_METRICS:
            y_vals = [row.get(metric_key) for row in rows]
            points = [(x, y) for x, y in zip(x_vals, y_vals) if isinstance(y, (int, float))]
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [float(p[1]) for p in points]

            fig = plt.figure(figsize=(8.0, 4.8))
            plt.plot(xs, ys, marker="o", linewidth=1.8, markersize=4)
            plt.title(f"{title} Over Time")
            plt.xlabel("Optimizer update_step")
            plt.ylabel(title)
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            plot_path = output_dir / filename
            fig.savefig(plot_path, dpi=180)
            plt.close(fig)
            plot_paths.append(plot_path)
    except Exception as exc:
        if show_summary:
            print(f"⚠️  Skipped held-out metric plots: {exc}")

    if show_summary:
        print(f"💾 Held-out metrics CSV saved to {csv_path}")
        if plot_paths:
            print(f"🖼️  Held-out metric plots saved ({len(plot_paths)} files) to {output_dir}")
    return {"csv_path": csv_path, "plot_paths": plot_paths, "points": len(rows)}


__all__ = ["export_heldout_tracking_artifacts"]
