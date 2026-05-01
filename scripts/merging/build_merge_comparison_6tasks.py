#!/usr/bin/env python3
"""Build method+task merge comparison CSV and interference plot for 6 tasks."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


INPUTS_BY_METHOD: Dict[str, List[Path]] = {
    "uniform_delta": [
        Path(
            "artifacts/merged/uniform_delta/emotion_intent_kws_langid/runs/"
            "run_20260209_223613/eval_results_test.json"
        ),
        Path(
            "artifacts/merged/uniform_delta/emotion_intent_kws_langid/runs/"
            "run_20260216_124047/eval_results_test.json"
        ),
    ],
    "uniform_scalar_delta": [
        Path(
            "artifacts/merged/uniform_scalar_delta/emotion_intent_kws_langid/runs/"
            "run_20260213_201016/eval_results_test.json"
        ),
        Path(
            "artifacts/merged/uniform_scalar_delta/emotion_intent_kws_langid/runs/"
            "run_20260216_130805/eval_results_test.json"
        ),
    ],
    "weighted_delta_n": [
        Path(
            "artifacts/merged/weighted_delta_n/emotion_intent_kws_langid/runs/"
            "run_supermerge_layer_wise_20260215_045019/eval_results_test.json"
        ),
        Path(
            "artifacts/merged/weighted_delta_n/emotion_intent_kws_langid/eval/test/"
            "eval_results_merged_weighted_delta_n_emotion_intent_kws_langid_test.json"
        ),
    ],
}

TASK_ORDER = ["emotion", "intent", "kws", "langid", "speaker_ver", "asr"]
METHOD_ORDER = ["uniform_delta", "uniform_scalar_delta", "weighted_delta_n"]
METHOD_LABELS = {
    "uniform_delta": "Uniform (1/T)",
    "uniform_scalar_delta": "Uniform scalar (lambda=0.574)",
    "weighted_delta_n": "SuperMerge layer-wise (scalar+simplex)",
}
RAW_METRIC_COLUMNS = [
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "wer",
    "loss",
    "recognized_rate",
    "num_samples",
    "runtime",
    "samples_per_second",
    "steps_per_second",
]

OUT_DIR = Path("analysis/merge_comparison")
OUT_CSV = OUT_DIR / "merge_summary_6tasks_method_task.csv"
OUT_PLOT = OUT_DIR / "merge_interference_by_task_method.png"


def _mean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _is_number(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _collect_rows() -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for method, paths in INPUTS_BY_METHOD.items():
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Missing input file: {path}")
            data = _load_json(path)
            results = data.get("results", {})
            if not isinstance(results, dict):
                raise ValueError(f"Missing/invalid 'results' in {path}")

            for task, task_metrics in results.items():
                if task not in TASK_ORDER:
                    continue
                if not isinstance(task_metrics, dict):
                    continue

                meta = task_metrics.get("interference_delta_meta", {})
                if not isinstance(meta, dict):
                    meta = {}

                primary_metric_name = meta.get("metric")
                primary_metric_value = None
                if isinstance(primary_metric_name, str):
                    metric_value = task_metrics.get(primary_metric_name)
                    if _is_number(metric_value):
                        primary_metric_value = float(metric_value)

                row = {
                    "method": method,
                    "task": task,
                    "primary_metric_name": primary_metric_name,
                    "primary_metric_value": primary_metric_value,
                    "interference_delta": task_metrics.get("interference_delta"),
                    "interference_base": meta.get("base"),
                    "interference_task_adapter": meta.get("task_adapter"),
                    "interference_merged": meta.get("merged"),
                    "split": data.get("split"),
                    "timestamp": data.get("timestamp"),
                    "merge_tag": data.get("merge_tag"),
                    "uniform_scalar_lambda": 0.574 if method == "uniform_scalar_delta" else None,
                }
                for metric_name in RAW_METRIC_COLUMNS:
                    row[metric_name] = task_metrics.get(metric_name)

                grouped[(method, task)].append(row)

    rows: list[dict] = []
    for method in METHOD_ORDER:
        for task in TASK_ORDER:
            vals = grouped.get((method, task), [])
            if not vals:
                continue
            if len(vals) == 1:
                rows.append(vals[0])
                continue

            merged: dict = {
                "method": method,
                "task": task,
                "primary_metric_name": vals[0].get("primary_metric_name"),
                "split": vals[0].get("split"),
                "timestamp": vals[0].get("timestamp"),
                "merge_tag": vals[0].get("merge_tag"),
                "uniform_scalar_lambda": 0.574 if method == "uniform_scalar_delta" else None,
            }
            for col in [
                "primary_metric_value",
                "interference_delta",
                "interference_base",
                "interference_task_adapter",
                "interference_merged",
            ] + RAW_METRIC_COLUMNS:
                merged[col] = _mean(v[col] for v in vals if _is_number(v.get(col)))
            rows.append(merged)

    return rows


def _write_csv(rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "task",
        "primary_metric_name",
        "primary_metric_value",
        "interference_delta",
        "interference_base",
        "interference_task_adapter",
        "interference_merged",
        "split",
        "timestamp",
        "merge_tag",
        "uniform_scalar_lambda",
    ] + RAW_METRIC_COLUMNS

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_interference(rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Matplotlib/NumPy are required to render this plot.") from exc

    values_by_method_task = {(r["method"], r["task"]): r.get("interference_delta") for r in rows}
    for method in METHOD_ORDER:
        for task in TASK_ORDER:
            if (method, task) not in values_by_method_task:
                raise ValueError(f"Missing (method, task)=({method}, {task}) for plotting.")

    x = np.arange(len(TASK_ORDER), dtype=float)
    width = 0.24
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars_uniform = ax.bar(
        x - width,
        [float(values_by_method_task[("uniform_delta", t)]) for t in TASK_ORDER],
        width=width,
        label=METHOD_LABELS["uniform_delta"],
        color=colors[0],
        alpha=0.9,
    )
    bars_uniform_scalar = ax.bar(
        x,
        [float(values_by_method_task[("uniform_scalar_delta", t)]) for t in TASK_ORDER],
        width=width,
        label=METHOD_LABELS["uniform_scalar_delta"],
        color=colors[1],
        alpha=0.9,
    )
    bars_supermerge = ax.bar(
        x + width,
        [float(values_by_method_task[("weighted_delta_n", t)]) for t in TASK_ORDER],
        width=width,
        label=METHOD_LABELS["weighted_delta_n"],
        color=colors[2],
        alpha=0.9,
    )

    ax.set_title("Interference Delta Comparison by Task", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Task", fontsize=11, fontweight="bold")
    ax.set_ylabel("Interference delta", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in TASK_ORDER], fontsize=10)
    ax.set_axisbelow(True)

    ax.grid(True, which="major", axis="y", alpha=0.4, linestyle="-", linewidth=0.8, color="gray")
    ax.grid(True, which="minor", axis="y", alpha=0.2, linestyle=":", linewidth=0.5, color="gray")
    ax.minorticks_on()

    all_values = [
        float(values_by_method_task[(method, task)])
        for method in METHOD_ORDER
        for task in TASK_ORDER
    ]
    pad = 0.12
    ymin = min(0.0, min(all_values)) - pad
    ymax = max(all_values) + pad
    ax.set_ylim(ymin, ymax)
    axis_lw = ax.spines["left"].get_linewidth()
    ax.axhline(0.0, color="black", linewidth=axis_lw, alpha=0.9)

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

    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _verify(rows: list[dict]) -> None:
    if len(rows) != 18:
        raise ValueError(f"Expected 18 rows (3 methods x 6 tasks), found {len(rows)}")
    counts = defaultdict(int)
    for row in rows:
        counts[row["method"]] += 1
    for method in METHOD_ORDER:
        if counts.get(method, 0) != 6:
            raise ValueError(f"Expected 6 tasks for {method}, found {counts.get(method, 0)}")
    if not OUT_PLOT.exists():
        raise FileNotFoundError(f"Plot not written: {OUT_PLOT}")
    if not OUT_CSV.exists():
        raise FileNotFoundError(f"CSV not written: {OUT_CSV}")


def main() -> None:
    rows = _collect_rows()
    _write_csv(rows)
    _plot_interference(rows)
    _verify(rows)
    print(f"Wrote CSV: {OUT_CSV}")
    print(f"Wrote plot: {OUT_PLOT}")


if __name__ == "__main__":
    main()
