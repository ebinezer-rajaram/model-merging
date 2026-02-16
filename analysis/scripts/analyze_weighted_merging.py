#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


LAMBDA_VALUES = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
TASK_METRICS = {
    "asr": "wer",
    "emotion": "macro_f1",
    "kws": "macro_f1",
    "intent": "accuracy",
    "langid": "accuracy",
    "speaker_id": "accuracy",
}
TASK_DISPLAY_NAMES = {
    "asr": "ASR",
    "kws": "KWS",
    "intent": "Intent",
    "emotion": "Emotion",
    "speaker_id": "Speaker ID",
    "langid": "LangID",
}


def load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pick_metric(task: str, metrics: dict) -> str:
    if task in TASK_METRICS and TASK_METRICS[task] in metrics:
        return TASK_METRICS[task]
    for key in metrics:
        if not key.startswith("_"):
            return key
    return "N/A"


def format_lambda(lambda_value: float) -> str:
    if lambda_value == 0.0:
        return "0"
    if lambda_value == 1.0:
        return "1"
    return f"{lambda_value:.1f}"


def resolve_pair(pair: str, tasks: set) -> tuple[str, str] | None:
    for task in tasks:
        prefix = f"{task}_"
        if pair.startswith(prefix):
            other = pair[len(prefix) :]
            if other in tasks:
                return task, other
    return None


def collect_pairs(artifacts_root: Path) -> dict[str, tuple[str, str]]:
    tasks = {p.name for p in artifacts_root.iterdir() if p.is_dir()}
    pairs = {}
    pattern = re.compile(r"best_merged_weighted_(.+)_lambda([0-9.]+)_adapter\.json")
    for path in artifacts_root.glob("*/metrics/eval/test/best_merged_weighted_*_adapter.json"):
        match = pattern.match(path.name)
        if not match:
            continue
        pair_name = match.group(1)
        if pair_name in pairs:
            continue
        resolved = resolve_pair(pair_name, tasks)
        if resolved:
            pairs[pair_name] = resolved
        else:
            print(f"Skipping unrecognized pair name: {pair_name}")
    return pairs


def adapter_metrics_path(artifacts_root: Path, task: str, adapter: str) -> Path:
    return artifacts_root / task / "metrics" / "eval" / "test" / f"best_{adapter}_adapter.json"


def base_metrics_path(artifacts_root: Path, task: str) -> Path:
    return artifacts_root / task / "metrics" / "eval" / "test" / "base_model.json"


def merged_metrics_path(artifacts_root: Path, task: str, pair: str, lambda_value: float) -> Path:
    lambda_str = format_lambda(lambda_value)
    filename = f"best_merged_weighted_{pair}_lambda{lambda_str}_adapter.json"
    return artifacts_root / task / "metrics" / "eval" / "test" / filename


def load_value(path: Path, metric: str) -> float | None:
    if not path.exists():
        return None
    metrics = load_metrics(path)
    return metrics.get(metric)


def collect_series(
    artifacts_root: Path,
    pair_name: str,
    task_a: str,
    task_b: str,
    metric_map: dict[str, str],
) -> dict[str, list[tuple[float, float]]]:
    series = {task_a: [], task_b: []}
    for task in [task_a, task_b]:
        metric = metric_map[task]
        for lambda_value in LAMBDA_VALUES:
            if lambda_value == 0.0:
                path = adapter_metrics_path(artifacts_root, task, task_b)
            elif lambda_value == 1.0:
                path = adapter_metrics_path(artifacts_root, task, task_a)
            else:
                path = merged_metrics_path(artifacts_root, task, pair_name, lambda_value)
            value = load_value(path, metric)
            if value is None:
                print(f"Missing {metric} for {task} at lambda {lambda_value}: {path}")
                continue
            series[task].append((lambda_value, value))
    return series


def metric_title(metric_key: str) -> tuple[str, str]:
    if metric_key == "wer":
        return "WER", "\u2193"
    if metric_key == "macro_f1":
        return "Macro F1", "\u2191"
    if metric_key == "accuracy":
        return "Accuracy", "\u2191"
    return metric_key.replace("_", " ").title(), "\u2191"


def display_task_name(task: str) -> str:
    return TASK_DISPLAY_NAMES.get(task, task)


def plot_pair(
    pair_name: str,
    task_a: str,
    task_b: str,
    series: dict[str, list[tuple[float, float]]],
    metric_map: dict[str, str],
    base_map: dict[str, float | None],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    xtick_labels = ["0", "0.1", "0.3", "0.5", "0.7", "0.9", "1"]
    legend_handles = []
    legend_labels = []
    for ax, task, tag in zip(axes, [task_a, task_b], ["Adapter A", "Adapter B"]):
        task_label = display_task_name(task)
        points = series[task]
        if not points:
            ax.set_title(f"{task_label} ({tag})")
            ax.set_xlabel("Merge weight (\u03bb)")
            ax.set_ylabel(metric_map[task])
            continue
        points = sorted(points, key=lambda item: item[0])
        xs = [item[0] for item in points]
        ys = [item[1] for item in points]
        line = ax.plot(xs, ys, marker="o", markersize=4, linewidth=2.2, label="Weighted merge")[0]
        if "Weighted merge" not in legend_labels:
            legend_handles.append(line)
            legend_labels.append("Weighted merge")
        metric_label, direction = metric_title(metric_map[task])
        ax.set_title(f"{task_label} ({metric_label} {direction})")
        ax.set_xlabel("Merge weight (\u03bb)")
        ax.set_ylabel(metric_label)
        base_value = base_map.get(task)
        if base_value is not None:
            base_line = ax.axhline(
                base_value,
                color="#b22222",
                linestyle=(0, (4, 3)),
                linewidth=1.0,
                alpha=0.6,
                label="Base model",
            )
            if "Base model" not in legend_labels:
                legend_handles.append(base_line)
                legend_labels.append("Base model")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35)
        ax.set_xticks(LAMBDA_VALUES, labels=xtick_labels)
        ax.set_xlim(min(LAMBDA_VALUES), max(LAMBDA_VALUES))
        # Tighter y-limits with padding.
        y_values = ys[:]
        if base_value is not None:
            y_values.append(base_value)
        y_min = min(y_values)
        y_max = max(y_values)
        pad = (y_max - y_min) * 0.12 if y_max > y_min else 0.01
        ax.set_ylim(y_min - pad, y_max + pad)
        if metric_map[task] in {"macro_f1", "accuracy"}:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
        # Endpoint annotations.
        ax.annotate(
            "Adapter B",
            xy=(0.0, 0.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            color="#555555",
        )
        ax.annotate(
            "Adapter A",
            xy=(1.0, 0.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            color="#555555",
        )
    pair_title = f"{display_task_name(task_a)} (Adapter A) vs {display_task_name(task_b)} (Adapter B)"
    fig.suptitle(pair_title, y=0.98)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=len(legend_labels),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.93),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"weighted_{pair_name}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_metric_map(artifacts_root: Path, tasks: set[str]) -> dict[str, str]:
    metric_map = {}
    for task in tasks:
        metrics_path = adapter_metrics_path(artifacts_root, task, task)
        if not metrics_path.exists():
            continue
        metrics = load_metrics(metrics_path)
        metric_map[task] = pick_metric(task, metrics)
    return metric_map


def build_base_map(
    artifacts_root: Path, tasks: set[str], metric_map: dict[str, str]
) -> dict[str, float | None]:
    base_map: dict[str, float | None] = {}
    for task in tasks:
        base_path = base_metrics_path(artifacts_root, task)
        if not base_path.exists() or task not in metric_map:
            base_map[task] = None
            continue
        metrics = load_metrics(base_path)
        base_map[task] = metrics.get(metric_map[task])
    return base_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze weighted merging results.")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts"),
        help="Path to artifacts directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/plots/weighted"),
        help="Directory for output plots.",
    )
    args = parser.parse_args()

    artifacts_root = args.artifacts_root
    pairs = collect_pairs(artifacts_root)
    tasks = {p.name for p in artifacts_root.iterdir() if p.is_dir()}
    metric_map = build_metric_map(artifacts_root, tasks)
    base_map = build_base_map(artifacts_root, tasks, metric_map)

    for pair_name, (task_a, task_b) in sorted(pairs.items()):
        if task_a not in metric_map or task_b not in metric_map:
            print(f"Skipping {pair_name}: missing metric map for {task_a} or {task_b}")
            continue
        series = collect_series(artifacts_root, pair_name, task_a, task_b, metric_map)
        plot_pair(pair_name, task_a, task_b, series, metric_map, base_map, args.output_dir)
        print(f"Wrote plot for {pair_name}")


if __name__ == "__main__":
    main()
