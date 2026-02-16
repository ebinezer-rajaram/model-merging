#!/usr/bin/env python3
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


TASK_METRICS = {
    "asr": "wer",
    "emotion": "macro_f1",
    "kws": "macro_f1",
    "intent": "accuracy",
    "langid": "accuracy",
    "speaker_ver": "accuracy",
    "speaker_id": "accuracy",
}
TASK_DISPLAY_NAMES = {
    "asr": "ASR",
    "kws": "KWS",
    "intent": "Intent",
    "emotion": "Emotion",
    "speaker_id": "Speaker ID",
    "speaker_ver": "Speaker Ver",
    "langid": "LangID",
}


def load_json(path: Path) -> dict:
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
    return f"{lambda_value:.3g}"


def resolve_pair(pair: str, tasks: set[str]) -> tuple[str, str] | None:
    for task in tasks:
        prefix = f"{task}_"
        if pair.startswith(prefix):
            other = pair[len(prefix) :]
            if other in tasks:
                return task, other
    return None


def collect_pairs(merged_root: Path, tasks: set[str], split: str) -> dict[str, tuple[str, str]]:
    pairs = {}
    pattern = re.compile(r"eval_results_merged_weighted_delta_(.+)_lambda[0-9.]+_.*\.json")
    for path in merged_root.glob(f"*/eval/{split}/eval_results_merged_weighted_delta_*_*.json"):
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


def adapter_metrics_path(artifacts_root: Path, task: str, adapter: str, split: str) -> Path:
    metrics_dir = artifacts_root / task / "metrics" / "eval" / split
    filename = f"best_{adapter}_adapter.json"
    direct_path = metrics_dir / filename
    if direct_path.exists():
        return direct_path
    candidates = sorted(metrics_dir.glob(f"best_{adapter}_adapter*.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else direct_path


def base_metrics_path(artifacts_root: Path, task: str, split: str) -> Path:
    metrics_dir = artifacts_root / task / "metrics" / "eval" / split
    direct_path = metrics_dir / "base_model.json"
    if direct_path.exists():
        return direct_path
    candidates = sorted(metrics_dir.glob("base_model*.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else direct_path


def collect_eval_paths(merged_root: Path, split: str) -> dict[str, dict[float, list[Path]]]:
    pattern = re.compile(r"eval_results_merged_weighted_delta_(.+)_lambda([0-9.]+)_.*\.json")
    mapping: dict[str, dict[float, list[Path]]] = {}
    for path in merged_root.glob(f"*/eval/{split}/eval_results_merged_weighted_delta_*_*.json"):
        match = pattern.match(path.name)
        if not match:
            continue
        pair_name = match.group(1)
        lambda_str = match.group(2)
        try:
            lambda_value = float(lambda_str)
        except ValueError:
            continue
        pair_map = mapping.setdefault(pair_name, {})
        pair_map.setdefault(lambda_value, []).append(path)
    return mapping


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def select_latest(paths: list[Path]) -> dict | None:
    best_data = None
    best_timestamp = None
    for path in paths:
        data = load_json(path)
        timestamp = parse_timestamp(data.get("timestamp"))
        if timestamp is None and best_data is None:
            best_data = data
            continue
        if timestamp is not None and (best_timestamp is None or timestamp > best_timestamp):
            best_data = data
            best_timestamp = timestamp
    return best_data


def load_metric_from_eval(data: dict, task: str, metric: str) -> float | None:
    results = data.get("results", {})
    task_metrics = results.get(task)
    if not isinstance(task_metrics, dict):
        return None
    return task_metrics.get(metric)


def collect_series(
    eval_paths: dict[str, dict[float, list[Path]]],
    pair_name: str,
    task_a: str,
    task_b: str,
    metric_map: dict[str, str],
) -> dict[str, list[tuple[float, float]]]:
    series = {task_a: [], task_b: []}
    pair_runs = eval_paths.get(pair_name, {})
    for lambda_value in sorted(pair_runs.keys()):
        eval_data = select_latest(pair_runs[lambda_value])
        if not eval_data:
            continue
        for task in [task_a, task_b]:
            metric = metric_map[task]
            value = load_metric_from_eval(eval_data, task, metric)
            if value is None:
                print(f"Missing {metric} for {task} at lambda {lambda_value} in {pair_name}")
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
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharex=True)
    legend_handles = []
    legend_labels = []
    xticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    xtick_labels = [format_lambda(value) for value in xticks]
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
        line = ax.plot(xs, ys, marker="o", markersize=4, linewidth=2.2, label="Weighted delta merge")[0]
        if "Weighted delta merge" not in legend_labels:
            legend_handles.append(line)
            legend_labels.append("Weighted delta merge")
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
        ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xticks(xticks, labels=xtick_labels)
        ax.set_xlim(0.0, 1.0)
        y_values = ys[:]
        if base_value is not None:
            y_values.append(base_value)
        y_min = min(y_values)
        y_max = max(y_values)
        pad = (y_max - y_min) * 0.2 if y_max > y_min else 0.02
        ax.set_ylim(y_min - pad, y_max + pad)
        if metric_map[task] in {"macro_f1", "accuracy"}:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
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
    output_path = output_dir / f"weighted_delta_{pair_name}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_metric_map(artifacts_root: Path, tasks: set[str], split: str) -> dict[str, str]:
    metric_map = {}
    for task in tasks:
        metrics_path = adapter_metrics_path(artifacts_root, task, task, split)
        if metrics_path.exists():
            metrics = load_json(metrics_path)
        else:
            fallback = adapter_metrics_path(artifacts_root, task, task, "test")
            if not fallback.exists():
                continue
            metrics = load_json(fallback)
        metric_map[task] = pick_metric(task, metrics)
    return metric_map


def build_base_map(
    artifacts_root: Path, tasks: set[str], metric_map: dict[str, str], split: str
) -> dict[str, float | None]:
    base_map: dict[str, float | None] = {}
    for task in tasks:
        base_path = base_metrics_path(artifacts_root, task, split)
        if base_path.exists() and task in metric_map:
            metrics = load_json(base_path)
        else:
            fallback = base_metrics_path(artifacts_root, task, "test")
            if not fallback.exists() or task not in metric_map:
                base_map[task] = None
                continue
            metrics = load_json(fallback)
        base_map[task] = metrics.get(metric_map[task])
    return base_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze weighted delta merging results.")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts"),
        help="Path to artifacts directory.",
    )
    parser.add_argument(
        "--merged-root",
        type=Path,
        default=Path("artifacts/merged/weighted_delta"),
        help="Path to weighted delta merged artifacts directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "validation"],
        help="Dataset split to read (test or validation).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/plots/weighted_delta"),
        help="Directory for output plots.",
    )
    args = parser.parse_args()

    artifacts_root = args.artifacts_root
    merged_root = args.merged_root
    tasks = {p.name for p in artifacts_root.iterdir() if p.is_dir()}
    pairs = collect_pairs(merged_root, tasks, args.split)
    metric_map = build_metric_map(artifacts_root, tasks, args.split)
    base_map = build_base_map(artifacts_root, tasks, metric_map, args.split)
    eval_paths = collect_eval_paths(merged_root, args.split)

    for pair_name, (task_a, task_b) in sorted(pairs.items()):
        if task_a not in metric_map or task_b not in metric_map:
            print(f"Skipping {pair_name}: missing metric map for {task_a} or {task_b}")
            continue
        series = collect_series(eval_paths, pair_name, task_a, task_b, metric_map)
        plot_pair(pair_name, task_a, task_b, series, metric_map, base_map, args.output_dir)
        print(f"Wrote plot for {pair_name}")


if __name__ == "__main__":
    main()
