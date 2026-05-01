#!/usr/bin/env python3
"""Build MMSU per-subtask interference tables from eval JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload)!r}")
    return payload


def _resolve_speech_qa_metrics(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if "subtask_accuracy" in payload:
        return payload

    results = payload.get("results")
    if isinstance(results, Mapping):
        speech_qa = results.get("speech_qa")
        if isinstance(speech_qa, Mapping):
            return speech_qa

    raise ValueError("Could not find Speech-QA metrics with 'subtask_accuracy'.")


def _normalize_subtask_accuracy(raw: Any) -> Dict[str, float]:
    if not isinstance(raw, Mapping):
        raise ValueError("Expected 'subtask_accuracy' to be a JSON object.")
    normalized: Dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, (int, float)):
            normalized[str(key)] = float(value)
    if not normalized:
        raise ValueError("No numeric subtask accuracies found.")
    return normalized


def _parse_variant_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Variant must use NAME=PATH format, got: {value!r}")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Variant name is empty: {value!r}")
    path = Path(raw_path).expanduser()
    return name, path


def build_delta_table(
    *,
    base_subtask_accuracy: Mapping[str, float],
    variant_subtask_accuracy: Mapping[str, Mapping[str, float]],
) -> Dict[str, Dict[str, float]]:
    all_subtasks = set(base_subtask_accuracy.keys())
    for scores in variant_subtask_accuracy.values():
        all_subtasks.update(scores.keys())

    table: Dict[str, Dict[str, float]] = {}
    for subtask in sorted(all_subtasks):
        base_value = float(base_subtask_accuracy.get(subtask, 0.0))
        row: Dict[str, float] = {"base_accuracy": base_value}
        for variant_name, scores in sorted(variant_subtask_accuracy.items()):
            value = float(scores.get(subtask, 0.0))
            row[variant_name] = value - base_value
        table[subtask] = row
    return table


def rank_sensitive_subtasks(delta_table: Mapping[str, Mapping[str, float]]) -> list[Dict[str, Any]]:
    ranked = []
    for subtask, row in delta_table.items():
        delta_values = [
            float(value)
            for key, value in row.items()
            if key != "base_accuracy" and isinstance(value, (int, float))
        ]
        if not delta_values:
            continue
        mean = sum(delta_values) / float(len(delta_values))
        variance = sum((value - mean) ** 2 for value in delta_values) / float(len(delta_values))
        ranked.append(
            {
                "task_name": str(subtask),
                "max_abs_delta": max(abs(value) for value in delta_values),
                "variance": variance,
                "num_variants": len(delta_values),
            }
        )
    ranked.sort(key=lambda row: (row["max_abs_delta"], row["variance"]), reverse=True)
    return ranked


def _write_delta_csv(path: Path, delta_table: Mapping[str, Mapping[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    variant_names: list[str] = []
    for row in delta_table.values():
        for key in row:
            if key not in {"base_accuracy", *variant_names}:
                if key != "base_accuracy":
                    variant_names.append(key)
    variant_names = sorted(name for name in variant_names if name != "base_accuracy")

    fieldnames = ["task_name", "base_accuracy", *variant_names]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for subtask in sorted(delta_table.keys()):
            row = delta_table[subtask]
            payload = {"task_name": subtask, "base_accuracy": float(row.get("base_accuracy", 0.0))}
            for name in variant_names:
                payload[name] = float(row.get(name, 0.0))
            writer.writerow(payload)


def _write_heatmap(path: Path, delta_table: Mapping[str, Mapping[str, float]]) -> None:
    if not delta_table:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    subtasks = sorted(delta_table.keys())
    variants = sorted(
        {
            key
            for row in delta_table.values()
            for key in row.keys()
            if key != "base_accuracy"
        }
    )
    if not variants:
        return

    matrix = np.array(
        [
            [float(delta_table[subtask].get(variant, 0.0)) for variant in variants]
            for subtask in subtasks
        ],
        dtype=float,
    )

    fig_w = max(8.0, 1.1 * len(variants))
    fig_h = max(5.0, 0.3 * len(subtasks) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmax = float(np.max(np.abs(matrix))) if matrix.size else 1.0
    vmax = max(vmax, 1e-6)
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=30, ha="right")
    ax.set_yticks(range(len(subtasks)))
    ax.set_yticklabels(subtasks)
    ax.set_title("MMSU subtask Δaccuracy vs base")
    ax.set_xlabel("Variant")
    ax.set_ylabel("task_name")
    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Δaccuracy")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MMSU subtask interference table from eval metrics JSON files.")
    parser.add_argument("--base", required=True, help="Path to base Speech-QA eval JSON with subtask_accuracy.")
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Variant in NAME=PATH format. Can be provided multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/results/mmsu_subtask_interference",
        help="Directory for CSV/PNG/JSON outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_path = Path(args.base).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_metrics = _resolve_speech_qa_metrics(_load_json(base_path))
    base_subtask_accuracy = _normalize_subtask_accuracy(base_metrics.get("subtask_accuracy"))

    variants: Dict[str, Mapping[str, float]] = {}
    for raw in args.variant:
        name, path = _parse_variant_arg(raw)
        metrics = _resolve_speech_qa_metrics(_load_json(path))
        variants[name] = _normalize_subtask_accuracy(metrics.get("subtask_accuracy"))

    if not variants:
        raise ValueError("Provide at least one --variant NAME=PATH input.")

    delta_table = build_delta_table(
        base_subtask_accuracy=base_subtask_accuracy,
        variant_subtask_accuracy=variants,
    )
    ranked = rank_sensitive_subtasks(delta_table)

    csv_path = output_dir / "mmsu_subtask_delta_accuracy.csv"
    png_path = output_dir / "mmsu_subtask_delta_accuracy.png"
    sensitive_path = output_dir / "mmsu_sensitive_subtasks.json"

    _write_delta_csv(csv_path, delta_table)
    _write_heatmap(png_path, delta_table)
    sensitive_path.write_text(json.dumps(ranked, indent=2), encoding="utf-8")

    print(f"Saved: {csv_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {sensitive_path}")


if __name__ == "__main__":
    main()
