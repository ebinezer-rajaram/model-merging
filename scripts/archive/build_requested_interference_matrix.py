#!/usr/bin/env python3
"""Build an 8-task cross-adapter matrix with raw metrics and missing cells greyed out."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

from _repo import find_repo_root
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EVAL_TASKS = [
    "asr",
    "emotion",
    "intent",
    "kws",
    "langid",
    "speaker_ver",
    "vocalsound",
    "speech_qa",
]
ADAPTER_TASKS = [
    "asr",
    "emotion",
    "intent",
    "kws",
    "langid",
    "speaker_ver",
    "vocalsound",
]
ADAPTER_COLUMNS = ["base_model", *ADAPTER_TASKS]

# (metric_key, higher_is_better)
TASK_METRICS: Dict[str, Tuple[str, bool]] = {
    "asr": ("wer", False),
    "emotion": ("macro_f1", True),
    "intent": ("accuracy", True),
    "kws": ("macro_f1", True),
    "langid": ("accuracy", True),
    "speaker_ver": ("accuracy", True),
    "speech_qa": ("accuracy", True),
    "vocalsound": ("accuracy", True),
}

EPS = 1e-8


def load_metric(file_path: Path, metric_key: str) -> Optional[float]:
    if not file_path.exists():
        return None
    try:
        data = json.loads(file_path.read_text())
    except Exception:
        return None
    value = data.get(metric_key)
    return float(value) if isinstance(value, (int, float)) else None


def oriented(value: Optional[float], higher_is_better: bool) -> Optional[float]:
    if value is None:
        return None
    return value if higher_is_better else -value


def to_label(task: str) -> str:
    return task.upper().replace("_", " ")


def build_matrices(artifacts_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame(index=EVAL_TASKS, columns=ADAPTER_COLUMNS, dtype=float)
    delta = pd.DataFrame(index=EVAL_TASKS, columns=ADAPTER_COLUMNS, dtype=float)

    for eval_task in EVAL_TASKS:
        metric_key, higher_is_better = TASK_METRICS[eval_task]
        eval_dir = artifacts_root / eval_task / "metrics" / "eval" / "test"

        base_v = load_metric(eval_dir / "base_model.json", metric_key)
        self_v = load_metric(eval_dir / f"best_{eval_task}_adapter.json", metric_key)
        base_s = oriented(base_v, higher_is_better)
        self_s = oriented(self_v, higher_is_better)
        denom = None
        if base_s is not None and self_s is not None:
            d = self_s - base_s
            if abs(d) >= EPS:
                denom = d

        for adapter_col in ADAPTER_COLUMNS:
            if adapter_col == "base_model":
                cell_v = base_v
            else:
                cell_v = load_metric(eval_dir / f"best_{adapter_col}_adapter.json", metric_key)
            raw.loc[eval_task, adapter_col] = np.nan if cell_v is None else cell_v

            cell_s = oriented(cell_v, higher_is_better)
            if adapter_col == "base_model":
                delta.loc[eval_task, adapter_col] = np.nan
            elif cell_s is None or base_s is None or denom is None:
                delta.loc[eval_task, adapter_col] = np.nan
            else:
                delta.loc[eval_task, adapter_col] = (cell_s - base_s) / denom

    raw.index = [to_label(t) for t in raw.index]
    raw.columns = ["BASE MODEL", *[to_label(t) for t in ADAPTER_TASKS]]
    delta.index = [to_label(t) for t in delta.index]
    delta.columns = ["BASE MODEL", *[to_label(t) for t in ADAPTER_TASKS]]
    return raw, delta


def compute_relative_to_base(raw_df: pd.DataFrame) -> pd.DataFrame:
    base_col = "BASE MODEL"
    rel = raw_df.copy()
    for i, row_label in enumerate(raw_df.index):
        task_key = row_label.lower().replace(" ", "_")
        _, higher_is_better = TASK_METRICS[task_key]
        base_value = raw_df.loc[row_label, base_col]
        for col in raw_df.columns:
            current_value = raw_df.loc[row_label, col]
            if pd.isna(current_value) or pd.isna(base_value) or base_value == 0:
                rel.loc[row_label, col] = np.nan
            elif higher_is_better:
                rel.loc[row_label, col] = (current_value - base_value) / abs(base_value)
            else:
                rel.loc[row_label, col] = (base_value - current_value) / abs(base_value)
    return rel


def plot_raw_heatmap(raw_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))

    rel_df = compute_relative_to_base(raw_df)

    # Match existing interference_matrix_raw coloring logic.
    normalized_data = np.zeros_like(rel_df.values, dtype=float)
    power = 0.8
    for i in range(rel_df.shape[0]):
        row_data = rel_df.iloc[i].values
        row_min = np.nanmin(row_data)
        row_max = np.nanmax(row_data)
        if row_max > 0:
            row_bound = row_max
        else:
            row_bound = abs(row_min) if row_min < 0 else 1
        for j in range(len(row_data)):
            if not np.isnan(row_data[j]) and row_bound > 0:
                linear_norm = row_data[j] / row_bound
                sign = np.sign(linear_norm)
                normalized_data[i, j] = sign * (abs(linear_norm) ** power)
            else:
                normalized_data[i, j] = 0 if not np.isnan(row_data[j]) else np.nan

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#bdbdbd")
    data = np.ma.masked_invalid(normalized_data)
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(raw_df.columns)))
    ax.set_yticks(np.arange(len(raw_df.index)))
    ax.set_xticklabels(raw_df.columns, rotation=40, ha="right")
    ax.set_yticklabels(raw_df.index)

    ax.set_xlabel("Adapter Task")
    ax.set_ylabel("Evaluation Task")
    ax.set_title(
        "Task Performance by Adapter\n(Colors: Signed Relative Change from Base | Green = Better | Red = Worse)",
        fontsize=20,
        pad=20,
    )

    ax.set_xticks(np.arange(-0.5, len(raw_df.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(raw_df.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)

    for r in range(raw_df.shape[0]):
        for c in range(raw_df.shape[1]):
            v = raw_df.iat[r, c]
            if np.isfinite(v):
                cv = normalized_data[r, c]
                txt_color = "white" if np.isfinite(cv) and abs(cv) > 0.5 else "black"
                val_str = f"{(v * 100.0):.2f}%"
                ax.text(c, r, val_str, ha="center", va="center", fontsize=12, color=txt_color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Signed Relative Change\n(normalized per row)", rotation=270, labelpad=25, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    ax.set_xlabel("Adapter Used", fontsize=16, labelpad=10)
    ax.set_ylabel("Evaluation Task", fontsize=16, labelpad=10)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo_root = find_repo_root(__file__)
    artifacts_root = repo_root / "artifacts"
    out_dir = repo_root / "analysis" / "results"
    plot_dir = repo_root / "analysis" / "plots" / "interference"

    raw_df, delta_df = build_matrices(artifacts_root)
    raw_pct_df = raw_df * 100.0
    delta_pct_df = delta_df * 100.0

    raw_csv = out_dir / "interference_matrix_requested_raw.csv"
    delta_csv = out_dir / "interference_matrix_requested_delta.csv"
    png_path = plot_dir / "interference_matrix_requested_raw.png"

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_pct_df.to_csv(raw_csv)
    delta_pct_df.to_csv(delta_csv)
    plot_raw_heatmap(raw_df, png_path)

    print(f"Wrote: {raw_csv}")
    print(f"Wrote: {delta_csv}")
    print(f"Wrote: {png_path}")


if __name__ == "__main__":
    main()
