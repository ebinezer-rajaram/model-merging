#!/usr/bin/env python3
"""
Generate interference matrix showing how training adapters for one task
affects performance on other tasks.

The matrix shows: rows = evaluation task, columns = adapter used
"""

import json
import os
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Define tasks and their metrics
TASK_METRICS = {
    'asr': ('wer', 'WER', False),  # (metric_key, display_name, higher_is_better)
    'emotion': ('macro_f1', 'Macro F1', True),
    'intent': ('accuracy', 'Accuracy', True),
    'kws': ('macro_f1', 'Macro F1', True),
    'langid': ('accuracy', 'Accuracy', True),
    'speaker_id': ('accuracy', 'Accuracy', True),
    'speaker_ver': ('accuracy', 'Accuracy', True),
}

# Define adapter names (including base model)
ADAPTERS = ['base_model', 'asr', 'emotion', 'intent', 'kws', 'langid', 'speaker_id', 'speaker_ver']
EPS = 1e-8


def load_metric(task_name, adapter_name):
    """
    Load the metric value for a given task evaluated with a given adapter.

    Args:
        task_name: Name of the task being evaluated
        adapter_name: Name of the adapter being used (or 'base_model')

    Returns:
        float: The metric value, or None if file doesn't exist
    """
    artifacts_dir = Path(__file__).parent.parent.parent / 'artifacts'

    if adapter_name == 'base_model':
        file_name = 'base_model.json'
    else:
        file_name = f'best_{adapter_name}_adapter.json'

    file_path = artifacts_dir / task_name / 'metrics' / 'eval' / 'test' / file_name

    if not file_path.exists():
        print(f"Warning: File not found: {file_path}")
        return None

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        metric_key = TASK_METRICS[task_name][0]

        if metric_key not in data:
            print(f"Warning: Metric '{metric_key}' not found in {file_path}")
            return None

        return data[metric_key]

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def load_metric_from_path(task_name, file_path):
    """Load the metric value for a given task from an explicit metrics file."""
    if not file_path.exists():
        print(f"Warning: File not found: {file_path}")
        return None

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        metric_key = TASK_METRICS[task_name][0]

        if metric_key not in data:
            print(f"Warning: Metric '{metric_key}' not found in {file_path}")
            return None

        return data[metric_key]

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def format_task_name(task_key):
    """Format task name with metric and direction arrow."""
    metric_key, metric_name, higher_is_better = TASK_METRICS[task_key]
    arrow = '↑' if higher_is_better else '↓'
    return f"{task_key.upper()}\n({metric_name} {arrow})"


def format_adapter_name(adapter_name):
    """Format adapter name consistently with task names."""
    return adapter_name.replace('_', ' ').upper()


def extract_task_key(task_label):
    """Extract base task key from formatted task label."""
    return task_label.split('\n')[0].lower()


def oriented_score(task_key, value):
    """Convert metric to a higher-is-better oriented score."""
    if pd.isna(value):
        return np.nan
    _, _, higher_is_better = TASK_METRICS[task_key]
    return value if higher_is_better else -value


def generate_interference_matrix():
    """
    Generate the interference matrix showing cross-task performance.

    Returns:
        pd.DataFrame: Matrix with eval tasks as rows, adapters as columns
    """
    matrix_data = []

    for task_name in TASK_METRICS.keys():
        row = {'Task': format_task_name(task_name)}

        for adapter_name in ADAPTERS:
            metric_value = load_metric(task_name, adapter_name)

            # Format the metric value
            if metric_value is not None:
                # Convert to percentage for display (except WER which is already in good format)
                if TASK_METRICS[task_name][0] == 'wer':
                    row[adapter_name] = metric_value
                else:
                    row[adapter_name] = metric_value
            else:
                row[adapter_name] = np.nan

        matrix_data.append(row)

    df = pd.DataFrame(matrix_data)
    df.set_index('Task', inplace=True)

    # Rename columns for better readability - use same formatting as tasks
    df.columns = [format_adapter_name(col) for col in df.columns]

    return df


def compute_delta_matrix(df, eps=EPS):
    """
    Compute normalized interference deltas relative to base and in-task adapters.

    Delta[i <- j] = (Score_i(adapter_j) - Score_i(base)) / (Score_i(adapter_i) - Score_i(base))
    """
    delta_data = []
    base_col = format_adapter_name('base_model')

    for task_label in df.index:
        row = {'Task': task_label}
        task_key = extract_task_key(task_label)
        task_adapter_col = format_adapter_name(task_key)

        base_value = df.loc[task_label, base_col] if base_col in df.columns else np.nan
        base_score = oriented_score(task_key, base_value)

        task_adapter_value = df.loc[task_label, task_adapter_col] if task_adapter_col in df.columns else np.nan
        task_adapter_score = oriented_score(task_key, task_adapter_value)

        denom = task_adapter_score - base_score if not pd.isna(task_adapter_score) and not pd.isna(base_score) else np.nan
        denom_valid = not pd.isna(denom) and abs(denom) >= eps

        for col in df.columns:
            if col == base_col:
                row[col] = np.nan
                continue

            current_value = df.loc[task_label, col]
            current_score = oriented_score(task_key, current_value)

            if pd.isna(current_score) or pd.isna(base_score) or not denom_valid:
                row[col] = np.nan
            else:
                row[col] = (current_score - base_score) / denom

        delta_data.append(row)

    delta_df = pd.DataFrame(delta_data)
    delta_df.set_index('Task', inplace=True)
    return delta_df


def compute_conflict_matrix(delta_df):
    """Compute directional conflict: Conflict = max(0, -Delta)."""
    conflict_df = delta_df.copy()
    conflict_df[:] = np.maximum(0, -delta_df.values)
    return conflict_df


def compute_interference_scores(df):
    """
    Compute interference scores: how much does using adapter X hurt performance on task Y?

    Positive values = interference (performance degraded)
    Negative values = positive transfer (performance improved)

    Args:
        df: DataFrame with raw performance values

    Returns:
        pd.DataFrame: Interference scores relative to task-specific adapter
    """
    interference_data = []

    for idx, task_name in enumerate(df.index):
        row = {'Task': task_name}

        # Extract the base task key from the formatted name (e.g., "ASR\n(WER ↓)" -> "asr")
        task_key = task_name.split('\n')[0].lower()
        metric_key, metric_name, higher_is_better = TASK_METRICS[task_key]

        # Find the task-specific adapter column (should have same base name)
        task_adapter_col = None
        for col in df.columns:
            col_key = col.split('\n')[0].lower().replace(' ', '_')
            if col_key == task_key:
                task_adapter_col = col
                break

        if task_adapter_col is None:
            print(f"Warning: Could not find adapter column for {task_name}")
            continue

        best_value = df.loc[task_name, task_adapter_col]

        # Compute interference for each adapter
        for col in df.columns:
            current_value = df.loc[task_name, col]

            if pd.isna(current_value) or pd.isna(best_value):
                row[col] = np.nan
            else:
                # For WER (lower is better), interference = current - best
                # For others (higher is better), interference = best - current
                if higher_is_better:
                    interference = best_value - current_value
                else:
                    interference = current_value - best_value

                row[col] = interference

        interference_data.append(row)

    interference_df = pd.DataFrame(interference_data)
    interference_df.set_index('Task', inplace=True)

    return interference_df


def compute_relative_to_base(df):
    """
    Compute signed relative change from base model.

    For each task:
    - Positive values = improvement over base
    - Negative values = degradation from base
    - Values are in relative change format: (current - base) / base

    Args:
        df: DataFrame with raw performance values

    Returns:
        pd.DataFrame: Signed relative change scores
    """
    relative_data = []
    base_col = 'BASE MODEL'

    for idx, task_name in enumerate(df.index):
        row = {'Task': task_name}

        # Extract the base task key from the formatted name
        task_key = task_name.split('\n')[0].lower()
        metric_key, metric_name, higher_is_better = TASK_METRICS[task_key]

        # Get base model performance
        base_value = df.loc[task_name, base_col]

        # Compute signed relative change for each adapter
        for col in df.columns:
            current_value = df.loc[task_name, col]

            if pd.isna(current_value) or pd.isna(base_value) or base_value == 0:
                row[col] = np.nan
            else:
                # Compute relative change
                if higher_is_better:
                    # For metrics where higher is better: positive change = improvement
                    relative_change = (current_value - base_value) / abs(base_value)
                else:
                    # For WER (lower is better): positive change = improvement (reduction)
                    relative_change = (base_value - current_value) / abs(base_value)

                row[col] = relative_change

        relative_data.append(row)

    relative_df = pd.DataFrame(relative_data)
    relative_df.set_index('Task', inplace=True)

    return relative_df


def plot_heatmap_sequential_row_normalized(df_colors, df_annot, title, output_path, cmap='Reds'):
    """
    Create a heatmap with row-wise normalized sequential colors.

    Args:
        df_colors: DataFrame for colors (will be normalized per row, min=0)
        df_annot: DataFrame for annotations (absolute values)
        title: Plot title
        output_path: Where to save the plot
        cmap: Matplotlib colormap name
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    normalized_data = np.zeros_like(df_colors.values, dtype=float)

    for i, row_name in enumerate(df_colors.index):
        row_data = df_colors.loc[row_name].values
        row_max = np.nanmax(row_data)

        for j, val in enumerate(row_data):
            if not np.isnan(val):
                normalized_data[i, j] = val / row_max if row_max > 0 else 0
            else:
                normalized_data[i, j] = np.nan

    im = ax.imshow(normalized_data, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Conflict\n(per row)', rotation=270, labelpad=25, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    ax.set_xticks(np.arange(len(df_colors.columns)))
    ax.set_yticks(np.arange(len(df_colors.index)))
    ax.set_xticklabels(df_colors.columns, fontsize=13)
    ax.set_yticklabels(df_colors.index, fontsize=13)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_xticks(np.arange(len(df_colors.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(df_colors.index)) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)

    for i in range(len(df_colors.index)):
        for j in range(len(df_colors.columns)):
            val = df_annot.iloc[i, j]
            if not np.isnan(val):
                text_color = 'white' if normalized_data[i, j] > 0.5 else 'black'
                if val == 0:
                    val_str = '0.00'
                else:
                    from math import log10, floor
                    magnitude = floor(log10(abs(val)))
                    if magnitude < 0:
                        decimals = -magnitude + 2
                    else:
                        decimals = 2
                    val_str = f'{val:.{decimals}f}'

                ax.text(j, i, val_str, ha="center", va="center", color=text_color, fontsize=12)

    ax.set_title(title, fontsize=20, pad=20)
    ax.set_xlabel('Adapter Used', fontsize=16, labelpad=10)
    ax.set_ylabel('Evaluation Task', fontsize=16, labelpad=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_path}")
    plt.close()


def rank_conflicting_pairs(conflict_df, top_k=10):
    """
    Rank task pairs by total conflict, skipping pairs with missing values.
    """
    rows = []
    task_labels = list(conflict_df.index)
    task_keys = [extract_task_key(label) for label in task_labels]

    for i in range(len(task_keys)):
        for j in range(i + 1, len(task_keys)):
            task_i = task_keys[i]
            task_j = task_keys[j]
            row_i = task_labels[i]
            row_j = task_labels[j]
            col_i = format_adapter_name(task_i)
            col_j = format_adapter_name(task_j)

            conflict_i_j = conflict_df.loc[row_i, col_j] if col_j in conflict_df.columns else np.nan
            conflict_j_i = conflict_df.loc[row_j, col_i] if col_i in conflict_df.columns else np.nan

            if not np.isfinite(conflict_i_j) or not np.isfinite(conflict_j_i):
                continue

            total_conflict = conflict_i_j + conflict_j_i
            asymmetry = abs(conflict_i_j - conflict_j_i)

            rows.append({
                'task_i': task_i,
                'task_j': task_j,
                'conflict_i<-j': conflict_i_j,
                'conflict_j<-i': conflict_j_i,
                'total_conflict': total_conflict,
                'asymmetry': asymmetry,
            })

    if not rows:
        return pd.DataFrame()

    ranking_df = pd.DataFrame(rows).sort_values('total_conflict', ascending=False)
    return ranking_df.head(top_k)


def resolve_pair(pair, tasks):
    """Resolve a merged pair string into task names if possible."""
    for task in tasks:
        prefix = f"{task}_"
        if pair.startswith(prefix):
            other = pair[len(prefix):]
            if other in tasks:
                return task, other
    return None


def collect_weighted_merge_entries(artifacts_root):
    """Collect merged weighted adapter metrics files grouped by pair and lambda."""
    tasks = {p.name for p in artifacts_root.iterdir() if p.is_dir()}
    entries = {}
    pattern = re.compile(r"best_merged_weighted_(.+)_lambda([0-9.]+)_adapter\.json")

    for path in artifacts_root.glob("*/metrics/eval/test/best_merged_weighted_*_adapter.json"):
        match = pattern.match(path.name)
        if not match:
            continue
        pair_name = match.group(1)
        lambda_value = float(match.group(2))
        resolved = resolve_pair(pair_name, tasks)
        if not resolved:
            continue
        task_name = path.parents[3].name
        key = (pair_name, lambda_value, resolved)
        entries.setdefault(key, {})[task_name] = path

    return entries


def compute_conflict_value(task_key, base_value, task_adapter_value, current_value, eps=EPS):
    """Compute directional conflict for a single evaluation task and adapter value."""
    base_score = oriented_score(task_key, base_value)
    task_adapter_score = oriented_score(task_key, task_adapter_value)
    current_score = oriented_score(task_key, current_value)

    if pd.isna(base_score) or pd.isna(task_adapter_score) or pd.isna(current_score):
        return np.nan

    denom = task_adapter_score - base_score
    if abs(denom) < eps:
        return np.nan

    delta = (current_score - base_score) / denom
    return max(0, -delta)


def compute_weighted_merge_conflict_reduction(conflict_df, performance_matrix, output_dir):
    """Compute conflict reduction for weighted merged adapters if present."""
    artifacts_root = Path(__file__).parent.parent.parent / 'artifacts'
    entries = collect_weighted_merge_entries(artifacts_root)
    if not entries:
        return

    base_col = format_adapter_name('base_model')
    rows = []

    for (pair_name, lambda_value, (task_a, task_b)), task_paths in sorted(entries.items()):
        if task_a not in task_paths or task_b not in task_paths:
            continue

        for task_i, task_j in [(task_a, task_b), (task_b, task_a)]:
            task_label = format_task_name(task_i)
            col_j = format_adapter_name(task_j)

            base_value = performance_matrix.loc[task_label, base_col] if base_col in performance_matrix.columns else np.nan
            task_adapter_value = performance_matrix.loc[task_label, format_adapter_name(task_i)]

            merged_value = load_metric_from_path(task_i, task_paths[task_i])
            conflict_adapter = conflict_df.loc[task_label, col_j] if col_j in conflict_df.columns else np.nan
            conflict_merged = compute_conflict_value(task_i, base_value, task_adapter_value, merged_value)

            if not np.isfinite(conflict_adapter) or conflict_adapter == 0:
                conflict_reduction = np.nan
            else:
                conflict_reduction = (conflict_adapter - conflict_merged) / conflict_adapter

            rows.append({
                'pair': pair_name,
                'lambda': lambda_value,
                'task_i': task_i,
                'task_j': task_j,
                'conflict_adapter_i<-j': conflict_adapter,
                'conflict_merged_i<-j': conflict_merged,
                'conflict_reduction_i<-j': conflict_reduction,
            })

    if not rows:
        return

    reduction_df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'conflict_reduction_weighted.csv'
    reduction_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved weighted merge conflict reductions to: {output_path}")

    # Summary per pair/lambda using both directions when available.
    summary_rows = []
    for (pair_name, lambda_value), group in reduction_df.groupby(['pair', 'lambda']):
        pivot = {(row['task_i'], row['task_j']): row for _, row in group.iterrows()}
        resolved = resolve_pair(pair_name, set(TASK_METRICS.keys()))
        if not resolved:
            continue
        task_a, task_b = resolved
        row_ab = pivot.get((task_a, task_b))
        row_ba = pivot.get((task_b, task_a))
        if row_ab is None or row_ba is None:
            continue
        if not np.isfinite(row_ab['conflict_adapter_i<-j']) or not np.isfinite(row_ba['conflict_adapter_i<-j']):
            continue
        total_adapter = row_ab['conflict_adapter_i<-j'] + row_ba['conflict_adapter_i<-j']
        total_merged = row_ab['conflict_merged_i<-j'] + row_ba['conflict_merged_i<-j']
        if total_adapter == 0 or not np.isfinite(total_adapter) or not np.isfinite(total_merged):
            total_reduction = np.nan
        else:
            total_reduction = (total_adapter - total_merged) / total_adapter
        summary_rows.append({
            'pair': pair_name,
            'lambda': lambda_value,
            'total_conflict_adapter': total_adapter,
            'total_conflict_merged': total_merged,
            'total_conflict_reduction': total_reduction,
        })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = output_dir / 'conflict_reduction_weighted_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"✓ Saved weighted merge conflict reduction summary to: {summary_path}")


def plot_heatmap_direct(df_colors, df_annot, title, output_path):
    """
    Create a heatmap with direct color mapping using per-row normalization.

    Args:
        df_colors: DataFrame for colors (diverging colormap centered at 0, normalized per row)
        df_annot: DataFrame for annotations (absolute values)
        title: Plot title
        output_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Normalize each row separately for coloring with non-linear scaling
    # The best (max) value in each row should map to +1 (greenest)
    # The worst (min) value in each row should map to -1 (reddest)
    normalized_data = np.zeros_like(df_colors.values, dtype=float)
    power = 0.8  # Power transform for faster color saturation

    for i, row_name in enumerate(df_colors.index):
        row_data = df_colors.iloc[i].values
        row_min = np.nanmin(row_data)
        row_max = np.nanmax(row_data)

        # Always use row_max as the bound so the best value gets full green
        # This ensures the max value maps to +1
        if row_max > 0:
            row_bound = row_max
        else:
            # If all values are negative/zero, use abs(min) as bound
            row_bound = abs(row_min) if row_min < 0 else 1

        # Normalize this row with power transform
        for j in range(len(row_data)):
            if not np.isnan(row_data[j]) and row_bound > 0:
                # Normalize so max value -> +1
                linear_norm = row_data[j] / row_bound
                sign = np.sign(linear_norm)
                normalized_data[i, j] = sign * (abs(linear_norm) ** power)
            else:
                normalized_data[i, j] = 0 if not np.isnan(row_data[j]) else np.nan

    # Create diverging colormap centered at 0
    cmap = plt.cm.RdYlGn
    im = ax.imshow(normalized_data, cmap=cmap, aspect='auto', vmin=-1, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Signed Relative Change\n(normalized per row)', rotation=270, labelpad=25, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Set ticks
    ax.set_xticks(np.arange(len(df_colors.columns)))
    ax.set_yticks(np.arange(len(df_colors.index)))
    ax.set_xticklabels(df_colors.columns, fontsize=13)
    ax.set_yticklabels(df_colors.index, fontsize=13)

    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add grid
    ax.set_xticks(np.arange(len(df_colors.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(df_colors.index)) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)

    # Add text annotations with absolute values
    for i in range(len(df_colors.index)):
        for j in range(len(df_colors.columns)):
            val = df_annot.iloc[i, j]
            if not np.isnan(val):
                # Choose text color based on normalized color intensity
                text_color = 'white' if abs(normalized_data[i, j]) > 0.5 else 'black'

                # Format to 3 significant figures, preserving trailing zeros
                if val == 0:
                    val_str = '0.00'
                else:
                    from math import log10, floor
                    magnitude = floor(log10(abs(val)))
                    if magnitude < 0:
                        decimals = -magnitude + 2
                    else:
                        decimals = 2
                    val_str = f'{val:.{decimals}f}'

                text = ax.text(j, i, val_str,
                             ha="center", va="center", color=text_color, fontsize=12)

    ax.set_title(title, fontsize=20, pad=20)
    ax.set_xlabel('Adapter Used', fontsize=16, labelpad=10)
    ax.set_ylabel('Evaluation Task', fontsize=16, labelpad=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_path}")
    plt.close()


def plot_heatmap_row_normalized(df_colors, df_annot, title, output_path):
    """
    Create a heatmap with row-wise normalized colors but absolute value annotations.

    Args:
        df_colors: DataFrame for colors (will be normalized per row)
        df_annot: DataFrame for annotations (absolute values)
        title: Plot title
        output_path: Where to save the plot
    """
    from matplotlib.colors import TwoSlopeNorm
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(14, 8))

    # Normalize each row separately
    # For each row, find min/max and create a colormap centered at 0
    normalized_data = np.zeros_like(df_colors.values, dtype=float)

    for i, row_name in enumerate(df_colors.index):
        row_data = df_colors.loc[row_name].values
        row_min = np.nanmin(row_data)
        row_max = np.nanmax(row_data)

        # Normalize this row to [-1, 1] range with 0 centered
        for j, val in enumerate(row_data):
            if not np.isnan(val):
                if val > 0:
                    # Positive values: scale to [0, 1]
                    normalized_data[i, j] = val / row_max if row_max > 0 else 0
                elif val < 0:
                    # Negative values: scale to [-1, 0]
                    normalized_data[i, j] = val / abs(row_min) if row_min < 0 else 0
                else:
                    normalized_data[i, j] = 0
            else:
                normalized_data[i, j] = np.nan

    # Create the heatmap using the normalized data for colors
    # but show the absolute values as annotations
    cmap = plt.cm.RdYlGn

    im = ax.imshow(normalized_data, cmap=cmap, aspect='auto', vmin=-1, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Improvement\n(per row)', rotation=270, labelpad=25, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Set ticks
    ax.set_xticks(np.arange(len(df_colors.columns)))
    ax.set_yticks(np.arange(len(df_colors.index)))
    ax.set_xticklabels(df_colors.columns, fontsize=13)
    ax.set_yticklabels(df_colors.index, fontsize=13)

    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add grid
    ax.set_xticks(np.arange(len(df_colors.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(df_colors.index)) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)

    # Add text annotations with absolute values
    for i in range(len(df_colors.index)):
        for j in range(len(df_colors.columns)):
            val = df_annot.iloc[i, j]
            if not np.isnan(val):
                # Choose text color based on normalized background color
                text_color = 'white' if abs(normalized_data[i, j]) > 0.5 else 'black'

                # Format to 3 significant figures, preserving trailing zeros
                # 0.5600 -> "0.560", 0.0202 -> "0.0202", 0.999 -> "0.999"
                if val == 0:
                    val_str = '0.00'
                else:
                    from math import log10, floor
                    # Calculate number of significant figures
                    magnitude = floor(log10(abs(val)))
                    # For values < 1, we need extra decimals for leading zeros
                    if magnitude < 0:
                        decimals = -magnitude + 2  # 2 more decimals after first sig fig
                    else:
                        decimals = 2  # For values >= 1
                    val_str = f'{val:.{decimals}f}'

                text = ax.text(j, i, val_str,
                             ha="center", va="center", color=text_color, fontsize=12)

    ax.set_title(title, fontsize=20, pad=20)
    ax.set_xlabel('Adapter Used', fontsize=16, labelpad=10)
    ax.set_ylabel('Evaluation Task', fontsize=16, labelpad=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_path}")
    plt.close()


def plot_heatmap_global_power(df_colors, df_annot, title, output_path, power=0.8):
    """
    Create a heatmap with global color normalization and power-law scaling.

    Args:
        df_colors: DataFrame for colors (diverging colormap centered at 0)
        df_annot: DataFrame for annotations (absolute values)
        title: Plot title
        output_path: Where to save the plot
        power: Power transform for faster color saturation
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    values = df_colors.values.astype(float)
    # Clamp for coloring so +1/-1 are the saturation endpoints.
    values_clipped = np.clip(values, -1, 1)

    normalized_data = np.zeros_like(values, dtype=float)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values_clipped[i, j]
            if np.isnan(val):
                normalized_data[i, j] = np.nan
            else:
                sign = np.sign(val)
                normalized_data[i, j] = sign * (abs(val) ** power)

    cmap = plt.cm.RdYlGn
    im = ax.imshow(normalized_data, cmap=cmap, aspect='auto', vmin=-1, vmax=1)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Signed Delta\n(global, power-scaled)', rotation=270, labelpad=25, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    ax.set_xticks(np.arange(len(df_colors.columns)))
    ax.set_yticks(np.arange(len(df_colors.index)))
    ax.set_xticklabels(df_colors.columns, fontsize=13)
    ax.set_yticklabels(df_colors.index, fontsize=13)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_xticks(np.arange(len(df_colors.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(df_colors.index)) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)

    for i in range(len(df_colors.index)):
        for j in range(len(df_colors.columns)):
            val = df_annot.iloc[i, j]
            if not np.isnan(val):
                text_color = 'white' if abs(normalized_data[i, j]) > 0.5 else 'black'
                if val == 0:
                    val_str = '0.00'
                else:
                    from math import log10, floor
                    magnitude = floor(log10(abs(val)))
                    if magnitude < 0:
                        decimals = -magnitude + 2
                    else:
                        decimals = 2
                    val_str = f'{val:.{decimals}f}'

                ax.text(j, i, val_str, ha="center", va="center", color=text_color, fontsize=12)

    ax.set_title(title, fontsize=20, pad=20)
    ax.set_xlabel('Adapter Used', fontsize=16, labelpad=10)
    ax.set_ylabel('Evaluation Task', fontsize=16, labelpad=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_path}")
    plt.close()


def main():
    print("Generating Interference Matrix...")
    print("=" * 80)

    # Generate raw performance matrix
    performance_matrix = generate_interference_matrix()

    print("\n" + "=" * 80)
    print("RAW PERFORMANCE MATRIX")
    print("=" * 80)
    print("Rows: Evaluation Task | Columns: Adapter Used\n")
    print(performance_matrix.to_string())

    # Save to CSV
    csv_output_dir = Path(__file__).parent.parent / 'results'
    csv_output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = csv_output_dir / 'interference_matrix_raw.csv'
    performance_matrix.to_csv(csv_path)
    print(f"\n✓ Saved raw performance matrix to: {csv_path}")

    # Compute performance relative to base model
    relative_to_base = compute_relative_to_base(performance_matrix)

    print("\n" + "=" * 80)
    print("SIGNED RELATIVE CHANGE FROM BASE MODEL")
    print("=" * 80)
    print("Positive = Better than base | Negative = Worse than base")
    print("Values show relative change: (current - base) / |base|\n")
    print(relative_to_base.to_string())

    csv_path_relative = csv_output_dir / 'interference_matrix_relative_to_base.csv'
    relative_to_base.to_csv(csv_path_relative)
    print(f"\n✓ Saved relative to base matrix to: {csv_path_relative}")

    # Create plots directory
    plot_output_dir = Path(__file__).parent.parent / 'plots' / 'interference'
    plot_output_dir.mkdir(parents=True, exist_ok=True)

    # Plot raw performance matrix colored by signed relative change (directly, not row-normalized)
    # Use a simpler plotting function for direct color mapping
    plot_heatmap_direct(
        relative_to_base,  # Use signed relative change for colors (directly, centered at 0)
        performance_matrix,  # Show absolute values in annotations
        'Task Performance by Adapter\n(Colors: Signed Relative Change from Base | Green = Better | Red = Worse)',
        plot_output_dir / 'interference_matrix_raw.png'
    )

    # Generate interference scores
    print("\n" + "=" * 80)
    print("INTERFERENCE SCORES")
    print("=" * 80)
    print("Positive = Performance degradation (interference)")
    print("Negative = Performance improvement (positive transfer)")
    print("Relative to task-specific adapter performance\n")

    interference_scores = compute_interference_scores(performance_matrix)
    print(interference_scores.to_string())

    csv_path_interference = csv_output_dir / 'interference_matrix_scores.csv'
    interference_scores.to_csv(csv_path_interference)
    print(f"\n✓ Saved interference scores to: {csv_path_interference}")

    # Compute delta and conflict matrices
    print("\n" + "=" * 80)
    print("NORMALIZED INTERFERENCE DELTA")
    print("=" * 80)
    print("Delta[i <- j] = (Score_i(adapter_j) - Score_i(base)) / (Score_i(adapter_i) - Score_i(base))")
    print("NaN if in-task improvement is ~0 or missing\n")

    delta_matrix = compute_delta_matrix(performance_matrix)
    print(delta_matrix.to_string())

    csv_path_delta = csv_output_dir / 'interference_matrix_delta.csv'
    delta_matrix.to_csv(csv_path_delta)
    print(f"\n✓ Saved delta matrix to: {csv_path_delta}")

    conflict_matrix = compute_conflict_matrix(delta_matrix)
    csv_path_conflict = csv_output_dir / 'interference_matrix_conflict.csv'
    conflict_matrix.to_csv(csv_path_conflict)
    print(f"✓ Saved conflict matrix to: {csv_path_conflict}")

    # Plot interference scores with row-normalized diverging colormap
    # Red = interference (bad), Green = positive transfer (good)
    plot_heatmap_row_normalized(
        interference_scores,
        interference_scores,  # Show same values for both color and annotation
        'Interference Matrix\n(Relative to Task-Specific Adapter | Normalized per Task)\nRed = Interference | Green = Positive Transfer',
        plot_output_dir / 'interference_matrix_scores.png'
    )

    # Plot delta and conflict heatmaps (drop base model column)
    base_col = format_adapter_name('base_model')
    delta_plot = delta_matrix.drop(columns=[base_col], errors='ignore')
    conflict_plot = conflict_matrix.drop(columns=[base_col], errors='ignore')

    # Use simpler labels for the delta heatmap.
    delta_plot.index = [label.split('\n')[0] for label in delta_plot.index]
    plot_heatmap_global_power(
        delta_plot,
        delta_plot,
        'Delta Matrix\n(Normalized by In-Task Improvement)',
        plot_output_dir / 'interference_matrix_delta.png'
    )

    plot_heatmap_sequential_row_normalized(
        conflict_plot,
        conflict_plot,
        'Conflict Matrix\n(Conflict = max(0, -Delta) | Normalized per Task)',
        plot_output_dir / 'interference_matrix_conflict.png'
    )

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Diagonal values (should be 0 or close to 0)
    print("\nDiagonal (task-specific adapter on own task):")
    for task in interference_scores.index:
        # Extract base task key from formatted name
        task_key = task.split('\n')[0].lower()
        task_col = None
        for col in interference_scores.columns:
            col_key = col.split('\n')[0].lower().replace(' ', '_')
            if col_key == task_key:
                task_col = col
                break

        if task_col and task_col in interference_scores.columns:
            val = interference_scores.loc[task, task_col]
            print(f"  {task_key}: {val:.6f}")

    # Average interference per adapter (excluding diagonal)
    print("\nAverage interference per adapter (excluding diagonal):")
    for col in interference_scores.columns:
        # Exclude diagonal values
        values = []
        for task in interference_scores.index:
            # Extract base keys for comparison
            task_key = task.split('\n')[0].lower()
            col_key = col.split('\n')[0].lower().replace(' ', '_')

            if col_key != task_key:
                val = interference_scores.loc[task, col]
                if not pd.isna(val):
                    values.append(val)

        if values:
            avg_interference = np.mean(values)
            col_display = col.split('\n')[0]  # Just show the base name without metric
            print(f"  {col_display}: {avg_interference:.6f}")

    print("\n" + "=" * 80)
    print("TOP CONFLICTING TASK PAIRS")
    print("=" * 80)
    top_pairs = rank_conflicting_pairs(conflict_matrix, top_k=10)
    if top_pairs.empty:
        print("No valid task pairs found for ranking.")
    else:
        print(top_pairs.to_string(index=False))
        top_pairs_path = csv_output_dir / 'conflict_topk.csv'
        top_pairs.to_csv(top_pairs_path, index=False)
        print(f"\n✓ Saved top conflict pairs to: {top_pairs_path}")

    # Optional: merged conflict reduction reporting (weighted merges)
    merge_results_dir = Path(__file__).parent.parent / 'results' / 'weighted'
    compute_weighted_merge_conflict_reduction(conflict_matrix, performance_matrix, merge_results_dir)

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == '__main__':
    main()
