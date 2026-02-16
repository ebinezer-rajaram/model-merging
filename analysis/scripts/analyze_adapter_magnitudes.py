#!/usr/bin/env python3
"""
Analyze the magnitude distribution of best adapters across layers for each task.

This script:
1. Loads the best adapter weights for each task (ASR, emotion, intent, KWS, langid, speaker_id)
2. Computes magnitude statistics (L2 norm) for each layer
3. Visualizes the distribution across layers
4. Compares magnitude patterns between different tasks
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from safetensors.torch import load_file
import torch

# Define tasks and their adapter names
# Will use the 'best' symlink for each adapter
TASK_ADAPTERS = {
    'asr': 'qwen2_5_omni_lora_asr_100h',
    'emotion': 'qwen2_5_omni_lora_emotion_audio_v2',
    'intent': 'qwen2_5_omni_lora_intent',
    'kws': 'qwen2_5_omni_lora_kws',
    'langid': 'qwen2_5_omni_lora_langid',
    'speaker_id': 'qwen2_5_omni_lora_speaker_id_voxceleb2',
}

# LoRA target modules
TARGET_MODULES = [
    'audio_tower.proj',
    'q_proj',
    'k_proj',
    'v_proj',
    'o_proj',
    'gate_proj',
    'up_proj',
    'down_proj',
]


def load_adapter_weights(adapter_path: Path) -> Dict[str, torch.Tensor]:
    """Load adapter weights from safetensors file."""
    safetensors_path = adapter_path / 'adapter_model.safetensors'

    if not safetensors_path.exists():
        raise FileNotFoundError(f"Adapter weights not found at {safetensors_path}")

    print(f"Loading adapter from: {safetensors_path}")
    weights = load_file(str(safetensors_path))
    return weights


def parse_layer_info(weight_name: str) -> Tuple[str, int, str]:
    """
    Parse layer information from weight name.

    Returns:
        (module_type, layer_num, matrix_type)
        e.g., 'model.layers.0.self_attn.q_proj.lora_A.weight' -> ('q_proj', 0, 'lora_A')
    """
    parts = weight_name.split('.')

    # Find layer number
    layer_num = None
    for i, part in enumerate(parts):
        if part == 'layers' and i + 1 < len(parts):
            try:
                layer_num = int(parts[i + 1])
                break
            except ValueError:
                pass

    # Handle audio tower separately (layer -1)
    if 'audio_tower' in weight_name:
        layer_num = -1

    # Find module type (q_proj, k_proj, etc.)
    module_type = None
    for target in TARGET_MODULES:
        if target in weight_name:
            module_type = target
            break

    # Find matrix type (lora_A or lora_B)
    matrix_type = None
    if 'lora_A' in weight_name:
        matrix_type = 'lora_A'
    elif 'lora_B' in weight_name:
        matrix_type = 'lora_B'

    return module_type, layer_num, matrix_type


def compute_magnitude_stats(weights: Dict[str, torch.Tensor]) -> pd.DataFrame:
    """
    Compute magnitude statistics for each layer and module.

    Returns:
        DataFrame with columns: layer, module, matrix_type, l2_norm, mean_abs, max_abs, std
    """
    stats = []

    for weight_name, weight_tensor in weights.items():
        module_type, layer_num, matrix_type = parse_layer_info(weight_name)

        if layer_num is None or module_type is None or matrix_type is None:
            continue

        # Convert to numpy for computation (handle BFloat16)
        if weight_tensor.dtype == torch.bfloat16:
            weight_np = weight_tensor.cpu().float().numpy()
        else:
            weight_np = weight_tensor.cpu().numpy()

        # Compute statistics
        l2_norm = np.linalg.norm(weight_np)
        mean_abs = np.mean(np.abs(weight_np))
        max_abs = np.max(np.abs(weight_np))
        std_val = np.std(weight_np)

        stats.append({
            'layer': layer_num,
            'module': module_type,
            'matrix_type': matrix_type,
            'l2_norm': l2_norm,
            'mean_abs': mean_abs,
            'max_abs': max_abs,
            'std': std_val,
            'num_params': weight_np.size
        })

    return pd.DataFrame(stats)


def aggregate_by_layer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate statistics by layer (combining all modules and matrix types).
    """
    # Group by layer and compute total L2 norm and mean statistics
    layer_stats = df.groupby('layer').agg({
        'l2_norm': 'sum',  # Total L2 norm across all modules in layer
        'mean_abs': 'mean',
        'max_abs': 'max',
        'std': 'mean',
        'num_params': 'sum'
    }).reset_index()

    return layer_stats


def aggregate_by_module(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate statistics by module type (combining all layers).
    """
    module_stats = df.groupby('module').agg({
        'l2_norm': 'sum',
        'mean_abs': 'mean',
        'max_abs': 'max',
        'std': 'mean',
        'num_params': 'sum'
    }).reset_index()

    return module_stats


def plot_magnitude_by_layer(all_task_stats: Dict[str, pd.DataFrame], output_dir: Path):
    """Plot L2 norm magnitude distribution across layers for all tasks."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Line plot of L2 norm by layer
    ax1 = axes[0]
    for task_name, df in all_task_stats.items():
        layer_stats = aggregate_by_layer(df)
        # Filter out audio tower (layer -1) for cleaner visualization
        layer_stats_filtered = layer_stats[layer_stats['layer'] >= 0]
        ax1.plot(layer_stats_filtered['layer'], layer_stats_filtered['l2_norm'],
                marker='o', label=task_name.upper(), linewidth=2, markersize=6)

    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('Total L2 Norm', fontsize=12)
    ax1.set_title('Adapter Magnitude Distribution Across Layers', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Heatmap of normalized magnitudes
    ax2 = axes[1]

    # Create matrix for heatmap
    task_names = list(all_task_stats.keys())
    max_layer = max(df[df['layer'] >= 0]['layer'].max() for df in all_task_stats.values())

    heatmap_data = np.zeros((len(task_names), max_layer + 1))

    for i, (task_name, df) in enumerate(all_task_stats.items()):
        layer_stats = aggregate_by_layer(df)
        layer_stats_filtered = layer_stats[layer_stats['layer'] >= 0]

        for _, row in layer_stats_filtered.iterrows():
            layer_idx = int(row['layer'])
            heatmap_data[i, layer_idx] = row['l2_norm']

    # Normalize each row (task) to [0, 1] for better visualization
    heatmap_data_norm = heatmap_data / heatmap_data.max(axis=1, keepdims=True)

    im = ax2.imshow(heatmap_data_norm, cmap='viridis', aspect='auto')
    ax2.set_xticks(np.arange(max_layer + 1))
    ax2.set_yticks(np.arange(len(task_names)))
    ax2.set_xticklabels(np.arange(max_layer + 1))
    ax2.set_yticklabels([t.upper() for t in task_names])
    ax2.set_xlabel('Layer', fontsize=12)
    ax2.set_ylabel('Task', fontsize=12)
    ax2.set_title('Normalized Adapter Magnitude by Layer (Row-Normalized)', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Normalized L2 Norm', rotation=270, labelpad=20)

    plt.tight_layout()
    output_path = output_dir / 'adapter_magnitude_by_layer.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_path}")
    plt.close()


def plot_magnitude_by_module(all_task_stats: Dict[str, pd.DataFrame], output_dir: Path):
    """Plot magnitude distribution by module type for all tasks."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Prepare data for grouped bar chart
    tasks = list(all_task_stats.keys())
    modules = TARGET_MODULES

    x = np.arange(len(modules))
    width = 0.12  # Width of bars

    for i, (task_name, df) in enumerate(all_task_stats.items()):
        module_stats = aggregate_by_module(df)

        # Align module stats with TARGET_MODULES order
        values = []
        for module in modules:
            module_data = module_stats[module_stats['module'] == module]
            if len(module_data) > 0:
                values.append(module_data['l2_norm'].values[0])
            else:
                values.append(0)

        offset = (i - len(tasks) / 2) * width
        ax.bar(x + offset, values, width, label=task_name.upper())

    ax.set_xlabel('Module Type', fontsize=12)
    ax.set_ylabel('Total L2 Norm', fontsize=12)
    ax.set_title('Adapter Magnitude Distribution by Module Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(modules, rotation=45, ha='right')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'adapter_magnitude_by_module.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_path}")
    plt.close()


def plot_magnitude_distribution(all_task_stats: Dict[str, pd.DataFrame], output_dir: Path):
    """Plot distribution of weight magnitudes for each task."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, (task_name, df) in enumerate(all_task_stats.items()):
        ax = axes[i]

        # Plot histogram of mean_abs values across all layers/modules
        ax.hist(df['mean_abs'], bins=50, alpha=0.7, color=f'C{i}', edgecolor='black')
        ax.set_xlabel('Mean Absolute Weight Value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{task_name.upper()}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add statistics text
        mean_val = df['mean_abs'].mean()
        median_val = df['mean_abs'].median()
        ax.text(0.65, 0.95, f'Mean: {mean_val:.4f}\nMedian: {median_val:.4f}',
               transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Hide unused subplot if odd number of tasks
    if len(all_task_stats) < len(axes):
        for j in range(len(all_task_stats), len(axes)):
            axes[j].axis('off')

    plt.suptitle('Distribution of Adapter Weight Magnitudes', fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'adapter_magnitude_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_path}")
    plt.close()


def plot_lora_ab_comparison(all_task_stats: Dict[str, pd.DataFrame], output_dir: Path):
    """Compare LoRA A and B matrix magnitudes."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Bar chart comparing A vs B magnitudes
    ax1 = axes[0]
    tasks = list(all_task_stats.keys())
    x = np.arange(len(tasks))
    width = 0.35

    lora_a_norms = []
    lora_b_norms = []

    for task_name, df in all_task_stats.items():
        lora_a = df[df['matrix_type'] == 'lora_A']['l2_norm'].sum()
        lora_b = df[df['matrix_type'] == 'lora_B']['l2_norm'].sum()
        lora_a_norms.append(lora_a)
        lora_b_norms.append(lora_b)

    ax1.bar(x - width/2, lora_a_norms, width, label='LoRA A', alpha=0.8)
    ax1.bar(x + width/2, lora_b_norms, width, label='LoRA B', alpha=0.8)
    ax1.set_xlabel('Task', fontsize=12)
    ax1.set_ylabel('Total L2 Norm', fontsize=12)
    ax1.set_title('LoRA A vs B Matrix Magnitudes', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([t.upper() for t in tasks], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Ratio of B/A
    ax2 = axes[1]
    ratios = [b / a if a > 0 else 0 for a, b in zip(lora_a_norms, lora_b_norms)]
    colors = ['green' if r > 1 else 'orange' for r in ratios]

    ax2.bar(x, ratios, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Equal Magnitude')
    ax2.set_xlabel('Task', fontsize=12)
    ax2.set_ylabel('Ratio (LoRA B / LoRA A)', fontsize=12)
    ax2.set_title('LoRA B/A Magnitude Ratio', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([t.upper() for t in tasks], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'lora_ab_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_path}")
    plt.close()


def generate_summary_table(all_task_stats: Dict[str, pd.DataFrame], output_dir: Path):
    """Generate summary statistics table."""
    summary_data = []

    for task_name, df in all_task_stats.items():
        # Overall statistics
        total_l2 = df['l2_norm'].sum()
        mean_l2 = df['l2_norm'].mean()
        total_params = df['num_params'].sum()

        # Layer statistics
        layer_stats = aggregate_by_layer(df)
        num_layers = len(layer_stats[layer_stats['layer'] >= 0])

        # Module statistics
        module_stats = aggregate_by_module(df)
        num_modules = len(module_stats)

        # LoRA A vs B
        lora_a_norm = df[df['matrix_type'] == 'lora_A']['l2_norm'].sum()
        lora_b_norm = df[df['matrix_type'] == 'lora_B']['l2_norm'].sum()

        summary_data.append({
            'Task': task_name.upper(),
            'Total L2 Norm': f'{total_l2:.2f}',
            'Mean L2 Norm': f'{mean_l2:.4f}',
            'Total Parameters': f'{total_params:,}',
            'Num Layers': num_layers,
            'Num Modules': num_modules,
            'LoRA A L2': f'{lora_a_norm:.2f}',
            'LoRA B L2': f'{lora_b_norm:.2f}',
            'B/A Ratio': f'{lora_b_norm / lora_a_norm if lora_a_norm > 0 else 0:.3f}'
        })

    summary_df = pd.DataFrame(summary_data)

    # Print to console
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    # Save to CSV
    csv_path = output_dir / 'adapter_magnitude_summary.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved summary table to: {csv_path}")

    return summary_df


def main():
    print("=" * 100)
    print("ADAPTER MAGNITUDE ANALYSIS")
    print("=" * 100)

    # Setup paths
    artifacts_dir = Path(__file__).parent.parent.parent / 'artifacts'
    output_dir = Path(__file__).parent.parent / 'results'
    plot_dir = Path(__file__).parent.parent / 'plots' / 'adapter_analysis'

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Load and analyze adapters for each task
    all_task_stats = {}

    for task_name, adapter_name in TASK_ADAPTERS.items():
        print(f"\n{'=' * 100}")
        print(f"Analyzing {task_name.upper()} adapter...")
        print(f"{'=' * 100}")

        # Use the 'best' symlink
        full_path = artifacts_dir / task_name / 'adapters' / adapter_name / 'best'

        try:
            # Load adapter weights
            weights = load_adapter_weights(full_path)
            print(f"✓ Loaded {len(weights)} weight tensors")

            # Compute magnitude statistics
            stats_df = compute_magnitude_stats(weights)
            print(f"✓ Computed statistics for {len(stats_df)} layer-module combinations")

            # Store for cross-task analysis
            all_task_stats[task_name] = stats_df

            # Save per-task detailed statistics
            task_csv_path = output_dir / f'adapter_stats_{task_name}.csv'
            stats_df.to_csv(task_csv_path, index=False)
            print(f"✓ Saved detailed statistics to: {task_csv_path}")

        except Exception as e:
            print(f"✗ Error processing {task_name}: {e}")
            continue

    if not all_task_stats:
        print("\n✗ No adapters were successfully loaded. Exiting.")
        return

    # Generate cross-task visualizations
    print(f"\n{'=' * 100}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'=' * 100}")

    plot_magnitude_by_layer(all_task_stats, plot_dir)
    plot_magnitude_by_module(all_task_stats, plot_dir)
    plot_magnitude_distribution(all_task_stats, plot_dir)
    plot_lora_ab_comparison(all_task_stats, plot_dir)

    # Generate summary table
    generate_summary_table(all_task_stats, output_dir)

    print(f"\n{'=' * 100}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 100}")
    print(f"Results saved to: {output_dir}")
    print(f"Plots saved to: {plot_dir}")


if __name__ == '__main__':
    main()
