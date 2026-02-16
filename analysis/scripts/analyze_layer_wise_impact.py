#!/usr/bin/env python3
"""
Analyze layer-wise impact of task vectors across transformer layers.

This script:
1. Extracts task vectors for each adapter
2. Computes L2 norm for each layer (aggregating all modules within a layer)
3. Creates a heatmap showing layer × task impact
4. Creates line plots showing layer profiles
5. Identifies which layers are most affected by each task
"""

import json
import os
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from safetensors import safe_open

# Define tasks and their adapter names
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


def extract_task_vector_from_lora(adapter_path: Path) -> Dict[str, torch.Tensor]:
    """Extract task vector from LoRA adapter.

    For LoRA, the adapted weight is: W' = W + (B @ A) * scaling
    Where scaling = lora_alpha / r

    The task vector (weight delta) is: delta_W = (B @ A) * scaling

    Args:
        adapter_path: Path to LoRA adapter directory

    Returns:
        Dictionary mapping layer names to task vectors (delta weights)
    """
    # Load adapter config
    config_path = adapter_path / "adapter_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    lora_alpha = config["lora_alpha"]
    lora_r = config["r"]
    scaling = lora_alpha / lora_r

    print(f"  📊 Adapter config: r={lora_r}, alpha={lora_alpha}, scaling={scaling:.4f}")

    # Load LoRA weights
    safetensors_path = adapter_path / "adapter_model.safetensors"
    lora_weights = {}

    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            lora_weights[key] = f.get_tensor(key).clone()

    # Group lora_A and lora_B by base parameter name
    lora_pairs = {}
    for key in lora_weights.keys():
        if ".lora_A." in key:
            base_key = key.replace(".lora_A.weight", "")
            if base_key not in lora_pairs:
                lora_pairs[base_key] = {}
            lora_pairs[base_key]["A"] = lora_weights[key]
        elif ".lora_B." in key:
            base_key = key.replace(".lora_B.weight", "")
            if base_key not in lora_pairs:
                lora_pairs[base_key] = {}
            lora_pairs[base_key]["B"] = lora_weights[key]

    # Compute task vectors: delta_W = B @ A * scaling
    task_vectors = {}
    for base_key, pair in lora_pairs.items():
        if "A" in pair and "B" in pair:
            A = pair["A"]  # shape: (r, in_features)
            B = pair["B"]  # shape: (out_features, r)

            # Compute delta_W = B @ A * scaling
            delta_W = (B @ A) * scaling
            task_vectors[base_key] = delta_W
        else:
            print(f"  ⚠️  Warning: Incomplete LoRA pair for {base_key}")

    print(f"  ✓ Extracted {len(task_vectors)} task vectors")

    return task_vectors


def parse_layer_info(param_name: str) -> Tuple[str, int, str]:
    """Extract layer number and module type from parameter name.

    Args:
        param_name: Full parameter name from task vector

    Returns:
        Tuple of (module_type, layer_num, component)
    """
    parts = param_name.split('.')

    # Handle audio tower (layer -1)
    if 'audio_tower' in param_name:
        return 'audio_tower.proj', -1, 'audio_tower'

    # Find layer number
    layer_num = None
    for i, part in enumerate(parts):
        if part == 'layers' and i + 1 < len(parts):
            try:
                layer_num = int(parts[i + 1])
                break
            except ValueError:
                pass

    # Find module type
    module_type = None
    component = None

    # Attention modules
    if any(x in param_name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
        for mod in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            if mod in param_name:
                module_type = mod
                component = 'attention'
                break

    # MLP modules
    elif any(x in param_name for x in ['gate_proj', 'up_proj', 'down_proj']):
        for mod in ['gate_proj', 'up_proj', 'down_proj']:
            if mod in param_name:
                module_type = mod
                component = 'mlp'
                break

    return module_type, layer_num, component


def compute_layer_wise_statistics(
    task_vectors: Dict[str, Dict[str, torch.Tensor]]
) -> pd.DataFrame:
    """Compute L2 norm statistics for each layer and task.

    Args:
        task_vectors: Dict mapping task names to their task vectors

    Returns:
        DataFrame with columns: task, layer, module, component, l2_norm, num_elements
    """
    stats = []

    for task_name, task_vector in task_vectors.items():
        for param_name, delta_W in task_vector.items():
            module_type, layer_num, component = parse_layer_info(param_name)

            if layer_num is None or module_type is None:
                continue  # Skip unparseable parameters

            # Convert to numpy
            if delta_W.dtype == torch.bfloat16:
                delta_W_np = delta_W.cpu().float().numpy()
            else:
                delta_W_np = delta_W.cpu().numpy()

            # Compute L2 norm
            l2_norm = np.linalg.norm(delta_W_np)

            stats.append({
                'task': task_name,
                'layer': layer_num,
                'module': module_type,
                'component': component,
                'l2_norm': l2_norm,
                'num_elements': delta_W_np.size
            })

    return pd.DataFrame(stats)


def aggregate_layer_task_matrix(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate statistics to create layer × task matrix.

    Aggregation: Sum L2 norms across all modules within each layer.

    Args:
        stats_df: DataFrame from compute_layer_wise_statistics()

    Returns:
        DataFrame with layers as rows, tasks as columns
    """
    # Sum L2 norms across all modules in each layer-task combination
    layer_task_agg = stats_df.groupby(['layer', 'task'])['l2_norm'].sum().reset_index()

    # Pivot to matrix format
    layer_task_df = layer_task_agg.pivot(
        index='layer',
        columns='task',
        values='l2_norm'
    )

    # Uppercase column names
    layer_task_df.columns = [c.upper() for c in layer_task_df.columns]

    # Sort index to ensure proper order
    layer_task_df = layer_task_df.sort_index()

    return layer_task_df


def plot_layer_task_heatmap(
    layer_task_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create heatmap of layer-wise task vector magnitudes.

    Args:
        layer_task_df: Layer × task matrix
        output_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(14, 12))

    # Normalize each column (task) to [0, 1]
    normalized_data = layer_task_df.values.copy()
    for col_idx in range(normalized_data.shape[1]):
        col_data = normalized_data[:, col_idx]
        col_min = col_data.min()
        col_max = col_data.max()
        if col_max > col_min:
            normalized_data[:, col_idx] = (col_data - col_min) / (col_max - col_min)
        else:
            normalized_data[:, col_idx] = 0

    # Create heatmap
    im = ax.imshow(normalized_data, cmap='viridis', aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(len(layer_task_df.columns)))
    ax.set_yticks(np.arange(len(layer_task_df.index)))
    ax.set_xticklabels(layer_task_df.columns)

    # Create y-axis labels with special label for audio tower
    y_labels = []
    for layer_idx in layer_task_df.index:
        if layer_idx == -1:
            y_labels.append('Audio Tower')
        else:
            y_labels.append(str(layer_idx))
    ax.set_yticklabels(y_labels)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Add grid
    ax.set_xticks(np.arange(len(layer_task_df.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(layer_task_df.index)) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)

    # Add annotations with actual values
    for i in range(len(layer_task_df.index)):
        for j in range(len(layer_task_df.columns)):
            value = layer_task_df.iloc[i, j]
            text_color = 'white' if normalized_data[i, j] < 0.5 else 'black'
            ax.text(
                j, i, f'{value:.1f}',
                ha="center", va="center",
                color=text_color,
                fontsize=8
            )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('L2 Norm (Normalized per Task)', rotation=270, labelpad=20)

    # Title and labels
    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title('Layer-wise Task Vector Impact\n(Normalized per Task)',
                fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved heatmap to: {output_path}")
    plt.close()


def plot_layer_profiles(
    layer_task_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create line plot showing layer profile for each task.

    Args:
        layer_task_df: Layer × task matrix
        output_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Filter out audio tower layer
    layer_task_filtered = layer_task_df[layer_task_df.index >= 0]

    colors = sns.color_palette('tab10', n_colors=len(layer_task_filtered.columns))

    for i, task_name in enumerate(layer_task_filtered.columns):
        ax.plot(
            layer_task_filtered.index,
            layer_task_filtered[task_name],
            marker='o',
            linewidth=2,
            markersize=6,
            label=task_name,
            color=colors[i]
        )

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Total L2 Norm', fontsize=12)
    ax.set_title('Layer-wise Task Vector Magnitude Profiles',
                fontsize=16, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved line plot to: {output_path}")
    plt.close()


def main():
    print("=" * 100)
    print("LAYER-WISE TASK VECTOR IMPACT ANALYSIS")
    print("=" * 100)

    # Setup paths
    artifacts_dir = Path(__file__).parent.parent.parent / 'artifacts'
    output_dir = Path(__file__).parent.parent / 'results'
    plot_dir = Path(__file__).parent.parent / 'plots' / 'layer_wise_impact'

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Extract task vectors for all tasks
    task_vectors = {}

    for task_name, adapter_name in TASK_ADAPTERS.items():
        print(f"\n{'=' * 100}")
        print(f"Extracting task vector for {task_name.upper()}...")
        print(f"{'=' * 100}")

        # Use the 'best' symlink
        adapter_path = artifacts_dir / task_name / 'adapters' / adapter_name / 'best'

        try:
            # Extract task vector
            task_vector = extract_task_vector_from_lora(adapter_path)
            task_vectors[task_name] = task_vector

        except FileNotFoundError as e:
            print(f"✗ Adapter not found for {task_name}: {e}")
            continue
        except Exception as e:
            print(f"✗ Error processing {task_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not task_vectors:
        raise RuntimeError("No task vectors were successfully extracted. Exiting.")

    # Compute layer-wise statistics
    print(f"\n{'=' * 100}")
    print("COMPUTING LAYER-WISE STATISTICS")
    print(f"{'=' * 100}")

    stats_df = compute_layer_wise_statistics(task_vectors)

    # Check for unparseable parameters
    total_params = sum(len(tv) for tv in task_vectors.values())
    parsed_params = len(stats_df)
    if parsed_params < total_params:
        print(f"⚠️  Warning: Parsed {parsed_params}/{total_params} parameters")

    # Aggregate to layer × task matrix
    layer_task_df = aggregate_layer_task_matrix(stats_df)

    # Print summary
    print("\n" + "=" * 100)
    print("LAYER × TASK MATRIX (L2 Norms)")
    print("=" * 100)
    print(layer_task_df.to_string())

    # Save results
    print(f"\n{'=' * 100}")
    print("SAVING RESULTS")
    print(f"{'=' * 100}")

    # Save detailed statistics
    detailed_path = output_dir / 'layer_wise_statistics_detailed.csv'
    stats_df.to_csv(detailed_path, index=False)
    print(f"✓ Saved detailed statistics to: {detailed_path}")

    # Save aggregated matrix
    matrix_path = output_dir / 'layer_task_matrix.csv'
    layer_task_df.to_csv(matrix_path)
    print(f"✓ Saved layer × task matrix to: {matrix_path}")

    # Generate plots
    print(f"\n{'=' * 100}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'=' * 100}")

    heatmap_path = plot_dir / 'layer_task_heatmap.png'
    plot_layer_task_heatmap(layer_task_df, heatmap_path)

    profile_path = plot_dir / 'layer_profiles_line.png'
    plot_layer_profiles(layer_task_df, profile_path)

    print(f"\n{'=' * 100}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 100}")
    print(f"Results saved to: {output_dir}")
    print(f"Plots saved to: {plot_dir}")


if __name__ == '__main__':
    main()
