#!/usr/bin/env python3
"""
Analyze global L2 norms of task vectors across different tasks.

This script:
1. Extracts task vectors for each adapter: Δθ = (B @ A) × (α/r)
2. Computes global L2 norm: ‖Δθ‖ for each task
3. Visualizes task vector magnitudes with a bar chart
4. Helps explain why some tasks dominate in naive merging
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


def compute_global_l2_norms(
    task_vectors: Dict[str, Dict[str, torch.Tensor]]
) -> pd.DataFrame:
    """Compute global L2 norm for each task vector.

    Formula: ||theta_task||_2 = sqrt(sum_{i} ||delta_W_i||_2^2)
    Where i iterates over all parameters in the task vector.

    Args:
        task_vectors: Dict mapping task names to their task vectors

    Returns:
        DataFrame with columns: task, global_l2_norm, num_parameters,
                               total_elements, mean_element_magnitude, max_element_magnitude
    """
    stats = []

    for task_name, task_vector in task_vectors.items():
        squared_norms = []
        total_elements = 0
        all_values = []

        for param_name, delta_W in task_vector.items():
            # Handle BFloat16
            if delta_W.dtype == torch.bfloat16:
                delta_W_np = delta_W.cpu().float().numpy()
            else:
                delta_W_np = delta_W.cpu().numpy()

            # Accumulate squared norm
            param_l2_squared = np.sum(delta_W_np ** 2)
            squared_norms.append(param_l2_squared)

            # Track statistics
            total_elements += delta_W_np.size
            all_values.append(delta_W_np.flatten())

        # Global L2 norm
        global_l2_norm = np.sqrt(np.sum(squared_norms))

        # Element statistics
        all_values = np.concatenate(all_values)
        mean_element_magnitude = np.mean(np.abs(all_values))
        max_element_magnitude = np.max(np.abs(all_values))

        stats.append({
            'task': task_name,
            'global_l2_norm': global_l2_norm,
            'num_parameters': len(task_vector),
            'total_elements': total_elements,
            'mean_element_magnitude': mean_element_magnitude,
            'max_element_magnitude': max_element_magnitude
        })

    return pd.DataFrame(stats)


def plot_task_vector_norms_bar(
    norms_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create bar chart of task vector norms.

    Args:
        norms_df: DataFrame from compute_global_l2_norms()
        output_path: Where to save the plot (PNG file)
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Sort by norm for better visualization
    norms_df_sorted = norms_df.sort_values('global_l2_norm', ascending=False)

    colors = sns.color_palette('Set2', n_colors=len(norms_df_sorted))
    tasks_upper = [t.upper() for t in norms_df_sorted['task']]
    norms = norms_df_sorted['global_l2_norm'].values

    bars = ax.bar(
        tasks_upper,
        norms,
        color=colors,
        edgecolor='black',
        linewidth=1.2
    )

    # Add value annotations
    for bar, norm in zip(bars, norms):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{norm:.2f}',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )

    ax.set_xlabel('Task', fontsize=14)
    ax.set_ylabel('Global L2 Norm', fontsize=14)
    ax.set_title('Task Vector Global L2 Norms', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_path}")
    plt.close()


def save_norms_table(
    norms_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save norms table to CSV.

    Args:
        norms_df: DataFrame from compute_global_l2_norms()
        output_path: CSV file path
    """
    norms_df.to_csv(output_path, index=False)
    print(f"✓ Saved results to: {output_path}")


def main():
    print("=" * 100)
    print("TASK VECTOR GLOBAL L2 NORM ANALYSIS")
    print("=" * 100)

    # Setup paths
    artifacts_dir = Path(__file__).parent.parent.parent / 'artifacts'
    output_dir = Path(__file__).parent.parent / 'results'
    plot_dir = Path(__file__).parent.parent / 'plots' / 'task_vector_norms'

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

    # Compute global L2 norms
    print(f"\n{'=' * 100}")
    print("COMPUTING GLOBAL L2 NORMS")
    print(f"{'=' * 100}")

    norms_df = compute_global_l2_norms(task_vectors)

    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY: TASK VECTOR NORMS")
    print("=" * 100)
    norms_df_sorted = norms_df.sort_values('global_l2_norm', ascending=False)
    print(norms_df_sorted.to_string(index=False))

    # Save results
    print(f"\n{'=' * 100}")
    print("SAVING RESULTS")
    print(f"{'=' * 100}")

    csv_path = output_dir / 'task_vector_norms.csv'
    save_norms_table(norms_df, csv_path)

    # Generate plot
    plot_path = plot_dir / 'global_l2_norms_bar.png'
    plot_task_vector_norms_bar(norms_df, plot_path)

    print(f"\n{'=' * 100}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 100}")
    print(f"Results saved to: {output_dir}")
    print(f"Plots saved to: {plot_dir}")


if __name__ == '__main__':
    main()
