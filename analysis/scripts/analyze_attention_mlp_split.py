#!/usr/bin/env python3
"""
Analyze task vector component split: Attention vs MLP vs Audio Tower.

This script:
1. Extracts task vectors for each adapter
2. Classifies parameters as attention, MLP, or audio_tower
3. Computes aggregate L2 norms for each component type
4. Creates visualizations showing component specialization
5. Helps explain architectural patterns in task adaptation
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

# Module classification
ATTENTION_MODULES = {'q_proj', 'k_proj', 'v_proj', 'o_proj'}
MLP_MODULES = {'gate_proj', 'up_proj', 'down_proj'}
AUDIO_MODULES = {'audio_tower.proj'}


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


def classify_module_component(module_name: str) -> str:
    """Classify module as attention, MLP, or audio_tower.

    Args:
        module_name: Module name (e.g., 'q_proj', 'gate_proj')

    Returns:
        Component type: 'attention', 'mlp', or 'audio_tower'
    """
    # Attention modules
    if any(proj in module_name for proj in ATTENTION_MODULES):
        return 'attention'

    # MLP modules
    elif any(proj in module_name for proj in MLP_MODULES):
        return 'mlp'

    # Audio tower
    elif 'audio_tower' in module_name:
        return 'audio_tower'

    else:
        return 'unknown'


def compute_component_statistics(
    task_vectors: Dict[str, Dict[str, torch.Tensor]]
) -> pd.DataFrame:
    """Compute statistics for each component type (attention/MLP) per task.

    Args:
        task_vectors: Dict mapping task names to their task vectors

    Returns:
        DataFrame with columns: task, component, total_l2_norm, num_parameters,
                               mean_l2_norm, total_elements
    """
    stats = []

    for task_name, task_vector in task_vectors.items():
        # Group by component
        component_groups = {
            'attention': [],
            'mlp': [],
            'audio_tower': []
        }

        for param_name, delta_W in task_vector.items():
            # Extract module name
            module_name = param_name  # Use full name for classification

            component = classify_module_component(module_name)
            if component == 'unknown':
                continue

            # Convert to numpy
            if delta_W.dtype == torch.bfloat16:
                delta_W_np = delta_W.cpu().float().numpy()
            else:
                delta_W_np = delta_W.cpu().numpy()

            # Compute L2 norm
            l2_norm = np.linalg.norm(delta_W_np)

            component_groups[component].append({
                'l2_norm': l2_norm,
                'num_elements': delta_W_np.size
            })

        # Aggregate statistics per component
        for component, params_list in component_groups.items():
            if not params_list:
                continue

            total_l2 = sum(p['l2_norm'] for p in params_list)
            num_params = len(params_list)
            mean_l2 = total_l2 / num_params if num_params > 0 else 0
            total_elements = sum(p['num_elements'] for p in params_list)

            stats.append({
                'task': task_name,
                'component': component,
                'total_l2_norm': total_l2,
                'num_parameters': num_params,
                'mean_l2_norm': mean_l2,
                'total_elements': total_elements
            })

    return pd.DataFrame(stats)


def compute_attention_mlp_ratios(comp_stats_df: pd.DataFrame) -> pd.DataFrame:
    """Compute ratio of attention to MLP magnitude for each task.

    Args:
        comp_stats_df: DataFrame from compute_component_statistics()

    Returns:
        DataFrame with columns: task, attention_l2, mlp_l2, audio_tower_l2,
                               attention_mlp_ratio, percentages
    """
    ratios = []

    for task_name in comp_stats_df['task'].unique():
        task_data = comp_stats_df[comp_stats_df['task'] == task_name]

        # Get component values
        attn_row = task_data[task_data['component'] == 'attention']
        mlp_row = task_data[task_data['component'] == 'mlp']
        audio_row = task_data[task_data['component'] == 'audio_tower']

        attn_l2 = attn_row['total_l2_norm'].values[0] if len(attn_row) > 0 else 0
        mlp_l2 = mlp_row['total_l2_norm'].values[0] if len(mlp_row) > 0 else 0
        audio_l2 = audio_row['total_l2_norm'].values[0] if len(audio_row) > 0 else 0

        total = attn_l2 + mlp_l2 + audio_l2

        # Compute ratio and percentages
        ratio = attn_l2 / mlp_l2 if mlp_l2 > 0 else np.inf
        attn_pct = (attn_l2 / total * 100) if total > 0 else 0
        mlp_pct = (mlp_l2 / total * 100) if total > 0 else 0
        audio_pct = (audio_l2 / total * 100) if total > 0 else 0

        ratios.append({
            'task': task_name,
            'attention_l2': attn_l2,
            'mlp_l2': mlp_l2,
            'audio_tower_l2': audio_l2,
            'attention_mlp_ratio': ratio,
            'attention_percentage': attn_pct,
            'mlp_percentage': mlp_pct,
            'audio_tower_percentage': audio_pct
        })

    return pd.DataFrame(ratios)


def plot_component_split_bar(
    ratios_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create stacked bar chart showing attention/MLP/audio split.

    Args:
        ratios_df: DataFrame from compute_attention_mlp_ratios()
        output_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    tasks = [t.upper() for t in ratios_df['task']]
    attn_pct = ratios_df['attention_percentage'].values
    mlp_pct = ratios_df['mlp_percentage'].values
    audio_pct = ratios_df['audio_tower_percentage'].values

    # Create stacked horizontal bars
    y_pos = np.arange(len(tasks))

    p1 = ax.barh(y_pos, attn_pct, color='#3498db', edgecolor='black', linewidth=1, label='Attention')
    p2 = ax.barh(y_pos, mlp_pct, left=attn_pct, color='#e74c3c', edgecolor='black', linewidth=1, label='MLP')
    p3 = ax.barh(y_pos, audio_pct, left=attn_pct + mlp_pct, color='#95a5a6', edgecolor='black', linewidth=1, label='Audio Tower')

    # Add percentage annotations
    for i, (task, attn, mlp, audio) in enumerate(zip(tasks, attn_pct, mlp_pct, audio_pct)):
        # Attention annotation
        if attn > 5:  # Only show if segment is large enough
            ax.text(attn / 2, i, f'{attn:.1f}%', ha='center', va='center',
                   color='white', fontweight='bold', fontsize=10)

        # MLP annotation
        if mlp > 5:
            ax.text(attn + mlp / 2, i, f'{mlp:.1f}%', ha='center', va='center',
                   color='white', fontweight='bold', fontsize=10)

        # Audio annotation
        if audio > 5:
            ax.text(attn + mlp + audio / 2, i, f'{audio:.1f}%', ha='center', va='center',
                   color='white', fontweight='bold', fontsize=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(tasks)
    ax.set_xlabel('Percentage (%)', fontsize=12)
    ax.set_title('Task Vector Component Split:\nAttention vs MLP vs Audio Tower',
                fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, shadow=True)
    ax.set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved stacked bar chart to: {output_path}")
    plt.close()


def plot_attention_mlp_ratio_bar(
    ratios_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create bar chart of attention/MLP ratios.

    Args:
        ratios_df: DataFrame from compute_attention_mlp_ratios()
        output_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    tasks = [t.upper() for t in ratios_df['task']]
    ratios = ratios_df['attention_mlp_ratio'].values

    # Color: green if > 1, orange if < 1
    colors = ['#27ae60' if r > 1 else '#e67e22' for r in ratios]

    bars = ax.bar(tasks, ratios, color=colors, edgecolor='black', linewidth=1.2)

    # Add horizontal line at y=1
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Equal (Ratio=1)')

    # Add value annotations
    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{ratio:.2f}',
            ha='center',
            va='bottom' if ratio > 1 else 'top',
            fontsize=11,
            fontweight='bold'
        )

    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Attention/MLP Ratio', fontsize=12)
    ax.set_title('Attention to MLP Ratio by Task', fontsize=16, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved ratio bar chart to: {output_path}")
    plt.close()


def plot_component_absolute_comparison(
    comp_stats_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create grouped bar chart comparing absolute L2 norms.

    Args:
        comp_stats_df: DataFrame from compute_component_statistics()
        output_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    tasks = sorted(comp_stats_df['task'].unique())
    x = np.arange(len(tasks))
    width = 0.25

    # Extract data
    attn_norms = []
    mlp_norms = []
    audio_norms = []

    for task in tasks:
        task_data = comp_stats_df[comp_stats_df['task'] == task]

        attn_row = task_data[task_data['component'] == 'attention']
        mlp_row = task_data[task_data['component'] == 'mlp']
        audio_row = task_data[task_data['component'] == 'audio_tower']

        attn_norms.append(attn_row['total_l2_norm'].values[0] if len(attn_row) > 0 else 0)
        mlp_norms.append(mlp_row['total_l2_norm'].values[0] if len(mlp_row) > 0 else 0)
        audio_norms.append(audio_row['total_l2_norm'].values[0] if len(audio_row) > 0 else 0)

    # Create bars
    ax.bar(x - width, attn_norms, width, label='Attention', color='#3498db', edgecolor='black')
    ax.bar(x, mlp_norms, width, label='MLP', color='#e74c3c', edgecolor='black')
    ax.bar(x + width, audio_norms, width, label='Audio Tower', color='#95a5a6', edgecolor='black')

    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Total L2 Norm', fontsize=12)
    ax.set_title('Component-wise L2 Norms by Task', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in tasks], rotation=45, ha='right')
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved grouped bar chart to: {output_path}")
    plt.close()


def main():
    print("=" * 100)
    print("ATTENTION vs MLP COMPONENT SPLIT ANALYSIS")
    print("=" * 100)

    # Setup paths
    artifacts_dir = Path(__file__).parent.parent.parent / 'artifacts'
    output_dir = Path(__file__).parent.parent / 'results'
    plot_dir = Path(__file__).parent.parent / 'plots' / 'attention_mlp_split'

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

    # Compute component statistics
    print(f"\n{'=' * 100}")
    print("COMPUTING COMPONENT STATISTICS")
    print(f"{'=' * 100}")

    comp_stats_df = compute_component_statistics(task_vectors)

    # Check for missing components
    for task_name in task_vectors.keys():
        task_data = comp_stats_df[comp_stats_df['task'] == task_name]
        components_present = set(task_data['component'])

        expected_components = {'attention', 'mlp', 'audio_tower'}
        missing_components = expected_components - components_present

        if missing_components:
            print(f"⚠️  Warning: Task '{task_name}' missing components: {missing_components}")

    # Compute ratios
    ratios_df = compute_attention_mlp_ratios(comp_stats_df)

    # Print summary tables
    print("\n" + "=" * 100)
    print("COMPONENT STATISTICS")
    print("=" * 100)
    print(comp_stats_df.to_string(index=False))

    print("\n" + "=" * 100)
    print("ATTENTION/MLP RATIOS")
    print("=" * 100)
    print(ratios_df.to_string(index=False))

    # Save results
    print(f"\n{'=' * 100}")
    print("SAVING RESULTS")
    print(f"{'=' * 100}")

    comp_path = output_dir / 'component_statistics.csv'
    comp_stats_df.to_csv(comp_path, index=False)
    print(f"✓ Saved component statistics to: {comp_path}")

    ratios_path = output_dir / 'attention_mlp_ratios.csv'
    ratios_df.to_csv(ratios_path, index=False)
    print(f"✓ Saved ratios to: {ratios_path}")

    # Generate plots
    print(f"\n{'=' * 100}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'=' * 100}")

    stacked_path = plot_dir / 'component_split_stacked_bar.png'
    plot_component_split_bar(ratios_df, stacked_path)

    ratio_path = plot_dir / 'attention_mlp_ratio_bar.png'
    plot_attention_mlp_ratio_bar(ratios_df, ratio_path)

    grouped_path = plot_dir / 'component_absolute_comparison.png'
    plot_component_absolute_comparison(comp_stats_df, grouped_path)

    print(f"\n{'=' * 100}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 100}")
    print(f"Results saved to: {output_dir}")
    print(f"Plots saved to: {plot_dir}")


if __name__ == '__main__':
    main()
