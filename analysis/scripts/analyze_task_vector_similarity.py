#!/usr/bin/env python3
"""
Analyze pairwise cosine similarity between task vectors.

This script:
1. Extracts task vectors for each adapter
2. Flattens all parameters into a single vector
3. Computes pairwise cosine similarity between all task pairs
4. Visualizes the similarity matrix with a heatmap
5. Helps explain which tasks have aligned vs conflicting updates
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


def flatten_task_vector(task_vector: Dict[str, torch.Tensor]) -> np.ndarray:
    """Flatten all parameters into a single 1D numpy array.

    Args:
        task_vector: Dict mapping parameter names to tensors

    Returns:
        1D numpy array containing all parameter values concatenated
    """
    flattened_parts = []

    # Sort keys for consistent ordering
    for param_name in sorted(task_vector.keys()):
        delta_W = task_vector[param_name]

        # Handle BFloat16
        if delta_W.dtype == torch.bfloat16:
            delta_W_np = delta_W.cpu().float().numpy()
        else:
            delta_W_np = delta_W.cpu().numpy()

        # Flatten and append
        flattened_parts.append(delta_W_np.flatten())

    # Concatenate all parts
    return np.concatenate(flattened_parts)


def compute_cosine_similarity_matrix(
    task_vectors: Dict[str, Dict[str, torch.Tensor]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise cosine similarity between all task vectors.

    Formula: cos_sim(theta_i, theta_j) = (theta_i · theta_j) / (||theta_i|| * ||theta_j||)

    Args:
        task_vectors: Dict mapping task names to their task vectors

    Returns:
        Tuple of (similarity_df, metadata_df)
    """
    # Flatten all task vectors
    task_names = sorted(task_vectors.keys())
    flattened_vectors = {}
    norms = {}

    print("\nFlattening task vectors...")
    for task_name in task_names:
        flat = flatten_task_vector(task_vectors[task_name])
        flattened_vectors[task_name] = flat
        norms[task_name] = np.linalg.norm(flat)
        print(f"  {task_name}: {len(flat):,} elements, norm={norms[task_name]:.2f}")

    # Check dimension consistency
    dimensions = [len(vec) for vec in flattened_vectors.values()]
    if len(set(dimensions)) > 1:
        raise ValueError(
            f"Task vectors have inconsistent dimensions: {set(dimensions)}. "
            "This should not happen if all adapters target the same base model."
        )

    # Check for zero norms
    zero_norm_tasks = [t for t, n in norms.items() if n == 0]
    if zero_norm_tasks:
        raise ValueError(f"Tasks have zero norm: {zero_norm_tasks}")

    # Compute pairwise similarities
    print("\nComputing pairwise cosine similarities...")
    n_tasks = len(task_names)
    similarity_matrix = np.zeros((n_tasks, n_tasks))

    for i, task_i in enumerate(task_names):
        vec_i = flattened_vectors[task_i]
        norm_i = norms[task_i]

        for j, task_j in enumerate(task_names):
            vec_j = flattened_vectors[task_j]
            norm_j = norms[task_j]

            # Cosine similarity
            dot_product = np.dot(vec_i, vec_j)
            similarity = dot_product / (norm_i * norm_j)

            similarity_matrix[i, j] = similarity

    # Create DataFrames
    task_names_upper = [t.upper() for t in task_names]
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=task_names_upper,
        columns=task_names_upper
    )

    metadata_df = pd.DataFrame({
        'task': task_names_upper,
        'norm': [norms[t] for t in task_names],
        'dimensions': [len(flattened_vectors[t]) for t in task_names]
    })

    return similarity_df, metadata_df


def plot_similarity_heatmap(
    similarity_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create heatmap of pairwise cosine similarities.

    Args:
        similarity_df: Square matrix of cosine similarities
        output_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap using imshow for precise control
    im = ax.imshow(
        similarity_df.values,
        cmap='RdYlGn',
        aspect='equal',
        vmin=-1,  # Allow for negative similarities (interference)
        vmax=1
    )

    # Set ticks
    ax.set_xticks(np.arange(len(similarity_df.columns)))
    ax.set_yticks(np.arange(len(similarity_df.index)))
    ax.set_xticklabels(similarity_df.columns)
    ax.set_yticklabels(similarity_df.index)

    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add grid
    ax.set_xticks(np.arange(len(similarity_df.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(similarity_df.index)) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=1)

    # Add annotations
    for i in range(len(similarity_df.index)):
        for j in range(len(similarity_df.columns)):
            value = similarity_df.iloc[i, j]

            # Bold text for diagonal
            fontweight = 'bold' if i == j else 'normal'

            # Text color based on actual colormap
            # RdYlGn goes: red (low) -> yellow (mid) -> green (high)
            # Dark text for extreme values, light text for middle range
            if value < -0.3 or value > 0.7:
                # Dark colors (red or green) - use white text
                text_color = 'white'
            else:
                # Light/yellow colors in the middle - use black text
                text_color = 'black'

            ax.text(
                j, i, f'{value:.3f}',
                ha="center", va="center",
                color=text_color,
                fontsize=10,
                fontweight=fontweight
            )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', rotation=270, labelpad=20, fontsize=12)

    # Labels and title
    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Task', fontsize=12)
    ax.set_title('Task Vector Pairwise Cosine Similarity', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_path}")
    plt.close()


def save_similarity_results(
    similarity_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    results_dir: Path,
) -> None:
    """Save similarity matrix and metadata to CSV files.

    Args:
        similarity_df: Cosine similarity matrix
        metadata_df: Task vector metadata
        results_dir: Directory to save results
    """
    # Save similarity matrix
    sim_path = results_dir / 'task_vector_similarity_matrix.csv'
    similarity_df.to_csv(sim_path)
    print(f"✓ Saved similarity matrix to: {sim_path}")

    # Save metadata
    meta_path = results_dir / 'task_vector_similarity_metadata.csv'
    metadata_df.to_csv(meta_path, index=False)
    print(f"✓ Saved metadata to: {meta_path}")


def main():
    print("=" * 100)
    print("TASK VECTOR COSINE SIMILARITY ANALYSIS")
    print("=" * 100)

    # Setup paths
    artifacts_dir = Path(__file__).parent.parent.parent / 'artifacts'
    output_dir = Path(__file__).parent.parent / 'results'
    plot_dir = Path(__file__).parent.parent / 'plots' / 'task_vector_similarity'

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

    # Compute cosine similarity matrix
    print(f"\n{'=' * 100}")
    print("COMPUTING COSINE SIMILARITY MATRIX")
    print(f"{'=' * 100}")

    similarity_df, metadata_df = compute_cosine_similarity_matrix(task_vectors)

    # Print similarity matrix
    print("\n" + "=" * 100)
    print("PAIRWISE COSINE SIMILARITY MATRIX")
    print("=" * 100)
    print(similarity_df.to_string())

    # Print metadata
    print("\n" + "=" * 100)
    print("TASK VECTOR METADATA")
    print("=" * 100)
    print(metadata_df.to_string(index=False))

    # Identify most similar and most dissimilar pairs
    print("\n" + "=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100)

    # Extract upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(similarity_df.values, dtype=bool), k=1)
    similarities_upper = similarity_df.values[mask]

    # Get task pairs
    task_pairs = []
    for i in range(len(similarity_df.index)):
        for j in range(i+1, len(similarity_df.columns)):
            task_pairs.append((similarity_df.index[i], similarity_df.columns[j], similarity_df.iloc[i, j]))

    # Sort by similarity
    task_pairs_sorted = sorted(task_pairs, key=lambda x: x[2], reverse=True)

    print("\nMost similar task pairs (highest cosine similarity):")
    for i, (task1, task2, sim) in enumerate(task_pairs_sorted[:3], 1):
        print(f"  {i}. {task1} <-> {task2}: {sim:.4f}")

    print("\nMost dissimilar task pairs (lowest cosine similarity):")
    for i, (task1, task2, sim) in enumerate(task_pairs_sorted[-3:], 1):
        print(f"  {i}. {task1} <-> {task2}: {sim:.4f}")

    # Save results
    print(f"\n{'=' * 100}")
    print("SAVING RESULTS")
    print(f"{'=' * 100}")

    save_similarity_results(similarity_df, metadata_df, output_dir)

    # Generate plot
    plot_path = plot_dir / 'cosine_similarity_heatmap.png'
    plot_similarity_heatmap(similarity_df, plot_path)

    print(f"\n{'=' * 100}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 100}")
    print(f"Results saved to: {output_dir}")
    print(f"Plots saved to: {plot_dir}")


if __name__ == '__main__':
    main()
