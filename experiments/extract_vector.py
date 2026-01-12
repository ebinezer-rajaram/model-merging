"""Extract task vectors from LoRA adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from safetensors import safe_open
from safetensors.torch import save_file


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

    print(f"📊 Adapter config: r={lora_r}, alpha={lora_alpha}, scaling={scaling:.4f}")

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
            print(f"⚠️  Warning: Incomplete LoRA pair for {base_key}")

    print(f"✅ Extracted {len(task_vectors)} task vectors")

    return task_vectors


def save_task_vector(
    task_vector: Dict[str, torch.Tensor],
    output_path: Path,
    adapter_path: Path,
) -> None:
    """Save task vector to disk.

    Args:
        task_vector: Dictionary of task vectors
        output_path: Path to save task vector
        adapter_path: Original adapter path (for metadata)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as SafeTensors
    save_file(task_vector, str(output_path))

    # Save metadata
    metadata_path = output_path.with_suffix(".json")
    metadata = {
        "source_adapter": str(adapter_path),
        "num_parameters": len(task_vector),
        "parameter_names": list(task_vector.keys()),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"💾 Saved task vector to {output_path}")
    print(f"   Metadata: {metadata_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract task vectors from LoRA adapters"
    )

    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Path to adapter directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for task vector (.safetensors)",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    adapter_path = Path(args.adapter).expanduser()
    if not adapter_path.is_absolute():
        package_root = Path(__file__).resolve().parent.parent
        adapter_path = package_root / adapter_path

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        package_root = Path(__file__).resolve().parent.parent
        output_path = package_root / output_path

    print(f"🔍 Extracting task vector from: {adapter_path}")

    task_vector = extract_task_vector_from_lora(adapter_path)
    save_task_vector(task_vector, output_path, adapter_path)

    print("✅ Done!")


if __name__ == "__main__":
    main()
