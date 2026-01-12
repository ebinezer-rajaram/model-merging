"""Merge LoRA adapter vectors using various strategies."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from experiments.extract_vector import extract_task_vector_from_lora


def load_adapter_weights(adapter_path: Path) -> Dict[str, torch.Tensor]:
    """Load LoRA adapter weights from SafeTensors file.

    Args:
        adapter_path: Path to adapter directory containing adapter_model.safetensors

    Returns:
        Dictionary mapping parameter names to tensors
    """
    safetensors_path = adapter_path / "adapter_model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"Adapter weights not found at {safetensors_path}")

    weights = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key).clone()

    return weights


def merge_adapters_uniform(
    adapter_weights: List[Dict[str, torch.Tensor]],
    merge_mode: str = "common",
) -> Dict[str, torch.Tensor]:
    """Merge multiple adapters using uniform averaging.

    Computes simple average: W_merged = (W_1 + W_2 + ... + W_n) / n

    Args:
        adapter_weights: List of adapter weight dictionaries
        merge_mode: How to handle different parameters across adapters:
            - "common": Only merge parameters present in ALL adapters (default)
            - "strict": Require all adapters to have identical parameters

    Returns:
        Merged adapter weights
    """
    if not adapter_weights:
        raise ValueError("No adapters provided for merging")

    if len(adapter_weights) == 1:
        return adapter_weights[0]

    # Find common and unique parameters
    all_keys = [set(adapter.keys()) for adapter in adapter_weights]
    common_keys = set.intersection(*all_keys)
    all_unique_keys = set.union(*all_keys)

    if merge_mode == "strict":
        # Verify all adapters have the same keys
        reference_keys = all_keys[0]
        for i, adapter_keys in enumerate(all_keys[1:], start=1):
            if adapter_keys != reference_keys:
                missing = reference_keys - adapter_keys
                extra = adapter_keys - reference_keys
                raise ValueError(
                    f"Adapter {i} has different parameters than adapter 0.\n"
                    f"Missing: {missing}\nExtra: {extra}"
                )
        keys_to_merge = reference_keys
    else:  # common mode
        keys_to_merge = common_keys
        unique_keys = all_unique_keys - common_keys

        if unique_keys:
            print(f"⚠️  Warning: {len(unique_keys)} parameters are not common across all adapters")
            print(f"   Only merging {len(common_keys)} common parameters")
            print(f"   Unique parameters will be excluded from the merge")

    # Average all common parameters
    merged_weights = {}
    num_adapters = len(adapter_weights)

    for key in keys_to_merge:
        # Sum all adapter weights for this parameter
        summed = torch.zeros_like(adapter_weights[0][key])
        for adapter in adapter_weights:
            summed += adapter[key]

        # Compute average
        merged_weights[key] = summed / num_adapters

    return merged_weights


def create_merged_adapter_config(
    reference_adapter_path: Path,
    merged_weights: Dict[str, torch.Tensor],
) -> Dict:
    """Create adapter config for merged adapter.

    Args:
        reference_adapter_path: Path to reference adapter
        merged_weights: Merged adapter weights

    Returns:
        Updated adapter config
    """
    config_path = reference_adapter_path / "adapter_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # Extract unique module names from merged weights
    # e.g., "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight" -> "q_proj"
    merged_modules = set()
    for key in merged_weights.keys():
        parts = key.split(".")
        # Find the module name before lora_A or lora_B
        for i, part in enumerate(parts):
            if part in ["lora_A", "lora_B"]:
                # Module name could be one or more parts before lora_A/lora_B
                # For audio_tower.proj, we need to include both parts
                if i >= 2 and parts[i-2] == "audio_tower":
                    merged_modules.add(f"{parts[i-2]}.{parts[i-1]}")
                else:
                    merged_modules.add(parts[i-1])
                break

    # Update target_modules to only include merged modules
    config["target_modules"] = sorted(list(merged_modules))

    return config


def save_merged_adapter(
    weights: Dict[str, torch.Tensor],
    output_path: Path,
    reference_adapter_path: Path,
) -> None:
    """Save merged adapter in PEFT format.

    Args:
        weights: Merged adapter weights
        output_path: Directory to save merged adapter
        reference_adapter_path: Path to reference adapter for copying config files
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Save weights as SafeTensors
    safetensors_path = output_path / "adapter_model.safetensors"
    save_file(weights, str(safetensors_path))

    # Create adapter config based on merged weights
    config = create_merged_adapter_config(reference_adapter_path, weights)
    config_dst = output_path / "adapter_config.json"
    with open(config_dst, "w") as f:
        json.dump(config, f, indent=2)

    print(f"✅ Saved merged adapter to {output_path}")
    print(f"   - Weights: {safetensors_path}")
    print(f"   - Config: {config_dst}")
    print(f"   - Target modules: {', '.join(config['target_modules'])}")


def merge_task_vectors_uniform(
    task_vectors: List[Dict[str, torch.Tensor]],
    merge_mode: str = "common",
) -> Dict[str, torch.Tensor]:
    """Merge multiple task vectors using uniform averaging.

    Args:
        task_vectors: List of task vector dictionaries
        merge_mode: How to handle different parameters ("common" or "strict")

    Returns:
        Merged task vector
    """
    if not task_vectors:
        raise ValueError("No task vectors provided for merging")

    if len(task_vectors) == 1:
        return task_vectors[0]

    # Find common and unique parameters
    all_keys = [set(tv.keys()) for tv in task_vectors]
    common_keys = set.intersection(*all_keys)
    all_unique_keys = set.union(*all_keys)

    if merge_mode == "strict":
        reference_keys = all_keys[0]
        for i, tv_keys in enumerate(all_keys[1:], start=1):
            if tv_keys != reference_keys:
                missing = reference_keys - tv_keys
                extra = tv_keys - reference_keys
                raise ValueError(
                    f"Task vector {i} has different parameters than task vector 0.\n"
                    f"Missing: {missing}\nExtra: {extra}"
                )
        keys_to_merge = reference_keys
    else:  # common mode
        keys_to_merge = common_keys
        unique_keys = all_unique_keys - common_keys

        if unique_keys:
            print(f"⚠️  Warning: {len(unique_keys)} parameters are not common across all task vectors")
            print(f"   Only merging {len(common_keys)} common parameters")

    # Average all common parameters
    merged_tv = {}
    num_vectors = len(task_vectors)

    for key in keys_to_merge:
        # Check shapes match
        shapes = [tv[key].shape for tv in task_vectors]
        if len(set(shapes)) > 1:
            print(f"⚠️  Skipping {key}: inconsistent shapes {shapes}")
            continue

        # Sum and average
        summed = torch.zeros_like(task_vectors[0][key])
        for tv in task_vectors:
            summed += tv[key]

        merged_tv[key] = summed / num_vectors

    return merged_tv


def save_task_vector(
    task_vector: Dict[str, torch.Tensor],
    output_path: Path,
    metadata: Dict = None,
) -> None:
    """Save task vector to disk.

    Args:
        task_vector: Dictionary of task vectors
        output_path: Path to save task vector
        metadata: Optional metadata to save
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as SafeTensors
    save_file(task_vector, str(output_path))

    # Save metadata
    if metadata:
        metadata_path = output_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"💾 Saved merged task vector to {output_path}")
    print(f"   Parameters: {len(task_vector)}")


def merge_uniform_via_task_vectors(
    adapter_paths: List[Path],
    output_path: Path,
    merge_mode: str = "common",
    show_progress: bool = True,
) -> None:
    """Merge adapters by extracting and merging task vectors.

    This approach works even when adapters have different LoRA ranks,
    as we merge in the full parameter space rather than the low-rank space.

    Args:
        adapter_paths: List of paths to adapter directories
        output_path: Directory to save merged task vector
        merge_mode: How to handle different parameters ("common" or "strict")
        show_progress: Whether to print progress messages
    """
    if show_progress:
        print(f"🔀 Merging {len(adapter_paths)} adapters via task vectors")
        print(f"   Mode: {merge_mode}")
        for i, path in enumerate(adapter_paths, 1):
            print(f"   {i}. {path}")

    # Extract task vectors from each adapter
    task_vectors = []
    for i, adapter_path in enumerate(adapter_paths, 1):
        if show_progress:
            print(f"\n📥 Extracting task vector {i}/{len(adapter_paths)}: {adapter_path.name}")
        tv = extract_task_vector_from_lora(adapter_path)
        task_vectors.append(tv)

    # Merge task vectors
    if show_progress:
        print(f"\n🧮 Computing uniform average of task vectors...")
    merged_tv = merge_task_vectors_uniform(task_vectors, merge_mode=merge_mode)

    # Save merged task vector
    metadata = {
        "source_adapters": [str(p) for p in adapter_paths],
        "merge_method": "uniform_average",
        "merge_mode": merge_mode,
        "num_adapters": len(adapter_paths),
    }

    save_task_vector(merged_tv, output_path, metadata)

    if show_progress:
        print(f"\n✅ Merge complete!")


def merge_uniform(
    adapter_paths: List[Path],
    output_path: Path,
    merge_mode: str = "common",
    show_progress: bool = True,
) -> None:
    """Perform uniform averaging merge of multiple adapters.

    Args:
        adapter_paths: List of paths to adapter directories
        output_path: Directory to save merged adapter
        merge_mode: How to handle different parameters ("common" or "strict")
        show_progress: Whether to print progress messages
    """
    if show_progress:
        print(f"🔀 Merging {len(adapter_paths)} adapters using uniform averaging")
        print(f"   Mode: {merge_mode}")
        for i, path in enumerate(adapter_paths, 1):
            print(f"   {i}. {path}")

    # Load all adapter weights
    all_weights = []
    for i, adapter_path in enumerate(adapter_paths, 1):
        if show_progress:
            print(f"📥 Loading adapter {i}/{len(adapter_paths)}: {adapter_path.name}")
        weights = load_adapter_weights(adapter_path)
        all_weights.append(weights)

    # Merge using uniform averaging
    if show_progress:
        print(f"🧮 Computing uniform average...")
    merged_weights = merge_adapters_uniform(all_weights, merge_mode=merge_mode)

    # Save merged adapter
    save_merged_adapter(merged_weights, output_path, adapter_paths[0])

    if show_progress:
        print(f"✅ Merge complete!")


def resolve_adapter_path(task_name: str, run_id: str = "best") -> Path:
    """Resolve adapter path from task name and run ID.

    Args:
        task_name: Task name (e.g., 'asr', 'emotion', 'intent')
        run_id: Run ID or 'best'/'latest'

    Returns:
        Path to adapter directory
    """
    package_root = Path(__file__).resolve().parent.parent

    # Map task names to adapter directories
    task_adapter_map = {
        "asr": "artifacts/asr/adapters/qwen2_5_omni_lora_asr_100h",
        "emotion": "artifacts/emotion/adapters/qwen2_5_omni_lora_emotion_audio_v2",
        "intent": "artifacts/intent/adapters/qwen2_5_omni_lora_intent",
        "speaker_id": "artifacts/speaker_id/adapters/qwen2_5_omni_lora_speaker",
        "st": "artifacts/st/en_de/adapters/qwen2_5_omni_lora_st",
    }

    if task_name not in task_adapter_map:
        raise ValueError(
            f"Unknown task: {task_name}. "
            f"Available tasks: {', '.join(task_adapter_map.keys())}"
        )

    adapter_base = package_root / task_adapter_map[task_name]

    # Resolve run_id
    if run_id in ["best", "latest"]:
        adapter_path = adapter_base / run_id
        if not adapter_path.exists():
            raise ValueError(
                f"No '{run_id}' run found for {task_name} adapter at {adapter_base}"
            )
    else:
        adapter_path = adapter_base / "runs" / run_id
        if not adapter_path.exists():
            raise ValueError(
                f"Run '{run_id}' not found for {task_name} adapter at {adapter_base}"
            )

    return adapter_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters using various strategies"
    )

    parser.add_argument(
        "--adapters",
        nargs="+",
        required=True,
        help="Adapter paths or task names (e.g., 'asr emotion intent' or paths)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for merged adapter",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="task_vector",
        choices=["uniform", "task_vector"],
        help="Merging method: 'uniform' (direct LoRA merge) or 'task_vector' (merge via task vectors, works with different ranks)",
    )
    parser.add_argument(
        "--merge-mode",
        type=str,
        default="common",
        choices=["common", "strict"],
        help="How to handle different parameters: 'common' (only merge common params) or 'strict' (require identical params)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="best",
        help="Run ID to use for task names (default: 'best')",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    # Resolve adapter paths
    adapter_paths = []
    for adapter_spec in args.adapters:
        path = Path(adapter_spec)
        if path.exists():
            # Treat as direct path
            adapter_paths.append(path.resolve())
        else:
            # Treat as task name
            try:
                adapter_path = resolve_adapter_path(adapter_spec, args.run_id)
                adapter_paths.append(adapter_path)
            except ValueError as e:
                print(f"❌ Error resolving adapter '{adapter_spec}': {e}")
                return

    # Resolve output path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        package_root = Path(__file__).resolve().parent.parent
        output_path = package_root / output_path

    # Perform merge
    if args.method == "uniform":
        merge_uniform(adapter_paths, output_path, merge_mode=args.merge_mode)
    elif args.method == "task_vector":
        merge_uniform_via_task_vectors(adapter_paths, output_path, merge_mode=args.merge_mode)
    else:
        print(f"❌ Unsupported merge method: {args.method}")


if __name__ == "__main__":
    main()
