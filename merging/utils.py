"""Shared utilities for adapter merging operations.

This module provides common functionality used across all merging methods:
- Adapter path resolution (finding best adapters by task name)
- Loading and saving adapter weights
- Metadata management
- Evaluation of merged adapters
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Add package root to path for imports
CURRENT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CURRENT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core import load_config
from core.training.run_manager import RunManager


# Task registry mapping task names to their config getters
# This is extensible - add new tasks here as they're created
TASK_REGISTRY = {
    "asr": "tasks.asr",
    "emotion": "tasks.emotion",
    "intent": "tasks.intent",
    "speaker_id": "tasks.speaker_id",
    "speaker_ver": "tasks.speaker_ver",
    "speech_qa": "tasks.speech_qa",
    "st": "tasks.st",
    "langid": "tasks.langid",
    "kws": "tasks.kws",
}


def get_task_module(task_name: str):
    """Dynamically import task module to get artifact directories.

    Args:
        task_name: Name of the task (e.g., "asr", "emotion")

    Returns:
        Task module with get_artifact_directories function

    Raises:
        ValueError: If task is not registered
    """
    if task_name not in TASK_REGISTRY:
        available = ", ".join(sorted(TASK_REGISTRY.keys()))
        raise ValueError(
            f"Unknown task: '{task_name}'. Available tasks: {available}"
        )

    module_path = TASK_REGISTRY[task_name]
    module = __import__(module_path, fromlist=["get_artifact_directories"])
    return module


def resolve_best_adapter(task_name: str) -> Tuple[Path, Dict]:
    """Resolve the best adapter for a given task.

    This function:
    1. Loads the task configuration to find adapter_subdir
    2. Gets artifact directories for the task
    3. Reads runs_registry.json to find the best performing run
    4. Returns path to best adapter and its metadata

    Args:
        task_name: Name of the task (e.g., "asr", "emotion")

    Returns:
        Tuple of (adapter_path, metadata_dict) where:
        - adapter_path: Path to best adapter directory
        - metadata_dict: Contains task, path, run_id, metrics, config_hash

    Raises:
        ValueError: If task is unknown or best adapter not found
        FileNotFoundError: If registry or adapter doesn't exist

    Example:
        >>> adapter_path, meta = resolve_best_adapter("asr")
        >>> print(adapter_path)
        /path/to/artifacts/asr/adapters/qwen2_5_omni_lora_asr_100h/best
        >>> print(meta["metrics"])
        {"wer": 0.022, "eval_loss": 0.0646}
    """
    # Get task module to access config path and artifact directories
    task_module = get_task_module(task_name)

    # Get config path for the task
    config_path = task_module.get_config_path(PACKAGE_ROOT)

    # Load task config to get adapter_subdir
    config = load_config(config_path)
    adapter_subdir = config["artifacts"]["adapter_subdir"]

    # Get artifact directories for this task
    artifact_dirs = task_module.get_artifact_directories(PACKAGE_ROOT)
    adapter_base = artifact_dirs["adapters"] / adapter_subdir

    # Read runs registry
    registry_path = adapter_base / "runs_registry.json"
    if not registry_path.exists():
        raise FileNotFoundError(
            f"No runs registry found for {task_name} at {registry_path}. "
            f"Has this task been trained yet?"
        )

    with open(registry_path) as f:
        registry = json.load(f)

    # Find best run
    best_run = next((r for r in registry["runs"] if r.get("is_best")), None)
    if not best_run:
        raise ValueError(
            f"No best run found for {task_name}. "
            f"Registry at {registry_path} may be corrupted."
        )

    # Resolve best symlink
    best_path = adapter_base / "best"
    if not best_path.exists():
        # Fallback to direct run path
        best_path = adapter_base / "runs" / best_run["run_id"]

    if not best_path.exists():
        raise FileNotFoundError(
            f"Best adapter not found at {best_path} for task {task_name}"
        )

    # Build metadata
    metadata = {
        "task": task_name,
        "path": str(best_path.resolve()),
        "run_id": best_run["run_id"],
        "metrics": best_run.get("metrics", {}),
        "config_hash": best_run.get("config_hash"),
        "timestamp": best_run.get("timestamp"),
    }

    return best_path, metadata


def load_adapter_weights(adapter_path: Path) -> Dict[str, torch.Tensor]:
    """Load LoRA adapter weights from SafeTensors file.

    Args:
        adapter_path: Path to adapter directory containing adapter_model.safetensors

    Returns:
        Dictionary mapping parameter names to tensors

    Raises:
        FileNotFoundError: If adapter_model.safetensors not found

    Example:
        >>> weights = load_adapter_weights(Path("artifacts/asr/adapters/.../best"))
        >>> print(list(weights.keys())[:3])
        ['base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight',
         'base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight',
         'base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight']
    """
    safetensors_path = adapter_path / "adapter_model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(
            f"Adapter weights not found at {safetensors_path}"
        )

    weights = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key).clone()

    return weights


def create_merged_adapter_config(
    reference_adapter_path: Path,
    merged_weights: Dict[str, torch.Tensor],
) -> Dict:
    """Create adapter config for merged adapter.

    Reads the reference adapter's config and updates target_modules
    to only include modules that are present in the merged weights.

    Args:
        reference_adapter_path: Path to reference adapter for base config
        merged_weights: Merged adapter weights dictionary

    Returns:
        Updated adapter config dictionary

    Example:
        >>> config = create_merged_adapter_config(ref_path, merged_weights)
        >>> print(config["target_modules"])
        ['q_proj', 'k_proj', 'v_proj', 'o_proj', ...]
    """
    config_path = reference_adapter_path / "adapter_config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Extract unique module names from merged weights
    # e.g., "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight" -> "q_proj"
    merged_modules = set()
    for key in merged_weights.keys():
        parts = key.split(".")
        # Find the module name before lora_A or lora_B
        for i, part in enumerate(parts):
            if part in ["lora_A", "lora_B"]:
                # Handle special case: audio_tower.proj
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
    metadata: Dict,
    register_run: bool = True,
) -> None:
    """Save merged adapter in PEFT format with metadata.

    This function:
    1. Creates output directory
    2. Saves weights as adapter_model.safetensors
    3. Creates adapter_config.json based on merged weights
    4. Saves merge_metadata.json with source information
    5. Optionally registers the run with RunManager

    Args:
        weights: Merged adapter weights dictionary
        output_path: Directory to save merged adapter (should be a run directory)
        reference_adapter_path: Path to reference adapter for config
        metadata: Merge metadata (method, sources, parameters, etc.)
        register_run: Whether to register this run with RunManager

    Example:
        >>> save_merged_adapter(
        ...     weights=merged_weights,
        ...     output_path=Path("artifacts/merged/uniform/asr_emotion/runs/run_20260116_143052"),
        ...     reference_adapter_path=asr_adapter_path,
        ...     metadata={
        ...         "merge_method": "uniform",
        ...         "source_adapters": [meta1, meta2],
        ...         ...
        ...     }
        ... )
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Save weights as SafeTensors
    safetensors_path = output_path / "adapter_model.safetensors"
    save_file(weights, str(safetensors_path))

    # Create and save adapter config
    config = create_merged_adapter_config(reference_adapter_path, weights)
    config_path = output_path / "adapter_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Save merge metadata
    metadata_path = output_path / "merge_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Register run if requested
    if register_run:
        # Create RunManager for the adapter base directory
        adapter_base = output_path.parent.parent  # runs/run_xxx -> base

        # Use timestamp as the ranking metric for merged adapters
        # (could be customized based on evaluation results)
        run_manager = RunManager(
            adapter_dir=adapter_base,
            metric_for_ranking="timestamp",
            greater_is_better=False,  # Earlier merges aren't necessarily better
        )

        # Register the run
        run_manager.register_run(
            run_dir=output_path,
            metrics={"timestamp": datetime.now().timestamp()},
            config=metadata,
        )

    print(f"✅ Saved merged adapter to {output_path}")
    print(f"   - Weights: {safetensors_path} ({safetensors_path.stat().st_size / 1e6:.1f} MB)")
    print(f"   - Config: {config_path}")
    print(f"   - Metadata: {metadata_path}")
    print(f"   - Target modules: {', '.join(config['target_modules'])}")


def create_merge_output_path(
    method: str,
    task_names: List[str],
    extra_params: Optional[Dict] = None,
    base_dir: Optional[Path] = None,
) -> Path:
    """Create output path for merged adapter with timestamped run directory.

    Generates paths like:
    - artifacts/merged/uniform/asr_emotion/runs/run_20260116_143052/
    - artifacts/merged/weighted/asr_emotion_lambda0.7/runs/run_20260116_144523/

    Args:
        method: Merge method name (e.g., "uniform", "weighted")
        task_names: List of task names being merged
        extra_params: Optional dict with extra parameters (e.g., {"lambda": 0.7})
        base_dir: Optional base directory (defaults to PACKAGE_ROOT/artifacts/merged)

    Returns:
        Path to run directory for the merged adapter

    Example:
        >>> path = create_merge_output_path("weighted", ["asr", "emotion"], {"lambda": 0.7})
        >>> print(path)
        /path/to/artifacts/merged/weighted/asr_emotion_lambda0.7/runs/run_20260116_143052
    """
    if base_dir is None:
        base_dir = PACKAGE_ROOT / "artifacts" / "merged"

    # Create task combination name
    task_combo = "_".join(sorted(task_names))

    # Add extra parameters to name if provided
    if extra_params:
        for key, value in sorted(extra_params.items()):
            # Format floats nicely (0.7 instead of 0.70000)
            if isinstance(value, float):
                task_combo += f"_{key}{value:.10g}"
            else:
                task_combo += f"_{key}{value}"

    # Create full path: merged/{method}/{task_combo}/runs/run_{timestamp}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"

    output_path = base_dir / method / task_combo / "runs" / run_id
    output_path.mkdir(parents=True, exist_ok=True)

    return output_path


def evaluate_merged_adapter(
    adapter_path: Path,
    source_tasks: List[str],
    split: str = "test",
    save_results: bool = True,
) -> Dict[str, Dict]:
    """Evaluate merged adapter on all source tasks.

    Runs evaluation on each source task and optionally saves results
    to eval_results.json in the adapter directory.

    Args:
        adapter_path: Path to merged adapter directory
        source_tasks: List of task names to evaluate on
        split: Dataset split to evaluate (train, validation, test)
        save_results: Whether to save results to eval_results.json

    Returns:
        Dictionary mapping task names to their evaluation metrics

    Example:
        >>> results = evaluate_merged_adapter(
        ...     Path("artifacts/merged/uniform/asr_emotion/best"),
        ...     ["asr", "emotion"],
        ...     split="test"
        ... )
        >>> print(results)
        {
            "asr": {"wer": 0.025, "loss": 0.072},
            "emotion": {"accuracy": 0.812, "loss": 0.543}
        }
    """
    from experiments.evaluate_task import evaluate

    results = {}
    print(f"\n📊 Evaluating merged adapter on {len(source_tasks)} tasks ({split} split)")

    for i, task in enumerate(source_tasks, 1):
        print(f"\n[{i}/{len(source_tasks)}] Evaluating on {task}...")

        try:
            result = evaluate(
                task=task,
                adapter=str(adapter_path),
                split=split,
                enable_cache=False,
                show_summary=True,
            )
            results[task] = result.metrics

            # Print key metrics
            print(f"✅ {task} evaluation complete:")
            for key, value in sorted(result.metrics.items())[:5]:  # Show top 5 metrics
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.4f}")

        except Exception as e:
            print(f"❌ Failed to evaluate on {task}: {e}")
            results[task] = {"error": str(e)}

    # Save results if requested
    if save_results:
        results_path = adapter_path / "eval_results.json"
        with open(results_path, "w") as f:
            json.dump({
                "split": split,
                "timestamp": datetime.now().isoformat(),
                "results": results,
            }, f, indent=2)
        print(f"\n💾 Evaluation results saved to {results_path}")

    return results


__all__ = [
    "resolve_best_adapter",
    "load_adapter_weights",
    "create_merged_adapter_config",
    "save_merged_adapter",
    "create_merge_output_path",
    "evaluate_merged_adapter",
    "get_task_module",
    "TASK_REGISTRY",
]
