"""Shared utilities for adapter merging operations.

This module provides common functionality used across all merging methods:
- Adapter path resolution (finding best adapters by task name)
- Loading and saving adapter weights
- Metadata management
- Evaluation of merged adapters
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Add repo root to path for imports
# `merging/core/utils.py` lives at `<repo>/merging/core/utils.py`, so the repo root is two levels up.
CURRENT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CURRENT_DIR.parent.parent
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


def _format_lambda(value: float) -> str:
    """Format lambda values consistently for path tags."""
    return f"{value:g}"


def _sanitize_run_token(value: object) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    token = token.strip("_")
    return token


def _optimizer_run_tag(extra_params: Optional[Dict]) -> Optional[str]:
    if not isinstance(extra_params, dict):
        return None
    optimizer = extra_params.get("optimizer")
    if not isinstance(optimizer, dict):
        return None
    optimizer_type = _sanitize_run_token(optimizer.get("type", ""))
    if not optimizer_type or optimizer_type == "none":
        return None

    provenance = optimizer.get("provenance")
    params = optimizer.get("params")
    variant_raw = None
    plus_plus = False
    if isinstance(provenance, dict):
        variant_raw = provenance.get("variant")
        plus_plus = bool(provenance.get("plus_plus", False))
    if variant_raw is None and isinstance(params, dict):
        variant_raw = params.get("variant")
        plus_plus = plus_plus or bool(params.get("plus_plus", False))

    variant = _sanitize_run_token(variant_raw) if variant_raw else ""
    if variant.endswith("_plus_plus"):
        plus_plus = True
        variant = variant.replace("_plus_plus", "")
    elif variant in {"plus_plus", "plusplus"}:
        plus_plus = True
        variant = "task_wise"

    parts = [p for p in [optimizer_type, variant] if p]
    if plus_plus:
        parts.append("plusplus")
    if not parts:
        return None
    return "_".join(parts)


def _is_run_dir(path: Path) -> bool:
    """Check if a path looks like a merged adapter run directory."""
    return (path / "adapter_model.safetensors").exists() or (path / "adapter_config.json").exists()


def _select_run_dir(path: Path, run_id: Optional[str] = None) -> Path:
    """Resolve a merged adapter run directory from a base path."""
    path = path.resolve()
    if _is_run_dir(path):
        return path

    if run_id:
        if run_id in ("best", "latest"):
            candidate = path / run_id
        else:
            candidate = path / "runs" / run_id
        if candidate.exists():
            return candidate.resolve()
        raise ValueError(f"Run '{run_id}' not found under {path}")

    for label in ("best", "latest"):
        candidate = path / label
        if candidate.exists():
            return candidate.resolve()

    runs_dir = path / "runs"
    if runs_dir.exists():
        runs = [p for p in runs_dir.iterdir() if p.is_dir()]
        if len(runs) == 1:
            return runs[0].resolve()

    raise ValueError(f"Could not resolve a merged adapter run under {path}. Specify --run-id.")


def resolve_merged_adapter_path(
    *,
    adapter_path: Optional[str | Path] = None,
    method: Optional[str] = None,
    task_names: Optional[List[str]] = None,
    lambda_weight: Optional[float] = None,
    run_id: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> Path:
    """Resolve a merged adapter run directory from either a path or merge spec."""
    if base_dir is None:
        base_dir = PACKAGE_ROOT / "artifacts" / "merged"

    if adapter_path is not None:
        path = Path(adapter_path).expanduser()
        if not path.is_absolute():
            path = PACKAGE_ROOT / path
        return _select_run_dir(path, run_id=run_id)

    if not method or not task_names:
        raise ValueError("Provide either adapter_path or both method and task_names.")

    task_combo_given = "_".join(task_names)
    task_combo_sorted = "_".join(sorted(task_names))
    lambda_tag = _format_lambda(lambda_weight) if lambda_weight is not None else None

    combos = {task_combo_given, task_combo_sorted}
    if len(task_names) == 2:
        combos.add("_".join([task_names[1], task_names[0]]))

    candidates: List[Path] = []
    for combo in combos:
        if not combo:
            continue
        if lambda_tag is not None:
            candidates.append(base_dir / method / f"{combo}_lambda{lambda_tag}")
            candidates.append(base_dir / method / combo / lambda_tag)
        candidates.append(base_dir / method / combo)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return _select_run_dir(candidate, run_id=run_id)

    expected = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not resolve merged adapter directory. Checked:\n  - "
        f"{expected}"
    )


def load_merge_metadata(adapter_path: Path) -> Dict:
    """Load merge metadata from a merged adapter run directory."""
    metadata_path = adapter_path / "merge_metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r") as handle:
        return json.load(handle)


def infer_task_from_path(path_str: str) -> Optional[str]:
    """Infer a task name from an adapter path when possible."""
    normalized = Path(path_str).resolve()
    parts = normalized.parts
    if "artifacts" in parts:
        idx = parts.index("artifacts")
        if idx + 1 < len(parts) and parts[idx + 1] in TASK_REGISTRY:
            return parts[idx + 1]
    return None


def _unique_items(items: Iterable[str]) -> List[str]:
    """Preserve order while de-duplicating items."""
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def infer_merged_source_tasks(metadata: Dict, fallback: Optional[List[str]] = None) -> List[str]:
    """Infer merged source tasks from merge metadata or fallback list."""
    tasks: List[str] = []
    for entry in metadata.get("source_adapters", []):
        task_name = entry.get("task")
        if task_name:
            tasks.append(task_name)
            continue
        path_str = entry.get("path")
        if path_str:
            inferred = infer_task_from_path(path_str)
            if inferred:
                tasks.append(inferred)

    if not tasks and fallback:
        tasks = list(fallback)

    return _unique_items(tasks)


def build_merge_tag(metadata: Dict, task_names: Optional[List[str]] = None) -> str:
    """Build a stable tag for merged adapters for metric file names."""
    method = metadata.get("merge_method", "merged")
    lambda_weight = metadata.get("lambda")
    tasks = task_names or infer_merged_source_tasks(metadata)
    tag_parts = ["merged", method]
    if tasks:
        tag_parts.extend(tasks)
    if lambda_weight is not None:
        tag_parts.append(f"lambda{_format_lambda(lambda_weight)}")
    return "_".join(tag_parts)


def resolve_merge_summary_dir(
    method_name: str,
    source_tasks: Optional[List[str]],
    metadata: Dict,
    base_dir: Optional[Path] = None,
) -> Path:
    """Resolve directory for merge evaluation summaries."""
    if base_dir is None:
        base_dir = PACKAGE_ROOT / "artifacts" / "merged"

    summary_dir = base_dir / method_name
    if source_tasks:
        task_combo = "_".join(sorted(source_tasks))
        summary_dir = summary_dir / task_combo
    return summary_dir


def resolve_merge_eval_dir(
    method_name: str,
    source_tasks: Optional[List[str]],
    split: str,
    base_dir: Optional[Path] = None,
) -> Path:
    """Resolve directory for merged evaluation outputs."""
    summary_dir = resolve_merge_summary_dir(method_name, source_tasks, {}, base_dir=base_dir)
    return summary_dir / "eval" / split


def update_results_index(
    eval_dir: Path,
    *,
    merge_tag: str,
    split: str,
    results_path: Path,
    metadata: Dict,
    summary: Dict,
    run_path: Optional[Path] = None,
) -> None:
    """Append an entry to the eval results index."""
    index_path = eval_dir.parent / "index.json"
    entry = {
        "timestamp": summary.get("timestamp"),
        "merge_tag": merge_tag,
        "split": split,
        "results_path": str(results_path),
        "run_path": str(run_path) if run_path else None,
        "method": metadata.get("merge_method"),
        "params": metadata.get("params", {}),
        "lambda": metadata.get("lambda"),
        "source_tasks": summary.get("source_tasks"),
        "evaluated_tasks": summary.get("evaluated_tasks"),
    }

    if index_path.exists():
        try:
            with index_path.open("r") as handle:
                index = json.load(handle)
        except Exception:
            index = {}
    else:
        index = {}

    index.setdefault("entries", [])
    index["entries"].append(entry)

    with index_path.open("w") as handle:
        json.dump(index, handle, indent=2)


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

    def _filter_metrics_for_metadata(raw: Dict) -> Dict:
        """Keep only small scalar metrics; drop large arrays (e.g., eval__predictions)."""
        filtered: Dict = {}
        for key, value in (raw or {}).items():
            # Common offenders: eval__predictions / eval__labels
            if "__" in key:
                continue
            if isinstance(value, (int, float, bool, str)):
                filtered[key] = value
        return filtered

    # Build metadata
    metadata = {
        "task": task_name,
        "path": str(best_path.resolve()),
        "run_id": best_run["run_id"],
        "metrics": _filter_metrics_for_metadata(best_run.get("metrics", {})),
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


def compute_delta_from_lora_weights(
    lora_weights: Dict[str, torch.Tensor],
    reference_adapter_path: Path,
) -> Dict[str, torch.Tensor]:
    """Compute full-rank task vectors (delta weights) from LoRA weights."""
    config_path = reference_adapter_path / "adapter_config.json"
    with open(config_path, "r") as handle:
        config = json.load(handle)

    lora_alpha = config["lora_alpha"]
    lora_r = config["r"]
    scaling = lora_alpha / lora_r

    lora_pairs: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in lora_weights.items():
        if ".lora_A." in key:
            base_key = key.replace(".lora_A.weight", "")
            lora_pairs.setdefault(base_key, {})["A"] = tensor
        elif ".lora_B." in key:
            base_key = key.replace(".lora_B.weight", "")
            lora_pairs.setdefault(base_key, {})["B"] = tensor

    task_vectors: Dict[str, torch.Tensor] = {}
    for base_key, pair in lora_pairs.items():
        if "A" in pair and "B" in pair:
            task_vectors[base_key] = (pair["B"] @ pair["A"]) * scaling

    return task_vectors


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
    - artifacts/merged/weighted/asr_emotion/runs/run_20260116_144523/

    Args:
        method: Merge method name (e.g., "uniform", "weighted")
        task_names: List of task names being merged
        extra_params: Optional dict with extra parameters (e.g., {"lambda": 0.7}).
            When optimizer metadata is present, optimizer type/variant are encoded in run folder name.
        base_dir: Optional base directory (defaults to PACKAGE_ROOT/artifacts/merged)

    Returns:
        Path to run directory for the merged adapter

    Example:
        >>> path = create_merge_output_path("weighted", ["asr", "emotion"], {"lambda": 0.7})
        >>> print(path)
        /path/to/artifacts/merged/weighted/asr_emotion/runs/run_20260116_143052
    """
    if base_dir is None:
        base_dir = PACKAGE_ROOT / "artifacts" / "merged"

    # Create task combination name
    task_combo = "_".join(sorted(task_names))

    # Create full path: merged/{method}/{task_combo}/runs/run_{timestamp}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = _optimizer_run_tag(extra_params)
    run_id = f"run_{run_tag}_{timestamp}" if run_tag else f"run_{timestamp}"

    output_path = base_dir / method / task_combo / "runs" / run_id
    output_path.mkdir(parents=True, exist_ok=True)

    return output_path


__all__ = [
    "resolve_best_adapter",
    "load_adapter_weights",
    "compute_delta_from_lora_weights",
    "create_merged_adapter_config",
    "save_merged_adapter",
    "create_merge_output_path",
    "get_task_module",
    "TASK_REGISTRY",
    "resolve_merged_adapter_path",
    "load_merge_metadata",
    "infer_merged_source_tasks",
    "infer_task_from_path",
    "build_merge_tag",
    "resolve_merge_summary_dir",
    "resolve_merge_eval_dir",
    "update_results_index",
]
