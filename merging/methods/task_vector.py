"""Task-vector-based merging utilities."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch

from experiments.extract_vector import extract_task_vector_from_lora
from merging.runtime.utils import save_merged_adapter


def merge_uniform_via_task_vectors(
    adapter_paths: List[Path],
    output_path: Path,
    merge_mode: str = "common",
    show_progress: bool = True,
) -> None:
    """Merge adapters by extracting and merging task vectors.

    This approach works even when adapters have different LoRA ranks,
    as we merge in the full parameter space rather than the low-rank space.
    """
    if show_progress:
        print(f"🔀 Merging {len(adapter_paths)} adapters via task vectors")
        print(f"   Mode: {merge_mode}")
        for i, path in enumerate(adapter_paths, 1):
            print(f"   {i}. {path}")

    task_vectors: List[Dict[str, torch.Tensor]] = []
    for i, adapter_path in enumerate(adapter_paths, 1):
        if show_progress:
            print(f"\n📥 Extracting task vector {i}/{len(adapter_paths)}: {adapter_path.name}")
        tv = extract_task_vector_from_lora(adapter_path)
        task_vectors.append(tv)

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
    else:
        keys_to_merge = common_keys
        unique_keys = all_unique_keys - common_keys
        if unique_keys:
            print(f"⚠️  Warning: {len(unique_keys)} parameters are not common across all task vectors")
            print(f"   Only merging {len(keys_to_merge)} common parameters")

    merged_tv: Dict[str, torch.Tensor] = {}
    num_vectors = len(task_vectors)

    if show_progress:
        print(f"\n🧮 Computing uniform average of task vectors...")

    for key in keys_to_merge:
        shapes = [tv[key].shape for tv in task_vectors]
        if len(set(shapes)) > 1:
            print(f"⚠️  Skipping {key}: inconsistent shapes {shapes}")
            continue

        summed = torch.zeros_like(task_vectors[0][key])
        for tv in task_vectors:
            summed += tv[key]
        merged_tv[key] = summed / num_vectors

    metadata = {
        "merge_method": "task_vector",
        "merge_mode": merge_mode,
        "num_adapters": len(adapter_paths),
        "timestamp": datetime.now().isoformat(),
        "source_adapters": [{"path": str(p)} for p in adapter_paths],
        "num_parameters": len(merged_tv),
    }

    if show_progress:
        print(f"\n💾 Saving merged task vector to {output_path}")

    save_merged_adapter(
        weights=merged_tv,
        output_path=output_path,
        reference_adapter_path=adapter_paths[0],
        metadata=metadata,
        register_run=True,
    )

    if show_progress:
        print(f"\n✅ Task vector merge complete!")
