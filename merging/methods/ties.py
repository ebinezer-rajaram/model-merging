"""TIES merge method scaffold.

This is intentionally registered as a separate merge method (not a transform).
Algorithm internals are pending a dedicated implementation pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from merging.engine.registry import MergeOutput, build_merge_metadata


def merge_ties_scaffold(
    *,
    adapter_paths: List[Path],
    source_metadata: List[Dict],
    merge_mode: str,
    params: Optional[Dict[str, object]],
) -> MergeOutput:
    metadata = build_merge_metadata(
        method="ties",
        merge_mode=merge_mode,
        num_adapters=len(adapter_paths),
        source_metadata=source_metadata,
        num_parameters=0,
        params=params or {},
        method_params=dict(params or {}),
        lambda_policy=(params or {}).get("lambda_policy"),
        transforms=(params or {}).get("transforms"),
        optimizer=(params or {}).get("optimizer"),
    )
    metadata["scaffold"] = {
        "implemented": False,
        "message": "TIES method scaffold only. Algorithm implementation is pending.",
    }
    raise NotImplementedError(
        "ties merge method is scaffolded but not implemented yet. "
        "Use weighted/uniform/task_vector/weighted_delta for runnable merges."
    )


__all__ = ["merge_ties_scaffold"]
