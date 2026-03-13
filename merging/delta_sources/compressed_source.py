"""Delta source backed by a reusable continual compressed artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch

from merging.artifacts.continual_format import ContinualArtifactReader
from merging.delta_sources.base import ParamDeltaSpec, ProvenanceNode, SourceMetadata


class CompressedMergedDeltaSource:
    """Compressed reusable merged artifact with lazy dense materialization."""

    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = Path(artifact_dir).resolve()
        self._reader = ContinualArtifactReader(self.artifact_dir)
        self._source_id = str(self.artifact_dir)

        self._param_specs: List[ParamDeltaSpec] = []
        for key in self._reader.list_param_keys():
            entry = self._reader.get_entry(key)
            shape = tuple(int(x) for x in entry["original_shape"])
            self._param_specs.append(ParamDeltaSpec(source_key=key, shape=shape))

    @property
    def source_id(self) -> str:
        return self._source_id

    def metadata(self) -> SourceMetadata:
        manifest = self._reader.manifest
        return SourceMetadata(
            source_type="compressed_merged_artifact",
            source_id=self._source_id,
            path=str(self.artifact_dir),
            task=None,
            extra={
                "schema_version": manifest.get("schema_version"),
                "num_params": len(self._param_specs),
                "created_at": manifest.get("created_at"),
                "stored_representation": manifest.get("stored_representation", {}),
            },
        )

    def provenance(self) -> ProvenanceNode:
        payload = self._reader.manifest.get("provenance_tree")
        if isinstance(payload, dict):
            return _provenance_from_dict(payload)
        return ProvenanceNode(
            kind="compressed_merged_artifact",
            label=self.artifact_dir.name,
            params={"path": str(self.artifact_dir)},
            children=[],
        )

    def constituent_tasks_flat(self) -> List[str]:
        tasks = self._reader.manifest.get("constituent_tasks_flat")
        if isinstance(tasks, list):
            return [str(x) for x in tasks if str(x)]
        return []

    def list_target_params(self) -> List[ParamDeltaSpec]:
        return list(self._param_specs)

    def has_param(self, source_key: str) -> bool:
        try:
            self._reader.get_entry(source_key)
        except KeyError:
            return False
        return True

    def materialize_dense_param_delta(
        self,
        source_key: str,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        return self._reader.materialize_dense_param_delta(source_key, dtype=dtype, device=device)

    def get_factor_tensors(self, source_key: str) -> tuple[torch.Tensor, torch.Tensor, float]:
        return self._reader.get_factor_tensors(source_key)


def _provenance_from_dict(payload: Dict[str, object]) -> ProvenanceNode:
    raw_children = payload.get("children")
    children: List[ProvenanceNode] = []
    if isinstance(raw_children, list):
        for child in raw_children:
            if isinstance(child, dict):
                children.append(_provenance_from_dict(child))
    params = payload.get("params")
    return ProvenanceNode(
        kind=str(payload.get("kind", "unknown")),
        label=str(payload.get("label", "unknown")),
        params=dict(params) if isinstance(params, dict) else {},
        children=children,
    )


__all__ = ["CompressedMergedDeltaSource"]
