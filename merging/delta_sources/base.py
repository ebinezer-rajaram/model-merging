"""Base interfaces for adapter/merged delta sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple

import torch


@dataclass(frozen=True)
class ParamDeltaSpec:
    """Schema for one materializable dense parameter delta."""

    source_key: str
    shape: Tuple[int, ...]


@dataclass(frozen=True)
class SourceMetadata:
    """Source descriptor and metadata/provenance summary."""

    source_type: str
    source_id: str
    path: Optional[str] = None
    task: Optional[str] = None
    extra: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "source_type": self.source_type,
            "source_id": self.source_id,
        }
        if self.path is not None:
            payload["path"] = self.path
        if self.task is not None:
            payload["task"] = self.task
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload


@dataclass(frozen=True)
class ProvenanceNode:
    """Hierarchical provenance tree node for merged artifacts."""

    kind: str
    label: str
    params: Dict[str, object] = field(default_factory=dict)
    children: List["ProvenanceNode"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "label": self.label,
            "params": dict(self.params),
            "children": [child.to_dict() for child in self.children],
        }


class DeltaSource(Protocol):
    """Protocol for any source that can materialize per-parameter dense deltas."""

    @property
    def source_id(self) -> str:
        ...

    def metadata(self) -> SourceMetadata:
        ...

    def provenance(self) -> ProvenanceNode:
        ...

    def constituent_tasks_flat(self) -> List[str]:
        ...

    def list_target_params(self) -> List[ParamDeltaSpec]:
        ...

    def has_param(self, source_key: str) -> bool:
        ...

    def materialize_dense_param_delta(
        self,
        source_key: str,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        ...


__all__ = [
    "DeltaSource",
    "ParamDeltaSpec",
    "ProvenanceNode",
    "SourceMetadata",
]
