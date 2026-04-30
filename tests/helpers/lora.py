from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from safetensors.torch import save_file

from merging.delta_sources.base import ParamDeltaSpec, ProvenanceNode, SourceMetadata


class FakeDeltaSource:
    def __init__(self, source_id: str, deltas: Dict[str, torch.Tensor], tasks: Optional[List[str]] = None) -> None:
        self._source_id = source_id
        self._deltas = dict(deltas)
        self._tasks = list(tasks or [])

    @property
    def source_id(self) -> str:
        return self._source_id

    def metadata(self) -> SourceMetadata:
        return SourceMetadata(source_type="fake", source_id=self._source_id, task=None)

    def provenance(self) -> ProvenanceNode:
        return ProvenanceNode(kind="fake", label=self._source_id)

    def constituent_tasks_flat(self) -> List[str]:
        return list(self._tasks)

    def list_target_params(self) -> List[ParamDeltaSpec]:
        return [
            ParamDeltaSpec(source_key=key, shape=tuple(int(x) for x in value.shape))
            for key, value in sorted(self._deltas.items())
        ]

    def has_param(self, source_key: str) -> bool:
        return source_key in self._deltas

    def materialize_dense_param_delta(
        self,
        source_key: str,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        out = self._deltas[source_key]
        if dtype is not None:
            out = out.to(dtype=dtype)
        if device is not None:
            out = out.to(device=device)
        return out


def write_lora_adapter(path: Path, *, a: torch.Tensor, b: torch.Tensor) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(
        json.dumps({"r": int(a.shape[0]), "lora_alpha": int(a.shape[0])}),
        encoding="utf-8",
    )
    save_file(
        {
            "base_model.model.layer.lora_A.weight": a,
            "base_model.model.layer.lora_B.weight": b,
        },
        str(path / "adapter_model.safetensors"),
    )
