from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch

from merging.artifacts.continual_format import ContinualArtifactReader
from merging.continual.engine import continual_merge_sources_to_artifact
from merging.continual.policy import ContinualMergePolicy
from merging.delta_sources.base import ParamDeltaSpec, ProvenanceNode, SourceMetadata


class _FakeDeltaSource:
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


def test_continual_merge_engine_formula(tmp_path: Path) -> None:
    key = "base_model.model.model.layers.0.self_attn.q_proj"
    x = _FakeDeltaSource("x", {key: torch.ones(2, 2)}, tasks=["emotion"])
    y = _FakeDeltaSource("y", {key: torch.full((2, 2), 3.0)}, tasks=["asr"])

    policy = ContinualMergePolicy(alpha=2.0, lambda_weight=0.25)
    result = continual_merge_sources_to_artifact(
        x_source=x,
        y_source=y,
        policy=policy,
        output_dir=tmp_path / "merged",
        energy_threshold=1.0,
        merge_mode="strict",
        store_dtype="float32",
    )

    assert result.num_merged_params == 1

    reader = ContinualArtifactReader(result.artifact_dir)
    dense = reader.materialize_dense_param_delta(key, dtype=torch.float32)
    expected = 2.0 * (0.25 * torch.ones(2, 2) + 0.75 * torch.full((2, 2), 3.0))
    assert torch.allclose(dense, expected)
