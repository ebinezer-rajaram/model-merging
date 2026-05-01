from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from safetensors.torch import save_file

from merging.delta_sources.base import ParamDeltaSpec, ProvenanceNode, SourceMetadata
from merging.engine.registry import MergeMethod, MergeOutput


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


def write_lora_adapter(
    path: Path,
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    rank: Optional[int] = None,
    alpha: Optional[int] = None,
    key: str = "base_model.model.layer",
    config_extra: Optional[Dict[str, Any]] = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    resolved_rank = int(rank if rank is not None else a.shape[0])
    resolved_alpha = int(alpha if alpha is not None else resolved_rank)
    config: Dict[str, Any] = {"r": resolved_rank, "lora_alpha": resolved_alpha}
    if config_extra:
        config.update(config_extra)
    (path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file(
        {
            f"{key}.lora_A.weight": a,
            f"{key}.lora_B.weight": b,
        },
        str(path / "adapter_model.safetensors"),
    )


def write_incomplete_lora_adapter(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(json.dumps({"r": 1, "lora_alpha": 1}), encoding="utf-8")
    save_file(
        {"base_model.model.layer.lora_A.weight": torch.ones(1, 2)},
        str(path / "adapter_model.safetensors"),
    )


def fake_merge_method(
    *,
    name: str = "unit_fake",
    required_params: tuple[str, ...] = (),
    saveable: bool = False,
    merge_in_memory=None,
    save_fn=None,
) -> MergeMethod:
    def _merge_in_memory(**kwargs):
        return MergeOutput(
            merged_delta={"x": torch.ones(1)},
            merged_weights={"x": torch.ones(1)},
            metadata={"merge_method": name, "source_adapters": kwargs.get("source_metadata", [])},
        )

    return MergeMethod(
        name=name,
        required_params=required_params,
        params_defaults={},
        params_validator=None,
        min_adapters=2,
        max_adapters=2,
        saveable=saveable,
        merge_in_memory=merge_in_memory or _merge_in_memory,
        save_fn=save_fn,
    )


def write_sweep_json(
    path: Path,
    *,
    method: str,
    adapters: List[str],
    merge_mode: str = "common",
    runs: Optional[List[Dict[str, Any]]] = None,
    best_index: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "method": method,
        "adapters": adapters,
        "merge_mode": merge_mode,
        "runs": runs or [],
    }
    if best_index is not None:
        payload["best_index"] = int(best_index)
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def assert_tensor_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.allclose(actual, expected), f"actual={actual} expected={expected}"
