"""Exact weighted LoRA source reconstructed from continual provenance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch

from merging.artifacts.continual_format import load_continual_manifest
from merging.delta_sources.base import ParamDeltaSpec, ProvenanceNode, SourceMetadata
from merging.delta_sources.lora_source import LoRADeltaSource
from merging.runtime.utils import infer_task_from_path


@dataclass(frozen=True)
class _WeightedLeaf:
    coefficient: float
    source: LoRADeltaSource


def _ordered_unique(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in values:
        value = str(raw).strip().lower()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _provenance_from_dict(payload: Mapping[str, Any]) -> ProvenanceNode:
    children = [
        _provenance_from_dict(child)
        for child in payload.get("children", [])
        if isinstance(child, Mapping)
    ]
    params = payload.get("params")
    return ProvenanceNode(
        kind=str(payload.get("kind", "unknown")),
        label=str(payload.get("label", "unknown")),
        params=dict(params) if isinstance(params, Mapping) else {},
        children=children,
    )


def _infer_task(path: Path, raw_task: object = None) -> Optional[str]:
    if isinstance(raw_task, str) and raw_task.strip():
        return raw_task.strip().lower()
    return infer_task_from_path(str(path))


def _collect_weighted_lora_leaves(node: Mapping[str, Any], coefficient: float) -> List[_WeightedLeaf]:
    kind = str(node.get("kind", "")).strip().lower()
    params = node.get("params")
    params_map = dict(params) if isinstance(params, Mapping) else {}
    children = [child for child in node.get("children", []) if isinstance(child, Mapping)]

    if kind == "continual_merge":
        alpha = float(params_map.get("alpha", 1.0))
        lambda_weight = float(params_map.get("lambda", params_map.get("lambda_weight", 0.5)))
        if len(children) != 2:
            raise ValueError("continual_merge provenance nodes must have exactly two children.")
        return [
            *_collect_weighted_lora_leaves(children[0], coefficient * alpha * lambda_weight),
            *_collect_weighted_lora_leaves(children[1], coefficient * alpha * (1.0 - lambda_weight)),
        ]

    if kind == "lora_adapter":
        raw_path = params_map.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("lora_adapter provenance node is missing params.path.")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return [
            _WeightedLeaf(
                coefficient=float(coefficient),
                source=LoRADeltaSource(path, task=_infer_task(path, params_map.get("task"))),
            )
        ]

    if children:
        leaves: List[_WeightedLeaf] = []
        for child in children:
            leaves.extend(_collect_weighted_lora_leaves(child, coefficient))
        return leaves

    raise ValueError(f"Cannot reconstruct exact LoRA leaves from provenance kind='{kind}'.")


class WeightedLoRACompositeSource:
    """A fixed weighted sum of original LoRA leaves exposed as one DeltaSource.

    For fused LoRA evaluation, the weighted sum is represented exactly by rank
    concatenation: sum_i c_i * s_i * B_i @ A_i = B_concat @ A_concat.
    """

    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = Path(artifact_dir).resolve()
        self.manifest = load_continual_manifest(self.artifact_dir)
        provenance = self.manifest.get("provenance_tree")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"Continual artifact lacks a provenance_tree: {self.artifact_dir}")
        self._provenance = dict(provenance)
        self._leaves = _collect_weighted_lora_leaves(self._provenance, 1.0)
        if not self._leaves:
            raise ValueError(f"No LoRA leaves reconstructed from {self.artifact_dir}")

        specs_by_key: Dict[str, ParamDeltaSpec] = {}
        for leaf in self._leaves:
            for spec in leaf.source.list_target_params():
                prior = specs_by_key.get(spec.source_key)
                if prior is not None and tuple(prior.shape) != tuple(spec.shape):
                    raise ValueError(
                        f"Shape mismatch across weighted LoRA leaves for '{spec.source_key}': "
                        f"{tuple(prior.shape)} vs {tuple(spec.shape)}"
                    )
                specs_by_key[spec.source_key] = spec
        self._param_specs = [specs_by_key[key] for key in sorted(specs_by_key)]

    @property
    def source_id(self) -> str:
        return str(self.artifact_dir)

    def metadata(self) -> SourceMetadata:
        return SourceMetadata(
            source_type="weighted_lora_composite",
            source_id=self.source_id,
            path=str(self.artifact_dir),
            task=None,
            extra={
                "num_leaves": len(self._leaves),
                "num_params": len(self._param_specs),
                "source_artifact_type": self.manifest.get("artifact_type"),
            },
        )

    def provenance(self) -> ProvenanceNode:
        return _provenance_from_dict(self._provenance)

    def constituent_tasks_flat(self) -> List[str]:
        manifest_tasks = self.manifest.get("constituent_tasks_flat")
        tasks: List[str] = []
        if isinstance(manifest_tasks, list):
            tasks.extend(str(task) for task in manifest_tasks if str(task))
        for leaf in self._leaves:
            tasks.extend(leaf.source.constituent_tasks_flat())
        return _ordered_unique(tasks)

    def list_target_params(self) -> List[ParamDeltaSpec]:
        return list(self._param_specs)

    def has_param(self, source_key: str) -> bool:
        return any(leaf.source.has_param(source_key) for leaf in self._leaves)

    def materialize_dense_param_delta(
        self,
        source_key: str,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        out: Optional[torch.Tensor] = None
        for leaf in self._leaves:
            if not leaf.source.has_param(source_key):
                continue
            delta = leaf.source.materialize_dense_param_delta(source_key, dtype=dtype, device=device)
            weighted = delta * float(leaf.coefficient)
            out = weighted if out is None else out + weighted
        if out is None:
            raise KeyError(f"Weighted LoRA source '{self.source_id}' has no parameter '{source_key}'")
        return out

    def get_factor_tensors(self, source_key: str) -> tuple[torch.Tensor, torch.Tensor, float]:
        a_parts: List[torch.Tensor] = []
        b_parts: List[torch.Tensor] = []
        for leaf in self._leaves:
            if not leaf.source.has_param(source_key):
                continue
            a, b, scale = leaf.source.get_factor_tensors(source_key)
            a_parts.append(a)
            b_parts.append(b * float(leaf.coefficient) * float(scale))
        if not a_parts:
            raise KeyError(f"Weighted LoRA source '{self.source_id}' has no parameter '{source_key}'")
        return torch.cat(a_parts, dim=0), torch.cat(b_parts, dim=1), 1.0


__all__ = ["WeightedLoRACompositeSource"]
