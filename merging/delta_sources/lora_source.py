"""Delta source backed by a standard PEFT LoRA adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import torch
from safetensors import safe_open

from merging.delta_sources.base import ParamDeltaSpec, ProvenanceNode, SourceMetadata


def _resolve_pattern_value(pattern: Mapping[str, object], key: str, default: float) -> float:
    if key in pattern:
        return float(pattern[key])
    suffix_matches = [k for k in pattern.keys() if key.endswith(str(k))]
    if not suffix_matches:
        return float(default)
    best = max(suffix_matches, key=len)
    return float(pattern[best])


class LoRADeltaSource:
    """LoRA adapter source that materializes exact dense deltas on demand."""

    def __init__(self, adapter_path: Path, *, task: Optional[str] = None) -> None:
        self.adapter_path = Path(adapter_path).resolve()
        self._source_id = str(self.adapter_path)

        config_path = self.adapter_path / "adapter_config.json"
        weight_path = self.adapter_path / "adapter_model.safetensors"
        if not config_path.exists():
            raise FileNotFoundError(f"LoRA adapter config not found: {config_path}")
        if not weight_path.exists():
            raise FileNotFoundError(f"LoRA adapter weights not found: {weight_path}")

        with config_path.open("r") as handle:
            cfg = json.load(handle)

        default_rank = float(cfg.get("r", 0))
        default_alpha = float(cfg.get("lora_alpha", 0))
        if default_rank <= 0:
            raise ValueError(f"Adapter rank must be positive for {self.adapter_path}")

        rank_pattern = cfg.get("rank_pattern")
        alpha_pattern = cfg.get("alpha_pattern")
        if rank_pattern is not None and not isinstance(rank_pattern, Mapping):
            raise ValueError("rank_pattern must be a mapping when provided.")
        if alpha_pattern is not None and not isinstance(alpha_pattern, Mapping):
            raise ValueError("alpha_pattern must be a mapping when provided.")

        self._rank_pattern = dict(rank_pattern or {})
        self._alpha_pattern = dict(alpha_pattern or {})
        self._default_rank = default_rank
        self._default_alpha = default_alpha
        self._task = task

        pairs: Dict[str, Dict[str, torch.Tensor]] = {}
        with safe_open(weight_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if ".lora_A." in key:
                    base_key = key.replace(".lora_A.weight", "")
                    pairs.setdefault(base_key, {})["A"] = handle.get_tensor(key).clone()
                elif ".lora_B." in key:
                    base_key = key.replace(".lora_B.weight", "")
                    pairs.setdefault(base_key, {})["B"] = handle.get_tensor(key).clone()

        missing = [k for k, pair in pairs.items() if "A" not in pair or "B" not in pair]
        if missing:
            sample = ", ".join(sorted(missing)[:8])
            extra = " ..." if len(missing) > 8 else ""
            raise ValueError(
                f"Incomplete LoRA A/B factor pairs for {len(missing)} modules: {sample}{extra}"
            )

        self._pairs = pairs
        self._param_specs: List[ParamDeltaSpec] = []
        for key in sorted(self._pairs.keys()):
            a = self._pairs[key]["A"]
            b = self._pairs[key]["B"]
            if a.ndim != 2 or b.ndim != 2:
                raise ValueError(f"LoRA factors for '{key}' must both be 2D.")
            shape = (int(b.shape[0]), int(a.shape[1]))
            self._param_specs.append(ParamDeltaSpec(source_key=key, shape=shape))

    @property
    def source_id(self) -> str:
        return self._source_id

    def metadata(self) -> SourceMetadata:
        return SourceMetadata(
            source_type="lora_adapter",
            source_id=self._source_id,
            path=str(self.adapter_path),
            task=self._task,
            extra={
                "num_params": len(self._param_specs),
            },
        )

    def provenance(self) -> ProvenanceNode:
        label = self._task or self.adapter_path.name
        return ProvenanceNode(
            kind="lora_adapter",
            label=label,
            params={
                "path": str(self.adapter_path),
                "task": self._task,
            },
            children=[],
        )

    def constituent_tasks_flat(self) -> List[str]:
        return [self._task] if self._task else []

    def list_target_params(self) -> List[ParamDeltaSpec]:
        return list(self._param_specs)

    def has_param(self, source_key: str) -> bool:
        return source_key in self._pairs

    def _scale_for_key(self, source_key: str) -> float:
        rank = _resolve_pattern_value(self._rank_pattern, source_key, self._default_rank)
        alpha = _resolve_pattern_value(self._alpha_pattern, source_key, self._default_alpha)
        if rank <= 0:
            raise ValueError(f"Resolved non-positive rank for key '{source_key}'")
        return float(alpha / rank)

    def materialize_dense_param_delta(
        self,
        source_key: str,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if source_key not in self._pairs:
            raise KeyError(f"LoRA source '{self._source_id}' has no parameter '{source_key}'")
        pair = self._pairs[source_key]
        a = pair["A"]
        b = pair["B"]
        scale = self._scale_for_key(source_key)

        out_dtype = dtype if dtype is not None else torch.float32
        out_device = device if device is not None else torch.device("cpu")
        a_cast = a.to(device=out_device, dtype=out_dtype)
        b_cast = b.to(device=out_device, dtype=out_dtype)

        dense = (b_cast @ a_cast) * scale
        return dense

    def get_factor_tensors(self, source_key: str) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Return low-rank factors in the same orientation as continual SVD artifacts."""
        if source_key not in self._pairs:
            raise KeyError(f"LoRA source '{self._source_id}' has no parameter '{source_key}'")
        pair = self._pairs[source_key]
        return pair["A"].clone(), pair["B"].clone(), self._scale_for_key(source_key)


__all__ = ["LoRADeltaSource"]
