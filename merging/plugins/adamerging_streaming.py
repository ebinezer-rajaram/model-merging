"""Streaming parametrization helpers for AdaMerging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn.utils import parametrize


_DTYPE_ALIASES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


@dataclass(frozen=True)
class StreamingDeltaEntry:
    """Per-parameter delta data required for streaming merge."""

    param_key: str
    layer_idx: Optional[int]
    deltas: Sequence[torch.Tensor]


class TaskCoefficientProvider:
    """Holds current task/default/layer coefficients for parametrization forward passes."""

    def __init__(self) -> None:
        self._task_coeffs: Optional[torch.Tensor] = None
        self._default_coeffs: Optional[torch.Tensor] = None
        self._layer_coeffs: Dict[int, torch.Tensor] = {}

    def set_coefficients(
        self,
        *,
        task_coeffs: torch.Tensor,
        default_coeffs: Optional[torch.Tensor],
        layer_coeffs: Mapping[int, torch.Tensor],
    ) -> None:
        self._task_coeffs = task_coeffs
        self._default_coeffs = default_coeffs
        self._layer_coeffs = {int(k): v for k, v in layer_coeffs.items()}

    def get(self, layer_idx: Optional[int]) -> torch.Tensor:
        if self._task_coeffs is None:
            raise RuntimeError("TaskCoefficientProvider coefficients were not set before forward.")
        if layer_idx is not None and layer_idx in self._layer_coeffs:
            return self._layer_coeffs[layer_idx]
        if self._default_coeffs is not None:
            return self._default_coeffs
        return self._task_coeffs


class _WeightedDeltaParametrization(nn.Module):
    """Compute base + weighted delta sum on-demand for one parameter."""

    def __init__(
        self,
        *,
        layer_idx: Optional[int],
        deltas: Sequence[torch.Tensor],
        coefficient_provider: TaskCoefficientProvider,
        delta_residency: str,
        dtype_compute: str,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.deltas = [d.detach() for d in deltas]
        self.coefficient_provider = coefficient_provider
        self.delta_residency = delta_residency
        self.dtype_compute = dtype_compute
        self._cached_deltas: Dict[Tuple[torch.device, torch.dtype], List[torch.Tensor]] = {}

    def _resolve_compute_dtype(self, base: torch.Tensor) -> torch.dtype:
        if self.dtype_compute == "auto":
            return base.dtype
        mapped = _DTYPE_ALIASES.get(self.dtype_compute)
        if mapped is None:
            raise ValueError(f"Unsupported dtype_compute='{self.dtype_compute}'.")
        return mapped

    def _get_deltas(self, *, device: torch.device, dtype: torch.dtype) -> List[torch.Tensor]:
        if self.delta_residency == "gpu_cache":
            key = (device, dtype)
            cached = self._cached_deltas.get(key)
            if cached is None:
                cached = [delta.to(device=device, dtype=dtype) for delta in self.deltas]
                self._cached_deltas[key] = cached
            return cached
        if self.delta_residency == "cpu_stream":
            return [delta.to(device=device, dtype=dtype) for delta in self.deltas]
        raise ValueError(f"Unsupported delta_residency='{self.delta_residency}'.")

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        coeffs = self.coefficient_provider.get(self.layer_idx)
        compute_dtype = self._resolve_compute_dtype(base)
        deltas = self._get_deltas(device=base.device, dtype=compute_dtype)
        coeffs_cast = coeffs.to(device=base.device, dtype=compute_dtype)
        accum = torch.zeros_like(base, dtype=compute_dtype)
        for i, delta in enumerate(deltas):
            accum = accum + (delta * coeffs_cast[i])
        return base + accum.to(dtype=base.dtype)


def _resolve_module_and_param(model: nn.Module, param_key: str) -> Tuple[nn.Module, str]:
    if "." in param_key:
        module_path, param_name = param_key.rsplit(".", 1)
        module = model.get_submodule(module_path)
    else:
        module = model
        param_name = param_key
    return module, param_name


def register_streaming_parametrizations(
    *,
    model: nn.Module,
    entries: Iterable[StreamingDeltaEntry],
    coefficient_provider: TaskCoefficientProvider,
    delta_residency: str,
    dtype_compute: str,
) -> List[Tuple[nn.Module, str]]:
    """Register weighted-delta parametrizations and return handles for cleanup."""
    handles: List[Tuple[nn.Module, str]] = []
    for entry in entries:
        module, param_name = _resolve_module_and_param(model, entry.param_key)
        parametrization = _WeightedDeltaParametrization(
            layer_idx=entry.layer_idx,
            deltas=entry.deltas,
            coefficient_provider=coefficient_provider,
            delta_residency=delta_residency,
            dtype_compute=dtype_compute,
        )
        parametrize.register_parametrization(module, param_name, parametrization)
        handles.append((module, param_name))
    return handles


def unregister_streaming_parametrizations(handles: Iterable[Tuple[nn.Module, str]]) -> None:
    """Remove parametrizations registered by `register_streaming_parametrizations`."""
    for module, param_name in handles:
        parametrize.remove_parametrizations(module, param_name, leave_parametrized=False)
