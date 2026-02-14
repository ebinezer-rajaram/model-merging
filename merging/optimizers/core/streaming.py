"""Streaming parametrization helpers for AdaMerging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
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


@dataclass(frozen=True)
class StreamingLoraEntry:
    """Per-parameter LoRA factors required for fused LoRA merge."""

    param_key: str
    layer_idx: Optional[int]
    a_factors: Sequence[torch.Tensor]
    b_factors: Sequence[torch.Tensor]
    scales: Sequence[float]


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


class _WeightedDeltaApply(Function):
    """Memory-efficient weighted delta application.

    Avoids building a large autograd graph from repeated tensor add/mul ops.
    Backward computes gradients for coefficients directly from grad_output.
    """

    @staticmethod
    def forward(
        ctx,
        base: torch.Tensor,
        coeffs: torch.Tensor,
        deltas_source: Sequence[torch.Tensor],
        compute_dtype: torch.dtype,
    ) -> torch.Tensor:
        coeffs_compute = coeffs.to(device=base.device, dtype=compute_dtype)
        out = base.clone()
        for i, delta_src in enumerate(deltas_source):
            delta = delta_src.to(device=base.device, dtype=compute_dtype)
            out.add_(delta * coeffs_compute[i])
        ctx.deltas_source = deltas_source
        ctx.coeff_shape = tuple(coeffs.shape)
        ctx.coeff_dtype = coeffs.dtype
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_coeffs = torch.zeros(
            ctx.coeff_shape,
            device=grad_output.device,
            dtype=ctx.coeff_dtype,
        )
        grad_out_f32 = grad_output.float()
        for i, delta_src in enumerate(ctx.deltas_source):
            delta = delta_src.to(device=grad_output.device, dtype=torch.float32)
            # d(base + sum_i coeff_i * delta_i) / d(coeff_i) = delta_i
            grad_coeffs[i] = (grad_out_f32 * delta).sum().to(dtype=ctx.coeff_dtype)
        return None, grad_coeffs, None, None


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
        # Deltas are constants for lambda optimization: never track grad/history.
        self.deltas = [d.detach().requires_grad_(False) for d in deltas]
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

    def _get_delta_sources(self, *, device: torch.device, dtype: torch.dtype) -> Sequence[torch.Tensor]:
        if self.delta_residency == "gpu_cache":
            key = (device, dtype)
            cached = self._cached_deltas.get(key)
            if cached is None:
                cached = [delta.to(device=device, dtype=dtype) for delta in self.deltas]
                self._cached_deltas[key] = cached
            return cached
        if self.delta_residency == "cpu_stream":
            # Keep CPU tensors as source; stream one-at-a-time in custom autograd.
            return self.deltas
        raise ValueError(f"Unsupported delta_residency='{self.delta_residency}'.")

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        coeffs = self.coefficient_provider.get(self.layer_idx)
        compute_dtype = self._resolve_compute_dtype(base)
        delta_sources = self._get_delta_sources(device=base.device, dtype=compute_dtype)
        merged = _WeightedDeltaApply.apply(
            base.to(dtype=compute_dtype),
            coeffs,
            delta_sources,
            compute_dtype,
        )
        return merged.to(dtype=base.dtype)


def _resolve_module_and_param(model: nn.Module, param_key: str) -> Tuple[nn.Module, str]:
    if "." in param_key:
        module_path, param_name = param_key.rsplit(".", 1)
        module = model.get_submodule(module_path)
    else:
        module = model
        param_name = param_key
    return module, param_name


def _resolve_parent_and_name(model: nn.Module, module_path: str) -> Tuple[nn.Module, str]:
    if "." in module_path:
        parent_path, name = module_path.rsplit(".", 1)
        parent = model.get_submodule(parent_path)
    else:
        parent = model
        name = module_path
    return parent, name


class _FusedWeightedLinear(nn.Module):
    """Linear layer with additive weighted deltas applied at output compute time."""

    def __init__(
        self,
        *,
        base_linear: nn.Linear,
        layer_idx: Optional[int],
        deltas: Sequence[torch.Tensor],
        coefficient_provider: TaskCoefficientProvider,
        delta_residency: str,
        dtype_compute: str,
    ) -> None:
        super().__init__()
        self.base_linear = base_linear
        self.layer_idx = layer_idx
        self.deltas = [d.detach().requires_grad_(False) for d in deltas]
        self.coefficient_provider = coefficient_provider
        self.delta_residency = delta_residency
        self.dtype_compute = dtype_compute
        self._cached_deltas: Dict[Tuple[torch.device, torch.dtype], List[torch.Tensor]] = {}

    def _resolve_compute_dtype(self, x: torch.Tensor) -> torch.dtype:
        if self.dtype_compute == "auto":
            return self.base_linear.weight.dtype
        mapped = _DTYPE_ALIASES.get(self.dtype_compute)
        if mapped is None:
            raise ValueError(f"Unsupported dtype_compute='{self.dtype_compute}'.")
        return mapped

    def _get_delta_sources(self, *, device: torch.device, dtype: torch.dtype) -> Sequence[torch.Tensor]:
        if self.delta_residency == "gpu_cache":
            key = (device, dtype)
            cached = self._cached_deltas.get(key)
            if cached is None:
                cached = [delta.to(device=device, dtype=dtype) for delta in self.deltas]
                self._cached_deltas[key] = cached
            return cached
        if self.delta_residency == "cpu_stream":
            return self.deltas
        raise ValueError(f"Unsupported delta_residency='{self.delta_residency}'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coeffs = self.coefficient_provider.get(self.layer_idx)
        compute_dtype = self._resolve_compute_dtype(x)
        x_compute = x if x.dtype == compute_dtype else x.to(dtype=compute_dtype)
        base_weight = self.base_linear.weight.to(dtype=compute_dtype)
        base_bias = self.base_linear.bias
        if base_bias is not None and base_bias.dtype != compute_dtype:
            base_bias = base_bias.to(dtype=compute_dtype)
        out = F.linear(x_compute, base_weight, base_bias)
        coeffs_compute = coeffs.to(device=x_compute.device, dtype=compute_dtype)
        delta_sources = self._get_delta_sources(device=x_compute.device, dtype=compute_dtype)
        for i, delta_src in enumerate(delta_sources):
            delta = delta_src.to(device=x_compute.device, dtype=compute_dtype)
            out = out + (coeffs_compute[i] * F.linear(x_compute, delta, None))
        return out if out.dtype == x.dtype else out.to(dtype=x.dtype)


class _FusedLoraWeightedLinear(nn.Module):
    """Linear layer with weighted LoRA updates applied at output compute time."""

    def __init__(
        self,
        *,
        base_linear: nn.Linear,
        layer_idx: Optional[int],
        a_factors: Sequence[torch.Tensor],
        b_factors: Sequence[torch.Tensor],
        scales: Sequence[float],
        coefficient_provider: TaskCoefficientProvider,
        delta_residency: str,
        dtype_compute: str,
    ) -> None:
        super().__init__()
        self.base_linear = base_linear
        self.layer_idx = layer_idx
        self.a_factors = [a.detach().requires_grad_(False) for a in a_factors]
        self.b_factors = [b.detach().requires_grad_(False) for b in b_factors]
        self.scales = [float(s) for s in scales]
        self.coefficient_provider = coefficient_provider
        self.delta_residency = delta_residency
        self.dtype_compute = dtype_compute
        self._cached_factors: Dict[Tuple[torch.device, torch.dtype], List[Tuple[torch.Tensor, torch.Tensor]]] = {}

    def _resolve_compute_dtype(self, x: torch.Tensor) -> torch.dtype:
        if self.dtype_compute == "auto":
            return self.base_linear.weight.dtype
        mapped = _DTYPE_ALIASES.get(self.dtype_compute)
        if mapped is None:
            raise ValueError(f"Unsupported dtype_compute='{self.dtype_compute}'.")
        return mapped

    def _get_factor_sources(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Sequence[Tuple[torch.Tensor, torch.Tensor]]:
        if self.delta_residency == "gpu_cache":
            key = (device, dtype)
            cached = self._cached_factors.get(key)
            if cached is None:
                cached = [
                    (a.to(device=device, dtype=dtype), b.to(device=device, dtype=dtype))
                    for a, b in zip(self.a_factors, self.b_factors)
                ]
                self._cached_factors[key] = cached
            return cached
        if self.delta_residency == "cpu_stream":
            return list(zip(self.a_factors, self.b_factors))
        raise ValueError(f"Unsupported delta_residency='{self.delta_residency}'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coeffs = self.coefficient_provider.get(self.layer_idx)
        compute_dtype = self._resolve_compute_dtype(x)
        x_compute = x if x.dtype == compute_dtype else x.to(dtype=compute_dtype)
        base_weight = self.base_linear.weight.to(dtype=compute_dtype)
        base_bias = self.base_linear.bias
        if base_bias is not None and base_bias.dtype != compute_dtype:
            base_bias = base_bias.to(dtype=compute_dtype)
        out = F.linear(x_compute, base_weight, base_bias)
        coeffs_compute = coeffs.to(device=x_compute.device, dtype=compute_dtype)
        factor_sources = self._get_factor_sources(device=x_compute.device, dtype=compute_dtype)
        for i, (a_src, b_src) in enumerate(factor_sources):
            a = a_src.to(device=x_compute.device, dtype=compute_dtype)
            b = b_src.to(device=x_compute.device, dtype=compute_dtype)
            update = F.linear(F.linear(x_compute, a, None), b, None)
            out = out + (coeffs_compute[i] * self.scales[i] * update)
        return out if out.dtype == x.dtype else out.to(dtype=x.dtype)


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


def register_fused_linear_modules(
    *,
    model: nn.Module,
    entries: Iterable[StreamingDeltaEntry],
    coefficient_provider: TaskCoefficientProvider,
    delta_residency: str,
    dtype_compute: str,
) -> List[Tuple[nn.Module, str, nn.Module]]:
    """Replace eligible nn.Linear modules with fused weighted-delta wrappers."""
    handles: List[Tuple[nn.Module, str, nn.Module]] = []
    for entry in entries:
        if "." not in entry.param_key:
            continue
        module_path, param_name = entry.param_key.rsplit(".", 1)
        if param_name != "weight":
            continue
        try:
            module = model.get_submodule(module_path)
        except Exception:
            continue
        if not isinstance(module, nn.Linear):
            continue
        parent, child_name = _resolve_parent_and_name(model, module_path)
        fused = _FusedWeightedLinear(
            base_linear=module,
            layer_idx=entry.layer_idx,
            deltas=entry.deltas,
            coefficient_provider=coefficient_provider,
            delta_residency=delta_residency,
            dtype_compute=dtype_compute,
        )
        setattr(parent, child_name, fused)
        handles.append((parent, child_name, module))
    return handles


def unregister_fused_linear_modules(handles: Iterable[Tuple[nn.Module, str, nn.Module]]) -> None:
    """Restore modules replaced by `register_fused_linear_modules`."""
    for parent, child_name, original in handles:
        setattr(parent, child_name, original)


def register_fused_lora_linear_modules(
    *,
    model: nn.Module,
    entries: Iterable[StreamingLoraEntry],
    coefficient_provider: TaskCoefficientProvider,
    delta_residency: str,
    dtype_compute: str,
) -> List[Tuple[nn.Module, str, nn.Module]]:
    """Replace eligible nn.Linear modules with fused weighted-LoRA wrappers."""
    handles: List[Tuple[nn.Module, str, nn.Module]] = []
    errors: List[str] = []
    for entry in entries:
        if "." not in entry.param_key:
            errors.append(f"{entry.param_key}: parameter is not module-scoped.")
            continue
        module_path, param_name = entry.param_key.rsplit(".", 1)
        if param_name != "weight":
            errors.append(f"{entry.param_key}: only '.weight' parameters are supported.")
            continue
        try:
            module = model.get_submodule(module_path)
        except Exception:
            errors.append(f"{entry.param_key}: module path '{module_path}' not found.")
            continue
        if not isinstance(module, nn.Linear):
            errors.append(f"{entry.param_key}: module '{module_path}' is not nn.Linear.")
            continue
        if len(entry.a_factors) != len(entry.b_factors) or len(entry.a_factors) != len(entry.scales):
            errors.append(f"{entry.param_key}: inconsistent adapter factor counts.")
            continue
        if not entry.a_factors:
            errors.append(f"{entry.param_key}: empty LoRA factor set.")
            continue
        valid_shapes = True
        for idx, (a, b) in enumerate(zip(entry.a_factors, entry.b_factors)):
            if a.ndim != 2 or b.ndim != 2:
                errors.append(f"{entry.param_key}: adapter {idx} factors must be 2D tensors.")
                valid_shapes = False
                break
            rank = a.shape[0]
            if a.shape[1] != module.in_features:
                errors.append(
                    f"{entry.param_key}: adapter {idx} A shape {tuple(a.shape)} does not match in_features="
                    f"{module.in_features}."
                )
                valid_shapes = False
                break
            if b.shape[0] != module.out_features or b.shape[1] != rank:
                errors.append(
                    f"{entry.param_key}: adapter {idx} B shape {tuple(b.shape)} incompatible with "
                    f"out_features={module.out_features}, rank={rank}."
                )
                valid_shapes = False
                break
        if not valid_shapes:
            continue
        parent, child_name = _resolve_parent_and_name(model, module_path)
        fused = _FusedLoraWeightedLinear(
            base_linear=module,
            layer_idx=entry.layer_idx,
            a_factors=entry.a_factors,
            b_factors=entry.b_factors,
            scales=entry.scales,
            coefficient_provider=coefficient_provider,
            delta_residency=delta_residency,
            dtype_compute=dtype_compute,
        )
        setattr(parent, child_name, fused)
        handles.append((parent, child_name, module))
    if errors:
        sample = "; ".join(errors[:8])
        extra = " ..." if len(errors) > 8 else ""
        raise ValueError(f"fused_lora_linear registration failed: {sample}{extra}")
    if not handles:
        raise ValueError("fused_lora_linear registration produced no eligible nn.Linear modules.")
    return handles


def unregister_fused_lora_linear_modules(handles: Iterable[Tuple[nn.Module, str, nn.Module]]) -> None:
    """Restore modules replaced by `register_fused_lora_linear_modules`."""
    for parent, child_name, original in handles:
        setattr(parent, child_name, original)
