"""SVD compression utilities for per-parameter dense deltas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class SVDCompressionStats:
    """Compression diagnostics for one parameter."""

    original_shape: Tuple[int, ...]
    matrix_shape: Tuple[int, int]
    rank: int
    max_rank: int
    retained_energy: float
    relative_error: float
    frobenius_norm: float


@dataclass(frozen=True)
class CompressedParam:
    """Factorized representation of one dense parameter delta."""

    a_factor: torch.Tensor
    b_factor: torch.Tensor
    scale: float
    stats: SVDCompressionStats


def _reshape_for_svd(tensor: torch.Tensor) -> tuple[torch.Tensor, Tuple[int, ...], Tuple[int, int]]:
    """Convert an arbitrary dense tensor into a 2D matrix for SVD."""
    if tensor.ndim == 0:
        original_shape = ()
        matrix = tensor.reshape(1, 1)
    elif tensor.ndim == 1:
        original_shape = tuple(int(x) for x in tensor.shape)
        matrix = tensor.reshape(1, int(tensor.shape[0]))
    else:
        original_shape = tuple(int(x) for x in tensor.shape)
        rows = int(tensor.shape[0])
        cols = int(tensor.numel() // rows)
        matrix = tensor.reshape(rows, cols)

    matrix_shape = (int(matrix.shape[0]), int(matrix.shape[1]))
    return matrix, original_shape, matrix_shape


def _resolve_store_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype).strip().lower()
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported SVD store dtype: {dtype}")


def _select_rank_by_energy(singular_values: torch.Tensor, energy_threshold: float) -> int:
    if singular_values.numel() == 0:
        return 0
    if not 0.0 < float(energy_threshold) <= 1.0:
        raise ValueError("energy_threshold must be in (0,1].")

    s2 = singular_values.square()
    total = float(s2.sum().item())
    if total <= 0.0:
        return 0

    cumulative = torch.cumsum(s2, dim=0) / total
    # first index where retained energy >= threshold
    idx = int(torch.searchsorted(cumulative, torch.tensor(float(energy_threshold), device=cumulative.device)).item())
    return min(int(singular_values.numel()), idx + 1)


def compress_dense_delta_to_svd(
    dense_delta: torch.Tensor,
    *,
    energy_threshold: float,
    min_rank: int = 0,
    max_rank: Optional[int] = None,
    compute_dtype: torch.dtype = torch.float32,
    store_dtype: str | torch.dtype = torch.float16,
) -> CompressedParam:
    """Compress a dense delta tensor with truncated SVD using energy retention."""
    if not torch.isfinite(dense_delta).all():
        raise ValueError("dense_delta contains non-finite values.")
    if min_rank < 0:
        raise ValueError(f"min_rank must be >= 0, got {min_rank}")

    matrix, original_shape, matrix_shape = _reshape_for_svd(dense_delta)
    rows, cols = matrix_shape
    max_possible_rank = min(rows, cols)

    if max_rank is None:
        max_rank = max_possible_rank
    if max_rank < 0:
        raise ValueError(f"max_rank must be >= 0, got {max_rank}")
    max_rank = min(max_possible_rank, max_rank)

    if min_rank > max_rank:
        raise ValueError(
            f"min_rank ({min_rank}) cannot be greater than effective max_rank ({max_rank})."
        )

    matrix_compute = matrix.to(dtype=compute_dtype)
    try:
        u, s, vh = torch.linalg.svd(matrix_compute, full_matrices=False)
    except RuntimeError as exc:
        raise RuntimeError(f"SVD failed for matrix shape {matrix_shape}: {exc}") from exc

    selected_rank = _select_rank_by_energy(s, energy_threshold)
    selected_rank = max(min_rank, min(selected_rank, max_rank))

    if selected_rank == 0:
        b = torch.zeros((rows, 0), dtype=_resolve_store_dtype(store_dtype))
        a = torch.zeros((0, cols), dtype=_resolve_store_dtype(store_dtype))
        reconstructed = torch.zeros_like(matrix_compute)
        retained = 1.0 if float(matrix_compute.square().sum().item()) == 0.0 else 0.0
    else:
        u_r = u[:, :selected_rank]
        s_r = s[:selected_rank]
        vh_r = vh[:selected_rank, :]

        # M ~= (U_r * S_r) @ Vh_r
        b_compute = u_r * s_r.unsqueeze(0)
        a_compute = vh_r
        reconstructed = b_compute @ a_compute
        total_energy = float(s.square().sum().item())
        retained = float(s_r.square().sum().item() / total_energy) if total_energy > 0.0 else 1.0

        out_dtype = _resolve_store_dtype(store_dtype)
        b = b_compute.to(dtype=out_dtype).contiguous()
        a = a_compute.to(dtype=out_dtype).contiguous()

    frob = float(torch.linalg.norm(matrix_compute).item())
    err = float(torch.linalg.norm((matrix_compute - reconstructed)).item())
    rel_err = (err / frob) if frob > 0.0 else 0.0

    stats = SVDCompressionStats(
        original_shape=original_shape,
        matrix_shape=matrix_shape,
        rank=int(selected_rank),
        max_rank=int(max_possible_rank),
        retained_energy=float(retained),
        relative_error=float(rel_err),
        frobenius_norm=float(frob),
    )
    return CompressedParam(a_factor=a, b_factor=b, scale=1.0, stats=stats)


def reconstruct_dense_delta_from_svd(
    *,
    a_factor: torch.Tensor,
    b_factor: torch.Tensor,
    scale: float,
    original_shape: Sequence[int],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Reconstruct a dense delta tensor from SVD factors and reshape to original shape."""
    if a_factor.ndim != 2 or b_factor.ndim != 2:
        raise ValueError("a_factor and b_factor must both be 2D tensors.")
    if b_factor.shape[1] != a_factor.shape[0]:
        raise ValueError(
            f"Factor rank mismatch: b_factor shape={tuple(b_factor.shape)} and a_factor shape={tuple(a_factor.shape)}"
        )

    rank = int(a_factor.shape[0])
    rows = int(b_factor.shape[0])
    cols = int(a_factor.shape[1])
    out_dtype = dtype if dtype is not None else torch.float32
    out_device = device if device is not None else b_factor.device

    if rank == 0:
        matrix = torch.zeros((rows, cols), device=out_device, dtype=out_dtype)
    else:
        b = b_factor.to(device=out_device, dtype=out_dtype)
        a = a_factor.to(device=out_device, dtype=out_dtype)
        matrix = (b @ a) * float(scale)

    if len(original_shape) == 0:
        return matrix.reshape(())
    if len(original_shape) == 1:
        return matrix.reshape(original_shape[0])
    return matrix.reshape(tuple(int(x) for x in original_shape))


__all__ = [
    "CompressedParam",
    "SVDCompressionStats",
    "compress_dense_delta_to_svd",
    "reconstruct_dense_delta_from_svd",
]
