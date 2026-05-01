from __future__ import annotations

import pytest
import torch

from merging.compression.svd import compress_dense_delta_to_svd, reconstruct_dense_delta_from_svd


def test_svd_roundtrip_low_error_at_high_energy() -> None:
    torch.manual_seed(0)
    dense = torch.randn(16, 12, dtype=torch.float32)

    compressed = compress_dense_delta_to_svd(
        dense,
        energy_threshold=0.999,
        compute_dtype=torch.float32,
        store_dtype="float32",
    )
    reconstructed = reconstruct_dense_delta_from_svd(
        a_factor=compressed.a_factor,
        b_factor=compressed.b_factor,
        scale=compressed.scale,
        original_shape=compressed.stats.original_shape,
        dtype=torch.float32,
    )

    rel_err = torch.linalg.norm(dense - reconstructed) / torch.linalg.norm(dense)
    assert rel_err.item() < 0.2
    assert compressed.stats.retained_energy >= 0.999 - 1e-6
    assert compressed.stats.original_shape == tuple(dense.shape)
    assert compressed.stats.matrix_shape == tuple(dense.shape)


def test_svd_zero_tensor_rank_zero_and_shape_restoration() -> None:
    dense = torch.zeros(8, dtype=torch.float32)
    compressed = compress_dense_delta_to_svd(
        dense,
        energy_threshold=0.99,
        compute_dtype=torch.float32,
        store_dtype="float32",
    )
    assert compressed.stats.rank == 0
    reconstructed = reconstruct_dense_delta_from_svd(
        a_factor=compressed.a_factor,
        b_factor=compressed.b_factor,
        scale=1.0,
        original_shape=compressed.stats.original_shape,
        dtype=torch.float32,
    )
    assert reconstructed.shape == dense.shape
    assert torch.equal(reconstructed, dense)


def test_svd_validation_for_dtype_rank_and_factors() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        compress_dense_delta_to_svd(torch.tensor([float("nan")]), energy_threshold=1.0)
    with pytest.raises(ValueError, match="min_rank"):
        compress_dense_delta_to_svd(torch.ones(2, 2), energy_threshold=1.0, min_rank=-1)
    with pytest.raises(ValueError, match="cannot be greater"):
        compress_dense_delta_to_svd(torch.ones(2, 2), energy_threshold=1.0, min_rank=2, max_rank=1)
    with pytest.raises(ValueError, match="Unsupported SVD store dtype"):
        compress_dense_delta_to_svd(torch.ones(2, 2), energy_threshold=1.0, store_dtype="bad")
    with pytest.raises(ValueError, match="both be 2D"):
        reconstruct_dense_delta_from_svd(a_factor=torch.ones(1), b_factor=torch.ones(1, 1), scale=1.0, original_shape=(1,))
    with pytest.raises(ValueError, match="rank mismatch"):
        reconstruct_dense_delta_from_svd(a_factor=torch.ones(2, 2), b_factor=torch.ones(2, 1), scale=1.0, original_shape=(2, 2))
