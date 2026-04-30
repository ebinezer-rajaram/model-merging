from __future__ import annotations

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


def test_svd_zero_tensor_rank_zero() -> None:
    dense = torch.zeros(8, 4, dtype=torch.float32)
    compressed = compress_dense_delta_to_svd(
        dense,
        energy_threshold=0.99,
        compute_dtype=torch.float32,
        store_dtype="float32",
    )
    assert compressed.stats.rank == 0
