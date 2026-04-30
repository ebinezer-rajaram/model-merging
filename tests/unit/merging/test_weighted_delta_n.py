from __future__ import annotations

import pytest
import torch

from merging.methods.weighted_delta_n import merge_task_vectors_weighted_n


def test_weighted_delta_n_merges_common_keys_with_layer_override() -> None:
    key0 = "base_model.model.model.layers.0.self_attn.q_proj.weight"
    key1 = "base_model.model.model.layers.1.self_attn.q_proj.weight"
    vectors = [
        {key0: torch.ones(2), key1: torch.ones(2) * 10.0},
        {key0: torch.ones(2) * 3.0, key1: torch.ones(2) * 20.0},
    ]

    merged = merge_task_vectors_weighted_n(
        vectors,
        layer_task_coefficients={0: [0.25, 0.75]},
        default_task_coefficients=[0.5, 0.5],
        normalize_coefficients=False,
    )

    assert torch.allclose(merged[key0], torch.ones(2) * 2.5)
    assert torch.allclose(merged[key1], torch.ones(2) * 15.0)


def test_weighted_delta_n_validates_modes_and_coefficients() -> None:
    vectors = [{"x": torch.ones(1)}, {"x": torch.ones(1)}]

    with pytest.raises(ValueError, match="at least 2"):
        merge_task_vectors_weighted_n([{"x": torch.ones(1)}])
    with pytest.raises(ValueError, match="Unsupported merge_mode"):
        merge_task_vectors_weighted_n(vectors, merge_mode="bad")
    with pytest.raises(ValueError, match="length=2"):
        merge_task_vectors_weighted_n(vectors, task_coefficients=[1.0])
    with pytest.raises(ValueError, match="Coefficient sum"):
        merge_task_vectors_weighted_n(vectors, task_coefficients=[0.0, 0.0], normalize_coefficients=True)


def test_weighted_delta_n_strict_mode_rejects_key_mismatch() -> None:
    with pytest.raises(ValueError, match="different parameters"):
        merge_task_vectors_weighted_n(
            [{"x": torch.ones(1)}, {"y": torch.ones(1)}],
            merge_mode="strict",
        )
