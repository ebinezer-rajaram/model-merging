from __future__ import annotations

import math

import pytest
import torch

from merging.compression.svd import SVDCompressionStats, CompressedParam
from merging.continual.engine import (
    _aggregate_compression_stats,
    _merge_param_with_coefficients,
    _resolve_keys_for_merge,
    _resolve_merge_coefficients_for_existing_key,
    _validate_energy_threshold,
)
from merging.continual.policy import ContinualMergePolicy
from merging.delta_sources.base import ParamDeltaSpec


def test_policy_coefficients_validation_and_serialization() -> None:
    policy = ContinualMergePolicy(alpha=2.0, lambda_weight=0.25)
    x_coeff, y_coeff = policy.source_coefficients()
    assert math.isclose(x_coeff, 0.5)
    assert math.isclose(y_coeff, 1.5)
    assert policy.to_dict() == {"alpha": 2.0, "lambda": 0.25}

    with pytest.raises(ValueError):
        ContinualMergePolicy(alpha=-1.0, lambda_weight=0.5).validate()
    with pytest.raises(ValueError):
        ContinualMergePolicy(alpha=1.0, lambda_weight=1.5).validate()


def test_key_resolution_merge_modes_and_energy_validation() -> None:
    specs_a = [ParamDeltaSpec("a", (2,)), ParamDeltaSpec("b", (2,))]
    specs_b = [ParamDeltaSpec("b", (2,)), ParamDeltaSpec("c", (2,))]
    assert _resolve_keys_for_merge([specs_a, specs_b], merge_mode="common") == ["b"]
    with pytest.raises(ValueError, match="identical key sets"):
        _resolve_keys_for_merge([specs_a, specs_b], merge_mode="strict")
    with pytest.raises(ValueError, match="Unsupported merge_mode"):
        _resolve_keys_for_merge([specs_a], merge_mode="bad")

    _validate_energy_threshold(1.0)
    with pytest.raises(ValueError, match="energy_threshold"):
        _validate_energy_threshold(0.0)
    with pytest.raises(ValueError, match="energy_threshold"):
        _validate_energy_threshold(float("inf"))


def test_existing_merge_coefficient_recovery_and_merge_math() -> None:
    key = "base_model.model.model.layers.2.self_attn.q_proj.weight"
    assert _resolve_merge_coefficients_for_existing_key(
        merge_method="uniform_delta",
        merge_params={},
        coefficient_policy=None,
        source_key=key,
        num_sources=2,
    ) == [0.5, 0.5]
    assert _resolve_merge_coefficients_for_existing_key(
        merge_method="weighted_delta",
        merge_params={"lambda": 0.25},
        coefficient_policy=None,
        source_key=key,
        num_sources=2,
    ) == [0.25, 0.75]
    assert _resolve_merge_coefficients_for_existing_key(
        merge_method="weighted_delta_n",
        merge_params={},
        coefficient_policy={"layer_task_coefficients": {"2": [2.0, 1.0]}, "normalize_coefficients": True},
        source_key=key,
        num_sources=2,
    ) == pytest.approx([2 / 3, 1 / 3])

    merged = _merge_param_with_coefficients(
        [torch.ones(2), torch.full((2,), 3.0)],
        [0.25, 0.75],
        out_dtype=torch.float32,
    )
    assert torch.allclose(merged, torch.full((2,), 2.5))

    with pytest.raises(ValueError, match="exactly 2"):
        _resolve_merge_coefficients_for_existing_key(
            merge_method="weighted_delta",
            merge_params={"lambda": 0.5},
            coefficient_policy=None,
            source_key=key,
            num_sources=3,
        )
    with pytest.raises(ValueError, match="does not support"):
        _resolve_merge_coefficients_for_existing_key(
            merge_method="unknown",
            merge_params={},
            coefficient_policy=None,
            source_key=key,
            num_sources=2,
        )


def test_compression_stats_aggregate_empty_and_values() -> None:
    assert _aggregate_compression_stats([]) == {
        "avg_rank": 0.0,
        "avg_retained_energy": 0.0,
        "avg_relative_error": 0.0,
        "max_relative_error": 0.0,
    }
    params = [
        CompressedParam(torch.empty(0), torch.empty(0), 1.0, SVDCompressionStats((2, 2), (2, 2), 1, 2, 0.8, 0.2, 1.0)),
        CompressedParam(torch.empty(0), torch.empty(0), 1.0, SVDCompressionStats((2, 2), (2, 2), 2, 2, 1.0, 0.0, 2.0)),
    ]
    stats = _aggregate_compression_stats(params)
    assert stats["avg_rank"] == pytest.approx(1.5)
    assert stats["avg_retained_energy"] == pytest.approx(0.9)
    assert stats["max_relative_error"] == pytest.approx(0.2)
