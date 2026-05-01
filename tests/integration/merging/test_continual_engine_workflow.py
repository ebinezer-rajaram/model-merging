from __future__ import annotations

from pathlib import Path

import pytest
import torch

from merging.artifacts.continual_format import ContinualArtifactReader
from merging.continual.engine import continual_merge_sources_to_artifact
from merging.continual.policy import ContinualMergePolicy
from tests.helpers.merging import FakeDeltaSource


def test_continual_merge_engine_formula(tmp_path: Path) -> None:
    key = "base_model.model.model.layers.0.self_attn.q_proj"
    x = FakeDeltaSource("x", {key: torch.ones(2, 2)}, tasks=["emotion"])
    y = FakeDeltaSource("y", {key: torch.full((2, 2), 3.0)}, tasks=["asr"])

    result = continual_merge_sources_to_artifact(
        x_source=x,
        y_source=y,
        policy=ContinualMergePolicy(alpha=2.0, lambda_weight=0.25),
        output_dir=tmp_path / "merged",
        energy_threshold=1.0,
        merge_mode="strict",
        store_dtype="float32",
    )

    assert result.num_merged_params == 1
    reader = ContinualArtifactReader(result.artifact_dir)
    expected = 2.0 * (0.25 * torch.ones(2, 2) + 0.75 * torch.full((2, 2), 3.0))
    assert torch.allclose(reader.materialize_dense_param_delta(key, dtype=torch.float32), expected)


def test_continual_merge_common_mode_skips_shape_mismatch(tmp_path: Path) -> None:
    x = FakeDeltaSource("x", {"shared": torch.ones(2, 2), "x_only": torch.ones(1)})
    y = FakeDeltaSource("y", {"shared": torch.ones(3, 1), "y_only": torch.ones(1)})

    with pytest.raises(ValueError, match="no parameters"):
        continual_merge_sources_to_artifact(
            x_source=x,
            y_source=y,
            policy=ContinualMergePolicy(alpha=1.0, lambda_weight=0.5),
            output_dir=tmp_path / "merged",
            energy_threshold=1.0,
            merge_mode="common",
            store_dtype="float32",
        )


def test_continual_merge_strict_mode_rejects_key_and_shape_mismatches(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="identical key sets"):
        continual_merge_sources_to_artifact(
            x_source=FakeDeltaSource("x", {"x": torch.ones(1)}),
            y_source=FakeDeltaSource("y", {"y": torch.ones(1)}),
            policy=ContinualMergePolicy(alpha=1.0, lambda_weight=0.5),
            output_dir=tmp_path / "keys",
            energy_threshold=1.0,
            merge_mode="strict",
        )

    with pytest.raises(ValueError, match="Shape mismatch"):
        continual_merge_sources_to_artifact(
            x_source=FakeDeltaSource("x", {"shared": torch.ones(2, 2)}),
            y_source=FakeDeltaSource("y", {"shared": torch.ones(3, 1)}),
            policy=ContinualMergePolicy(alpha=1.0, lambda_weight=0.5),
            output_dir=tmp_path / "shapes",
            energy_threshold=1.0,
            merge_mode="strict",
        )
