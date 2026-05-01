from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from merging.artifacts.continual_format import ContinualArtifactReader, ContinualArtifactWriter, load_continual_manifest
from merging.delta_sources.compressed_source import CompressedMergedDeltaSource


def _write_artifact(out_dir: Path, *, shard_max_tensors: int = 2048) -> Path:
    writer = ContinualArtifactWriter(
        output_dir=out_dir,
        dense_merge_semantics={"semantic_type": "test"},
        stored_representation={"type": "factorized_svd", "energy_threshold": 0.99},
        provenance_tree={"kind": "root", "label": "root", "params": {}, "children": [{"kind": "leaf", "label": "asr"}]},
        constituent_tasks_flat=["emotion", "asr"],
        source_metadata=[{"source_type": "lora_adapter", "source_id": "a"}],
        shard_max_tensors=shard_max_tensors,
    )
    writer.add_param(
        source_key="base_model.model.model.layers.0.self_attn.q_proj",
        a_factor=torch.eye(2),
        b_factor=torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        scale=1.0,
        original_shape=(2, 2),
        matrix_shape=(2, 2),
        rank=2,
        retained_energy=1.0,
        relative_error=0.0,
        frobenius_norm=2.0,
    )
    return writer.finalize(diagnostics={"ok": True})


def test_continual_artifact_writer_reader_and_compressed_source_roundtrip(tmp_path: Path) -> None:
    out_dir = tmp_path / "artifact"
    _write_artifact(out_dir)

    reader = ContinualArtifactReader(out_dir)
    keys = reader.list_param_keys()
    assert len(keys) == 1
    assert torch.allclose(reader.materialize_dense_param_delta(keys[0], dtype=torch.float32), torch.eye(2) * 2.0)

    source = CompressedMergedDeltaSource(out_dir)
    assert source.metadata().source_type == "compressed_merged_artifact"
    assert source.provenance().children[0].label == "asr"
    assert source.constituent_tasks_flat() == ["emotion", "asr"]
    assert source.has_param(keys[0]) is True
    assert source.has_param("missing") is False
    assert torch.allclose(source.materialize_dense_param_delta(keys[0], dtype=torch.float32), torch.eye(2) * 2.0)


def test_continual_artifact_validation_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_continual_manifest(tmp_path / "missing")

    artifact = tmp_path / "bad"
    artifact.mkdir()
    (artifact / "continual_artifact_manifest.json").write_text(
        json.dumps({"schema_version": "bad", "artifact_type": "continual_compressed_delta", "target_params": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported continual artifact schema"):
        load_continual_manifest(artifact)

    empty_writer = ContinualArtifactWriter(
        output_dir=tmp_path / "empty",
        dense_merge_semantics={},
        stored_representation={},
        provenance_tree={"kind": "root", "label": "root"},
        constituent_tasks_flat=[],
    )
    with pytest.raises(ValueError, match="no parameters"):
        empty_writer.finalize()


def test_continual_artifact_writer_shards_multiple_params(tmp_path: Path) -> None:
    writer = ContinualArtifactWriter(
        output_dir=tmp_path / "sharded",
        dense_merge_semantics={},
        stored_representation={},
        provenance_tree={"kind": "root", "label": "root"},
        constituent_tasks_flat=[],
        shard_max_tensors=2,
    )
    for idx in range(3):
        writer.add_param(
            source_key=f"p{idx}",
            a_factor=torch.ones(1, 2),
            b_factor=torch.ones(2, 1),
            scale=1.0,
            original_shape=(2, 2),
            matrix_shape=(2, 2),
            rank=1,
            retained_energy=1.0,
            relative_error=0.0,
            frobenius_norm=1.0,
        )
    manifest_path = writer.finalize()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["shards"]) == 3
