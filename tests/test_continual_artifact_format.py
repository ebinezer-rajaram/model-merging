from __future__ import annotations

from pathlib import Path

import torch

from merging.artifacts.continual_format import ContinualArtifactReader, ContinualArtifactWriter


def test_continual_artifact_writer_reader_roundtrip(tmp_path: Path) -> None:
    out_dir = tmp_path / "artifact"
    writer = ContinualArtifactWriter(
        output_dir=out_dir,
        dense_merge_semantics={"semantic_type": "test"},
        stored_representation={"type": "factorized_svd", "energy_threshold": 0.99},
        provenance_tree={"kind": "root", "label": "root", "params": {}, "children": []},
        constituent_tasks_flat=["emotion", "asr"],
        source_metadata=[{"source_type": "lora_adapter", "source_id": "a"}],
    )

    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    b = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
    writer.add_param(
        source_key="base_model.model.model.layers.0.self_attn.q_proj",
        a_factor=a,
        b_factor=b,
        scale=1.0,
        original_shape=(2, 2),
        matrix_shape=(2, 2),
        rank=2,
        retained_energy=1.0,
        relative_error=0.0,
        frobenius_norm=2.0,
    )
    writer.finalize()

    reader = ContinualArtifactReader(out_dir)
    keys = reader.list_param_keys()
    assert len(keys) == 1
    dense = reader.materialize_dense_param_delta(keys[0], dtype=torch.float32)
    expected = b @ a
    assert torch.allclose(dense, expected)
