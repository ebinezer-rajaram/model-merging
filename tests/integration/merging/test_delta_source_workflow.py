from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from merging.delta_sources.lora_source import LoRADeltaSource
from merging.delta_sources.resolver import resolve_delta_source
from merging.runtime.task_vectors import extract_task_vector_from_lora, save_task_vector
from tests.helpers.merging import write_incomplete_lora_adapter, write_lora_adapter


def test_lora_delta_source_materializes_dense_delta_and_metadata(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    b = torch.eye(2, dtype=torch.float32)
    write_lora_adapter(adapter_dir, a=a, b=b, rank=2, alpha=4, key="base_model.model.model.layers.0.self_attn.q_proj")

    source = LoRADeltaSource(adapter_dir, task="asr")
    specs = source.list_target_params()
    assert len(specs) == 1
    assert source.metadata().task == "asr"
    assert source.constituent_tasks_flat() == ["asr"]

    dense = source.materialize_dense_param_delta(specs[0].source_key, dtype=torch.float32)
    assert torch.allclose(dense, (b @ a) * 2.0)
    with pytest.raises(KeyError):
        source.materialize_dense_param_delta("missing")


def test_lora_delta_source_rejects_bad_adapters(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        LoRADeltaSource(tmp_path / "missing")

    bad_rank = tmp_path / "bad_rank"
    write_lora_adapter(bad_rank, a=torch.ones(1, 2), b=torch.ones(2, 1), rank=0, alpha=1)
    with pytest.raises(ValueError, match="rank"):
        LoRADeltaSource(bad_rank)

    incomplete = tmp_path / "incomplete"
    write_incomplete_lora_adapter(incomplete)
    with pytest.raises(ValueError, match="Incomplete LoRA"):
        LoRADeltaSource(incomplete)


def test_task_vector_extraction_and_save_from_tiny_lora_adapter(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    write_lora_adapter(
        adapter,
        a=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        b=torch.eye(2),
        rank=2,
        alpha=4,
    )

    task_vector = extract_task_vector_from_lora(adapter)
    assert torch.allclose(task_vector["base_model.model.layer"], torch.tensor([[2.0, 4.0], [6.0, 8.0]]))

    output = tmp_path / "vectors" / "task_vector.safetensors"
    save_task_vector(task_vector, output, adapter)
    assert output.exists()
    assert json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))["num_parameters"] == 1


def test_resolver_handles_lora_paths_task_inference_and_invalid_modes(tmp_path: Path) -> None:
    adapter = tmp_path / "artifacts" / "kws" / "adapters" / "run"
    write_lora_adapter(adapter, a=torch.ones(1, 2), b=torch.ones(3, 1))

    source = resolve_delta_source(adapter)
    assert source.constituent_tasks_flat() == ["kws"]

    with pytest.raises(ValueError, match="continual_source_mode"):
        resolve_delta_source(adapter, continual_source_mode="bad")
    with pytest.raises(ValueError, match="neither a PEFT adapter"):
        resolve_delta_source(tmp_path)


def test_fused_lora_leaf_source_concatenates_weighted_prior_leaves(tmp_path: Path) -> None:
    speaker = tmp_path / "artifacts" / "speaker_ver" / "adapters" / "run"
    langid = tmp_path / "artifacts" / "langid" / "adapters" / "run"
    write_lora_adapter(speaker, a=torch.tensor([[1.0, 2.0]]), b=torch.tensor([[3.0], [4.0]]))
    write_lora_adapter(langid, a=torch.tensor([[5.0, 6.0]]), b=torch.tensor([[7.0], [8.0]]))

    artifact = tmp_path / "merged"
    factors = artifact / "factors"
    factors.mkdir(parents=True)
    save_file({"dummy": torch.zeros(1)}, str(factors / "shard_00000.safetensors"))
    (artifact / "continual_artifact_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "artifact_type": "continual_compressed_delta",
                "constituent_tasks_flat": ["speaker_ver", "langid"],
                "provenance_tree": {
                    "kind": "continual_merge",
                    "label": "continual_alpha_lambda",
                    "params": {"alpha": 2.0, "lambda": 0.25},
                    "children": [
                        {"kind": "lora_adapter", "label": "speaker", "params": {"path": str(speaker), "task": "speaker_ver"}, "children": []},
                        {"kind": "lora_adapter", "label": "langid", "params": {"path": str(langid), "task": "langid"}, "children": []},
                    ],
                },
                "target_params": [
                    {
                        "source_key": "base_model.model.layer",
                        "original_shape": [2, 2],
                        "matrix_shape": [2, 2],
                        "rank": 1,
                        "retained_energy": 1.0,
                        "relative_error": 0.0,
                        "frobenius_norm": 1.0,
                        "scale": 1.0,
                        "shard_file": "factors/shard_00000.safetensors",
                        "tensor_key_a": "dummy",
                        "tensor_key_b": "dummy",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    source = resolve_delta_source(artifact, continual_source_mode="fused_lora_leaves")
    a, b, scale = source.get_factor_tensors("base_model.model.layer")
    dense = b @ a * scale
    expected = (
        2.0 * 0.25 * torch.tensor([[3.0], [4.0]]) @ torch.tensor([[1.0, 2.0]])
        + 2.0 * 0.75 * torch.tensor([[7.0], [8.0]]) @ torch.tensor([[5.0, 6.0]])
    )

    assert source.constituent_tasks_flat() == ["speaker_ver", "langid"]
    assert a.shape == (2, 2)
    assert b.shape == (2, 2)
    assert torch.allclose(dense, expected)
