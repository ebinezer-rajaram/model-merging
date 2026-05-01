from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from merging.artifacts.continual_format import ContinualArtifactWriter
from merging.delta_sources.compressed_source import CompressedMergedDeltaSource
from merging.engine.registry import MergeMethod, MergeOutput
from merging.engine import runner
from merging.runtime.task_vectors import extract_task_vector_from_lora, save_task_vector


def _write_lora_adapter(adapter_dir: Path) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4}), encoding="utf-8")
    save_file(
        {
            "base_model.model.layer.lora_A.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "base_model.model.layer.lora_B.weight": torch.eye(2),
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )


def test_compressed_merged_delta_source_reads_metadata_provenance_and_dense_delta(tmp_path: Path) -> None:
    writer = ContinualArtifactWriter(
        output_dir=tmp_path / "artifact",
        dense_merge_semantics={"method": "unit"},
        stored_representation={"type": "svd"},
        provenance_tree={"kind": "merge", "label": "root", "children": [{"kind": "leaf", "label": "asr"}]},
        constituent_tasks_flat=["asr"],
        source_metadata=[{"source_type": "fake"}],
        shard_max_tensors=2,
    )
    writer.add_param(
        source_key="base_model.model.layer",
        a_factor=torch.eye(2),
        b_factor=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        scale=0.5,
        original_shape=(2, 2),
        matrix_shape=(2, 2),
        rank=2,
        retained_energy=1.0,
        relative_error=0.0,
        frobenius_norm=1.0,
    )
    writer.finalize(diagnostics={"ok": True})

    source = CompressedMergedDeltaSource(tmp_path / "artifact")
    assert source.metadata().source_type == "compressed_merged_artifact"
    assert source.provenance().children[0].label == "asr"
    assert source.constituent_tasks_flat() == ["asr"]
    assert source.has_param("base_model.model.layer") is True
    assert source.has_param("missing") is False
    dense = source.materialize_dense_param_delta("base_model.model.layer")
    assert torch.allclose(dense, torch.tensor([[0.5, 1.0], [1.5, 2.0]]))


def test_task_vector_extraction_and_save_from_tiny_lora_adapter(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    _write_lora_adapter(adapter)

    task_vector = extract_task_vector_from_lora(adapter)
    assert torch.allclose(task_vector["base_model.model.layer"], torch.tensor([[2.0, 4.0], [6.0, 8.0]]))

    output = tmp_path / "vectors" / "task_vector.safetensors"
    save_task_vector(task_vector, output, adapter)
    assert output.exists()
    assert json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))["num_parameters"] == 1


def test_run_merge_uses_resolved_adapters_optimizer_and_fake_method(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter_a = tmp_path / "a"
    adapter_b = tmp_path / "b"
    adapter_a.mkdir()
    adapter_b.mkdir()

    def fake_resolve(specs):
        assert specs == ["task_a", "task_b"]
        return [
            (adapter_a, {"task": "task_a", "metrics": {"accuracy": 0.8}}),
            (adapter_b, {"task": "task_b", "metrics": {"accuracy": 0.7}}),
        ]

    def fake_merge_in_memory(**kwargs):
        assert kwargs["adapter_paths"] == [adapter_a, adapter_b]
        assert kwargs["params"]["lambda"] == 0.6
        return MergeOutput(
            merged_delta={"x": torch.ones(1)},
            merged_weights=None,
            metadata={"merge_method": "fake", "source_adapters": kwargs["source_metadata"]},
        )

    fake_method = MergeMethod(
        name="fake",
        required_params=("lambda",),
        params_defaults={},
        params_validator=None,
        min_adapters=2,
        max_adapters=2,
        saveable=False,
        merge_in_memory=fake_merge_in_memory,
    )

    monkeypatch.setattr(runner, "resolve_adapter_specs", fake_resolve)
    monkeypatch.setattr(runner, "get_merge_method", lambda name: fake_method)
    monkeypatch.setattr(runner, "build_merge_tag", lambda metadata, task_names: "fake_tag")

    result = runner.run_merge(
        adapter_specs=["task_a", "task_b"],
        method="fake",
        lambda_weight=0.6,
        params=None,
        merge_mode="common",
        output=None,
        save_merged=False,
        show_progress=False,
    )

    assert result.task_names == ["task_a", "task_b"]
    assert result.merge_tag == "fake_tag"
    assert result.params["optimizer"]["type"] == "none"
