from __future__ import annotations

from pathlib import Path

import pytest
import torch

from merging.methods.dare import _validate_dare_params, fuse_deltas_uniform, merge_dare, sparsify_deltas
from merging.methods.ties import (
    _disjoint_mean,
    _elect_sign,
    _global_topk_threshold,
    _trim_task_vector_global,
    _validate_ties_params,
    merge_ties,
)
from merging.methods.task_vector import merge_uniform_via_task_vectors
from merging.methods.uniform import merge_adapters_uniform, merge_uniform
from merging.methods.uniform_scalar_delta import merge_task_vectors_uniform_scalar
from merging.methods.weighted import merge_adapters_weighted, merge_task_vectors_weighted, merge_weighted
from merging.methods.weighted_delta_n import merge_task_vectors_weighted_n


def test_uniform_and_weighted_adapters_cover_common_strict_and_edges() -> None:
    w1 = {"a": torch.tensor([1.0, 3.0]), "only1": torch.tensor([9.0])}
    w2 = {"a": torch.tensor([3.0, 5.0]), "only2": torch.tensor([7.0])}

    assert torch.allclose(merge_adapters_uniform([w1, w2])["a"], torch.tensor([2.0, 4.0]))
    assert merge_adapters_weighted(w1, w2, 0.0) == w2
    assert torch.allclose(
        merge_adapters_weighted(w1, w2, 0.25, lambda_resolver=lambda key: 0.75)["a"],
        torch.tensor([1.5, 3.5]),
    )

    with pytest.raises(ValueError, match="No adapters"):
        merge_adapters_uniform([])
    with pytest.raises(ValueError, match="different parameters"):
        merge_adapters_uniform([w1, w2], merge_mode="strict")
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        merge_adapters_weighted(w1, w2, 1.5)
    with pytest.raises(ValueError, match="lambda for key"):
        merge_adapters_weighted(w1, w2, 0.5, lambda_resolver=lambda key: -0.1)


def test_high_level_uniform_and_weighted_merge_save_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import merging.methods.uniform as uniform_mod
    import merging.methods.weighted as weighted_mod

    saved: list[dict] = []
    weights = {
        "a": {"x": torch.tensor([1.0])},
        "b": {"x": torch.tensor([3.0])},
    }

    monkeypatch.setattr(uniform_mod, "load_adapter_weights", lambda path: weights[path.name])
    monkeypatch.setattr(uniform_mod, "save_merged_adapter", lambda **kwargs: saved.append(kwargs))
    monkeypatch.setattr(weighted_mod, "load_adapter_weights", lambda path: weights[path.name])
    monkeypatch.setattr(weighted_mod, "save_merged_adapter", lambda **kwargs: saved.append(kwargs))

    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()

    assert merge_uniform([a, b], tmp_path / "uniform", show_progress=False) == tmp_path / "uniform"
    assert torch.allclose(saved[-1]["weights"]["x"], torch.tensor([2.0]))
    assert saved[-1]["metadata"]["merge_method"] == "uniform"

    metadata = [{"task": "a"}, {"task": "b"}]
    assert merge_weighted(a, b, 0.25, tmp_path / "weighted", source_metadata=metadata, show_progress=False) == tmp_path / "weighted"
    assert torch.allclose(saved[-1]["weights"]["x"], torch.tensor([2.5]))
    assert saved[-1]["metadata"]["source_adapters"][0]["weight"] == 0.25


def test_task_vector_methods_validate_shapes_modes_and_scaling() -> None:
    tv1 = {"a": torch.tensor([1.0, 2.0]), "bad": torch.ones(2)}
    tv2 = {"a": torch.tensor([3.0, 4.0]), "bad": torch.ones(3)}

    assert torch.allclose(merge_task_vectors_weighted(tv1, tv2, 0.25)["a"], torch.tensor([2.5, 3.5]))
    assert torch.allclose(merge_task_vectors_uniform_scalar([tv1, tv2], scale=0.5)["a"], torch.tensor([2.0, 3.0]))

    with pytest.raises(ValueError, match="at least 2"):
        merge_task_vectors_uniform_scalar([tv1])
    with pytest.raises(ValueError, match="Unsupported merge_mode"):
        merge_task_vectors_uniform_scalar([tv1, tv2], merge_mode="bad")
    with pytest.raises(ValueError, match="Shape mismatch"):
        merge_task_vectors_uniform_scalar([tv1, tv2], merge_mode="strict")
    with pytest.raises(ValueError, match="Shape mismatch"):
        merge_task_vectors_weighted(tv1, tv2, 0.5, merge_mode="strict")


def test_task_vector_save_path_merges_common_keys_and_reports_strict_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import merging.methods.task_vector as task_vector_mod

    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    vectors = {
        a: {"shared": torch.tensor([1.0, 3.0]), "a_only": torch.ones(1), "bad": torch.ones(2)},
        b: {"shared": torch.tensor([3.0, 5.0]), "b_only": torch.ones(1), "bad": torch.ones(3)},
    }
    saved: list[dict] = []
    monkeypatch.setattr(task_vector_mod, "extract_task_vector_from_lora", lambda path: vectors[path])
    monkeypatch.setattr(task_vector_mod, "save_merged_adapter", lambda **kwargs: saved.append(kwargs))

    merge_uniform_via_task_vectors([a, b], tmp_path / "merged", show_progress=True)

    assert torch.allclose(saved[0]["weights"]["shared"], torch.tensor([2.0, 4.0]))
    assert "bad" not in saved[0]["weights"]
    assert saved[0]["metadata"]["merge_method"] == "task_vector"
    assert "Warning" in capsys.readouterr().out

    with pytest.raises(ValueError, match="different parameters"):
        merge_uniform_via_task_vectors([a, b], tmp_path / "strict", merge_mode="strict", show_progress=False)


def test_builtin_method_entrypoints_apply_specs_and_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import merging.engine.builtin_methods as builtin

    a = tmp_path / "a"
    b = tmp_path / "b"
    c = tmp_path / "c"
    for path in (a, b, c):
        path.mkdir()

    weights = {
        a: {"x": torch.tensor([1.0]), "layer": torch.tensor([10.0])},
        b: {"x": torch.tensor([3.0]), "layer": torch.tensor([20.0])},
        c: {"x": torch.tensor([5.0]), "layer": torch.tensor([30.0])},
    }
    task_vectors = {
        a: {"x": torch.tensor([1.0]), "layer": torch.tensor([10.0])},
        b: {"x": torch.tensor([3.0]), "layer": torch.tensor([20.0])},
        c: {"x": torch.tensor([5.0]), "layer": torch.tensor([30.0])},
    }
    saved: list[dict] = []

    monkeypatch.setattr(builtin, "load_adapter_weights", lambda path: weights[path])
    monkeypatch.setattr(builtin, "extract_task_vector_from_lora", lambda path: task_vectors[path])
    monkeypatch.setattr(builtin, "compute_delta_from_lora_weights", lambda merged_weights, reference_path: dict(merged_weights))
    monkeypatch.setattr(builtin, "merge_uniform_via_task_vectors", lambda **kwargs: saved.append(kwargs))

    source_metadata = [{"task": "a"}, {"task": "b"}, {"task": "c"}]
    uniform = builtin._uniform_in_memory(
        adapter_paths=[a, b],
        source_metadata=source_metadata[:2],
        merge_mode="common",
        params=None,
    )
    assert torch.allclose(uniform.merged_delta["x"], torch.tensor([2.0]))
    assert uniform.metadata["merge_method"] == "uniform"

    weighted = builtin._weighted_in_memory(
        adapter_paths=[a, b],
        source_metadata=source_metadata[:2],
        merge_mode="common",
        params={"lambda": 0.25},
    )
    assert torch.allclose(weighted.merged_delta["x"], torch.tensor([2.5]))
    assert weighted.metadata["source_adapters"][0]["weight"] == 0.25

    task_vector = builtin._task_vector_in_memory(
        adapter_paths=[a, b],
        source_metadata=source_metadata[:2],
        merge_mode="common",
        params=None,
    )
    assert torch.allclose(task_vector.merged_delta["x"], torch.tensor([2.0]))

    uniform_delta = builtin._uniform_delta_in_memory(
        adapter_paths=[a, b],
        source_metadata=source_metadata[:2],
        merge_mode="common",
        params=None,
    )
    assert torch.allclose(uniform_delta.merged_delta["x"], torch.tensor([2.0]))

    weighted_delta = builtin._weighted_delta_in_memory(
        adapter_paths=[a, b],
        source_metadata=source_metadata[:2],
        merge_mode="common",
        params={"lambda": 0.75},
    )
    assert torch.allclose(weighted_delta.merged_delta["x"], torch.tensor([1.5]))

    weighted_delta_n = builtin._weighted_delta_n_in_memory(
        adapter_paths=[a, b, c],
        source_metadata=source_metadata,
        merge_mode="common",
        params={"task_coefficients": [0.2, 0.3, 0.5], "normalize_coefficients": False},
    )
    assert torch.allclose(weighted_delta_n.merged_delta["x"], torch.tensor([3.6]))
    assert weighted_delta_n.metadata["coefficient_policy"]["type"] == "task_coefficients"

    scalar = builtin._uniform_scalar_delta_in_memory(
        adapter_paths=[a, b],
        source_metadata=source_metadata[:2],
        merge_mode="common",
        params={"scale": 0.5},
    )
    assert torch.allclose(scalar.merged_delta["x"], torch.tensor([2.0]))

    assert builtin._task_vector_save(
        adapter_paths=[a, b],
        output_path=tmp_path / "saved",
        merge_mode="common",
        show_progress=False,
        params={"transforms": [{"name": "identity"}]},
    ) == tmp_path / "saved"
    assert saved[0]["transforms"][0].name == "identity"

    with pytest.raises(ValueError, match="requires lambda_weight"):
        builtin._weighted_in_memory(
            adapter_paths=[a, b],
            source_metadata=source_metadata[:2],
            merge_mode="common",
            params={},
        )


def test_weighted_delta_n_coefficients_layer_overrides_and_validation() -> None:
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

    with pytest.raises(ValueError, match="at least 2"):
        merge_task_vectors_weighted_n([{"x": torch.ones(1)}])
    with pytest.raises(ValueError, match="Unsupported merge_mode"):
        merge_task_vectors_weighted_n([{"x": torch.ones(1)}, {"x": torch.ones(1)}], merge_mode="bad")
    with pytest.raises(ValueError, match="length=2"):
        merge_task_vectors_weighted_n([{"x": torch.ones(1)}, {"x": torch.ones(1)}], task_coefficients=[1.0])
    with pytest.raises(ValueError, match="Coefficient sum"):
        merge_task_vectors_weighted_n(
            [{"x": torch.ones(1)}, {"x": torch.ones(1)}],
            task_coefficients=[0.0, 0.0],
            normalize_coefficients=True,
        )
    with pytest.raises(ValueError, match="different parameters"):
        merge_task_vectors_weighted_n([{"x": torch.ones(1)}, {"y": torch.ones(1)}], merge_mode="strict")


def test_dare_sparsify_fuse_and_entrypoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import merging.methods.dare as dare_mod

    vectors = [
        {"a": torch.ones(4), "b": torch.ones(2)},
        {"a": torch.full((4,), 3.0), "b": torch.full((2,), 5.0)},
    ]
    sparse1, stats1 = sparsify_deltas(vectors, drop_rate=0.5, seed=13)
    sparse2, stats2 = sparsify_deltas(vectors, drop_rate=0.5, seed=13)
    assert stats1 == stats2
    assert all(torch.equal(sparse1[i][key], sparse2[i][key]) for i in range(2) for key in sparse1[i])

    fused, fuse_stats = fuse_deltas_uniform(vectors, merge_mode="common")
    assert torch.allclose(fused["a"], torch.full((4,), 2.0))
    assert fuse_stats["merged_parameter_count"] == 2

    with pytest.raises(ValueError, match="drop_rate"):
        _validate_dare_params({"drop_rate": 1.0})
    with pytest.raises(ValueError, match="seed"):
        _validate_dare_params({"seed": True})

    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    task_vectors = {
        a: {"x": torch.tensor([1.0, 3.0]), "skip": torch.ones(2)},
        b: {"x": torch.tensor([5.0, 7.0]), "skip": torch.ones(3)},
    }
    monkeypatch.setattr(dare_mod, "extract_task_vector_from_lora", lambda path: task_vectors[path])
    result = merge_dare(
        adapter_paths=[a, b],
        source_metadata=[{"task": "a"}, {"task": "b"}],
        merge_mode="common",
        params={"drop_rate": 0.0, "seed": 1},
    )
    assert torch.allclose(result.merged_delta["x"], torch.tensor([3.0, 5.0]))
    assert result.metadata["dare_stats"]["skipped_shape_mismatch_count"] == 1

    with pytest.raises(ValueError, match="at least 2"):
        merge_dare(adapter_paths=[a], source_metadata=[], merge_mode="common", params={})


def test_ties_core_steps_validation_and_entrypoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import merging.methods.ties as ties_mod

    trimmed = _trim_task_vector_global({"a": torch.tensor([1.0, -4.0, 2.0])}, 50.0)
    assert torch.equal(trimmed["a"], torch.tensor([0.0, -4.0, 2.0]))
    assert _global_topk_threshold(torch.tensor([1.0, 3.0, 2.0]), 50.0).item() == pytest.approx(2.0)

    elected = _elect_sign(
        [
            {"a": torch.tensor([1.0, -2.0, 0.0])},
            {"a": torch.tensor([2.0, 3.0, 0.0])},
        ]
    )
    assert torch.equal(elected["a"], torch.tensor([1.0, 1.0, 0.0]))
    merged = _disjoint_mean(
        [
            {"a": torch.tensor([1.0, -2.0, 0.0])},
            {"a": torch.tensor([3.0, 4.0, 0.0])},
        ],
        elected,
    )
    assert torch.equal(merged["a"], torch.tensor([2.0, 4.0, 0.0]))

    with pytest.raises(ValueError, match="'k'"):
        _validate_ties_params({"k": 101})
    with pytest.raises(ValueError, match="'lambda'"):
        _validate_ties_params({"lambda": False})

    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    task_vectors = {
        a: {"x": torch.tensor([1.0, -2.0, 3.0]), "bad": torch.ones(2)},
        b: {"x": torch.tensor([2.0, 4.0, -6.0]), "bad": torch.ones(3)},
    }
    monkeypatch.setattr(ties_mod, "extract_task_vector_from_lora", lambda path: task_vectors[path])
    result = merge_ties(
        adapter_paths=[a, b],
        source_metadata=[{"task": "a"}, {"task": "b"}],
        merge_mode="common",
        params={"k": 100.0, "lambda": 0.5},
    )
    assert set(result.merged_delta) == {"x"}
    assert result.metadata["ties_stats"]["skipped_shape_mismatch_count"] == 1

    with pytest.raises(ValueError, match="at least 2"):
        merge_ties(adapter_paths=[a], source_metadata=[], merge_mode="common", params={})
