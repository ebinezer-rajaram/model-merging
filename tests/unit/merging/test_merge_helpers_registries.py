from __future__ import annotations

from pathlib import Path

import pytest
import torch

from merging.config.specs import LambdaPolicySpec
from merging.config.unified import (
    load_merge_config,
    merge_config_from_legacy_args,
    merge_spec_to_params,
    normalize_merge_config,
)
from merging.engine.registry import (
    MergeMethod,
    MergeOutput,
    build_merge_metadata,
    get_merge_method,
    normalize_params,
    register_merge_method,
)
from merging.methods.dare import _validate_dare_params, fuse_deltas_uniform, merge_dare, sparsify_deltas
from merging.methods.ties import (
    _disjoint_mean,
    _elect_sign,
    _global_topk_threshold,
    _trim_task_vector_global,
    _validate_ties_params,
    merge_ties,
)
from merging.methods.uniform import merge_adapters_uniform, merge_uniform
from merging.methods.uniform_scalar_delta import merge_task_vectors_uniform_scalar
from merging.methods.weighted import merge_adapters_weighted, merge_task_vectors_weighted, merge_weighted
from merging.optimizers.registry import (
    OptimizerContext,
    apply_optimizer_overrides,
    list_optimizers,
    optimize_lambda_policy,
)
from merging.policies.lambda_policy import (
    PerLayerLambdaPolicy,
    ScalarLambdaPolicy,
    build_lambda_policy,
    extract_layer_index,
)
from merging.transforms.registry import apply_transforms, get_transform, list_transforms


def test_uniform_and_weighted_merges_cover_common_strict_and_edges() -> None:
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

    def fake_load(path: Path):
        return weights[path.name]

    def fake_save(**kwargs):
        saved.append(kwargs)

    monkeypatch.setattr(uniform_mod, "load_adapter_weights", fake_load)
    monkeypatch.setattr(uniform_mod, "save_merged_adapter", fake_save)
    monkeypatch.setattr(weighted_mod, "load_adapter_weights", fake_load)
    monkeypatch.setattr(weighted_mod, "save_merged_adapter", fake_save)

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


def test_task_vector_merge_methods_validate_shapes_and_modes() -> None:
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


def test_dare_sparsify_and_fuse_are_deterministic_and_validate() -> None:
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


def test_merge_dare_entrypoint_with_monkeypatched_task_vectors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import merging.methods.dare as dare_mod

    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    vectors = {
        a: {"x": torch.tensor([1.0, 3.0]), "skip": torch.ones(2)},
        b: {"x": torch.tensor([5.0, 7.0]), "skip": torch.ones(3)},
    }

    monkeypatch.setattr(dare_mod, "extract_task_vector_from_lora", lambda path: vectors[path])
    result = merge_dare(
        adapter_paths=[a, b],
        source_metadata=[{"task": "a"}, {"task": "b"}],
        merge_mode="common",
        params={"drop_rate": 0.0, "seed": 1},
    )

    assert torch.allclose(result.merged_delta["x"], torch.tensor([3.0, 5.0]))
    assert result.metadata["dare_stats"]["skipped_shape_mismatch_count"] == 1
    assert result.metadata["dare_stats"]["effective_sparsity"] == 0.0

    with pytest.raises(ValueError, match="at least 2"):
        merge_dare(adapter_paths=[a], source_metadata=[], merge_mode="common", params={})


def test_ties_core_steps_and_validation() -> None:
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


def test_merge_ties_entrypoint_with_monkeypatched_task_vectors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import merging.methods.ties as ties_mod

    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    vectors = {
        a: {"x": torch.tensor([1.0, -2.0, 3.0]), "bad": torch.ones(2)},
        b: {"x": torch.tensor([2.0, 4.0, -6.0]), "bad": torch.ones(3)},
    }

    monkeypatch.setattr(ties_mod, "extract_task_vector_from_lora", lambda path: vectors[path])
    result = merge_ties(
        adapter_paths=[a, b],
        source_metadata=[{"task": "a"}, {"task": "b"}],
        merge_mode="common",
        params={"k": 100.0, "lambda": 0.5},
    )

    assert set(result.merged_delta) == {"x"}
    assert result.metadata["ties_stats"]["skipped_shape_mismatch_count"] == 1
    assert result.metadata["ties_stats"]["merged_parameter_count"] == 1

    with pytest.raises(ValueError, match="at least 2"):
        merge_ties(adapter_paths=[a], source_metadata=[], merge_mode="common", params={})


def test_transforms_lambda_policies_and_registries() -> None:
    key = "base_model.model.model.layers.3.self_attn.q_proj.weight"
    assert extract_layer_index(key) == 3
    assert ScalarLambdaPolicy(0.25).describe() == {"type": "scalar", "value": 0.25}
    assert PerLayerLambdaPolicy(0.5, {3: 0.8}).lambda_for_key(key) == pytest.approx(0.8)
    assert build_lambda_policy(LambdaPolicySpec(type="per_layer", default=0.4, overrides={3: 0.9})).lambda_for_key(key) == pytest.approx(0.9)

    weights = {key: torch.tensor([3.0, 4.0]), "bias": torch.tensor([10.0])}
    normalized = apply_transforms(
        weights,
        [type("Spec", (), {"name": "layer_l2_normalize", "params": {"target_norm": 10.0, "include_non_layer_keys": False}})()],
    )
    assert torch.allclose(normalized[key], torch.tensor([6.0, 8.0]))
    assert torch.equal(normalized["bias"], weights["bias"])
    assert torch.equal(apply_transforms(weights, [type("Spec", (), {"name": "identity", "params": {}})()])[key], weights[key])

    ties_zero = apply_transforms(
        {"a": torch.tensor([1.0, -2.0])},
        [type("Spec", (), {"name": "ties", "params": {"k": 0, "lambda": 2.0}})()],
    )
    assert torch.equal(ties_zero["a"], torch.zeros(2))
    ties_all = apply_transforms(
        {"a": torch.tensor([1.0, -2.0])},
        [type("Spec", (), {"name": "ties", "params": {"k": 100, "lambda": 2.0}})()],
    )
    assert torch.equal(ties_all["a"], torch.tensor([2.0, -4.0]))
    ties_mid = apply_transforms(
        {"a": torch.tensor([1.0, -4.0]), "b": torch.tensor([3.0])},
        [type("Spec", (), {"name": "ties", "params": {"k": 50}})()],
    )
    assert torch.equal(ties_mid["a"], torch.tensor([0.0, -4.0]))
    assert torch.equal(ties_mid["b"], torch.tensor([3.0]))

    assert "ties" in list_transforms()
    with pytest.raises(ValueError, match="Unknown transform"):
        get_transform("missing")
    with pytest.raises(ValueError, match="'k'"):
        apply_transforms(weights, [type("Spec", (), {"name": "ties", "params": {"k": "bad"}})()])
    with pytest.raises(ValueError, match="'lambda'"):
        apply_transforms(weights, [type("Spec", (), {"name": "ties", "params": {"lambda": True}})()])
    with pytest.raises(ValueError, match="target_norm"):
        apply_transforms(weights, [type("Spec", (), {"name": "layer_l2_normalize", "params": {"target_norm": -1}})()])
    with pytest.raises(ValueError, match="eps"):
        apply_transforms(weights, [type("Spec", (), {"name": "layer_l2_normalize", "params": {"eps": 0}})()])
    with pytest.raises(ValueError, match="include_non_layer_keys"):
        apply_transforms(
            weights,
            [type("Spec", (), {"name": "layer_l2_normalize", "params": {"include_non_layer_keys": "yes"}})()],
        )
    with pytest.raises(ValueError, match="non-empty"):
        from merging.transforms.registry import register_transform

        register_transform(" ", lambda weights, params: weights)

    with pytest.raises(ValueError, match="Scalar lambda"):
        ScalarLambdaPolicy(1.1)
    with pytest.raises(ValueError, match="Layer indices"):
        PerLayerLambdaPolicy(0.5, {-1: 0.5})
    with pytest.raises(ValueError, match="Per-layer lambda"):
        PerLayerLambdaPolicy(0.5, {0: 2.0})
    with pytest.raises(ValueError, match="Missing lambda"):
        build_lambda_policy(None)
    with pytest.raises(ValueError, match="scalar lambda policy"):
        build_lambda_policy(LambdaPolicySpec(type="scalar"))
    with pytest.raises(ValueError, match="per_layer lambda policy"):
        build_lambda_policy(LambdaPolicySpec(type="per_layer"))
    with pytest.raises(ValueError, match="Unsupported lambda"):
        build_lambda_policy(LambdaPolicySpec(type="unknown"))

    method = MergeMethod(
        name="unit_fake_method",
        required_params=("lambda",),
        params_defaults={"temperature": 1.0},
        params_validator=lambda params: None,
        min_adapters=2,
        max_adapters=2,
        saveable=False,
        merge_in_memory=lambda **kwargs: MergeOutput({}, None, {}),
    )
    try:
        register_merge_method(method)
    except ValueError:
        pass
    resolved = get_merge_method("unit_fake_method")
    with pytest.raises(ValueError, match="requires params"):
        resolved.validate(2, {})
    assert normalize_params(resolved, {"lambda": 0.7}) == {"temperature": 1.0, "lambda": 0.7}
    assert build_merge_metadata(
        method="m",
        merge_mode="common",
        num_adapters=2,
        source_metadata=[],
        num_parameters=1,
        params={"lambda": 0.2},
    )["lambda"] == 0.2


def test_unified_merge_config_normalizes_legacy_and_validates(tmp_path: Path) -> None:
    payload = {
        "adapters": ["a", "b"],
        "method": "weighted",
        "merge_mode": "strict",
        "params": {"temperature": 1.0},
        "lambda": 0.4,
        "transforms": [{"name": "identity", "params": {}}],
        "lambda_policy": {"type": "per_layer", "default": 0.5, "overrides": {"2": 0.8}},
        "optimizer": {"type": "bayes", "params": {"best_params": {"lambda": 0.7}}},
        "grid": {"lambda": [0.1, 0.2]},
        "eval_tasks": ["asr"],
        "split": "validation",
        "eval_subset": {"max_samples": 10},
        "output_dir": "out",
        "post_sweep_eval": {"split": "test", "eval_tasks": ["asr"]},
    }

    with pytest.warns(DeprecationWarning):
        config = normalize_merge_config(payload)
    assert config.method_params["lambda"] == 0.4
    assert config.lambda_policy.default == 0.5
    assert config.lambda_policy.overrides == {2: 0.8}
    assert config.search == {"type": "grid", "grid": {"lambda": [0.1, 0.2]}}
    assert config.to_merge_spec().method == "weighted"
    params = merge_spec_to_params(config)
    assert params["lambda_policy"]["overrides"] == {2: 0.8}
    assert params["optimizer"]["type"] == "bayes"

    path = tmp_path / "merge.yaml"
    path.write_text("adapters: [a, b]\nmethod: uniform\nsearch:\n  type: bayes\n", encoding="utf-8")
    assert load_merge_config(path).search["type"] == "bayes"

    legacy = merge_config_from_legacy_args(
        adapters=["a", "b"],
        method="weighted",
        merge_mode="common",
        lambda_weight=0.3,
        params={"transforms": [{"name": "identity"}], "optimizer": {"type": "none"}},
    )
    assert merge_spec_to_params(legacy)["lambda"] == 0.3

    invalid_payloads = [
        ({"adapters": ["a"]}, "adapters and method"),
        ({"adapters": "a", "method": "uniform"}, "adapters must be a list"),
        ({"adapters": ["a"], "method": "uniform", "merge_mode": "bad"}, "merge_mode"),
        ({"adapters": ["a"], "method": "uniform", "transforms": "bad"}, "transforms"),
        ({"adapters": ["a"], "method": "uniform", "lambda_policy": {"type": "bad"}}, "Unsupported"),
        ({"adapters": ["a"], "method": "uniform", "optimizer": []}, "optimizer"),
        ({"adapters": ["a"], "method": "uniform", "search": {"type": "bad"}}, "search.type"),
        ({"adapters": ["a"], "method": "uniform", "eval_tasks": "asr"}, "eval_tasks"),
        ({"adapters": ["a"], "method": "uniform", "split": "dev"}, "split"),
        ({"adapters": ["a"], "method": "uniform", "post_sweep_eval": {"split": "dev"}}, "post_sweep_eval.split"),
        ({"adapters": ["a"], "method": "uniform", "unknown": True}, "Unknown config"),
    ]
    for bad_payload, message in invalid_payloads:
        with pytest.raises(ValueError, match=message):
            normalize_merge_config(bad_payload)


def test_optimizer_registry_noop_and_bayes_explicit_best() -> None:
    from merging.config.specs import MergeSpec, OptimizerSpec

    context = OptimizerContext(
        method="weighted",
        adapter_specs=["a", "b"],
        adapter_paths=[Path("a"), Path("b")],
        source_metadata=[],
        merge_mode="common",
        output_dir=None,
        method_params={},
        lambda_policy=None,
    )
    spec = MergeSpec(adapters=["a", "b"], method="weighted", optimizer=None)
    assert optimize_lambda_policy(spec, context).provenance["status"] == "noop"

    bayes_spec = MergeSpec(
        adapters=["a", "b"],
        method="weighted",
        optimizer=OptimizerSpec(type="bayes", params={"best_params": {"lambda": 0.65}}),
    )
    result = optimize_lambda_policy(bayes_spec, context)
    assert result.method_params_overrides == {"lambda": 0.65}
    assert apply_optimizer_overrides({"lambda": 0.5}, result) == {"lambda": 0.65}
    assert "bayes" in list_optimizers()


def test_bayes_optimizer_reuses_compatible_sweep_files(tmp_path: Path) -> None:
    from merging.config.specs import MergeSpec, OptimizerSpec

    sweep = {
        "method": "weighted",
        "adapters": ["task_b", "task_a"],
        "merge_mode": "common",
        "runs": [
            {"params": {"lambda": 0.2}, "score": 1.0},
            {"params": {"lambda": 0.8}, "score": 2.0},
        ],
    }
    sweep_path = tmp_path / "sweep_001.json"
    sweep_path.write_text(__import__("json").dumps(sweep), encoding="utf-8")

    context = OptimizerContext(
        method="weighted",
        adapter_specs=["a", "b"],
        adapter_paths=[Path("a"), Path("b")],
        source_metadata=[{"task": "task_a"}, {"task": "task_b"}],
        merge_mode="common",
        output_dir=None,
        method_params={},
        lambda_policy=None,
    )
    spec = MergeSpec(
        adapters=["a", "b"],
        method="weighted",
        optimizer=OptimizerSpec(type="bayes", params={"sweep_paths": str(tmp_path)}),
    )
    result = optimize_lambda_policy(spec, context)
    assert result.method_params_overrides == {"lambda": 0.8}
    assert result.provenance["status"] == "applied_from_sweep"

    bad_spec = MergeSpec(
        adapters=["a", "b"],
        method="weighted",
        optimizer=OptimizerSpec(type="bayes", params={"sweep_paths": [123]}),
    )
    with pytest.raises(ValueError, match="sweep_paths"):
        optimize_lambda_policy(bad_spec, context)
