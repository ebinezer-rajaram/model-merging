from __future__ import annotations

from pathlib import Path

import pytest

from merging.config.specs import MergeSpec, OptimizerSpec
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
    list_merge_methods,
    normalize_params,
    register_merge_method,
)
from merging.optimizers.registry import OptimizerContext, apply_optimizer_overrides, list_optimizers, optimize_lambda_policy


def test_builtin_merge_method_registry_surface_is_explicit() -> None:
    expected = {
        "dare",
        "task_vector",
        "ties",
        "uniform",
        "uniform_delta",
        "uniform_scalar_delta",
        "weighted",
        "weighted_delta",
        "weighted_delta_n",
    }
    assert expected <= set(list_merge_methods())
    with pytest.raises(ValueError, match="Unknown merge method"):
        get_merge_method("missing")


def test_merge_method_validation_metadata_and_duplicate_registration() -> None:
    method = MergeMethod(
        name="unit_fake_method_config_registry",
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
    resolved = get_merge_method("unit_fake_method_config_registry")
    with pytest.raises(ValueError, match="requires params"):
        resolved.validate(2, {})
    with pytest.raises(ValueError, match="at most"):
        resolved.validate(3, {"lambda": 0.5})
    assert normalize_params(resolved, {"lambda": 0.7}) == {"temperature": 1.0, "lambda": 0.7}
    assert build_merge_metadata(
        method="m",
        merge_mode="common",
        num_adapters=2,
        source_metadata=[],
        num_parameters=1,
        params={"lambda": 0.2},
    )["lambda"] == 0.2
    with pytest.raises(ValueError, match="already registered"):
        register_merge_method(method)


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
    assert merge_spec_to_params(config)["optimizer"]["type"] == "bayes"

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


def test_optimizer_registry_noop_and_explicit_best() -> None:
    assert {"none", "bayes", "adamerging", "supermerge", "regret_smoothmax", "gradient"} <= set(
        list_optimizers()
    )
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
