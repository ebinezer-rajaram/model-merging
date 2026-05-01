from __future__ import annotations

import pytest
import torch

from merging.config.specs import LambdaPolicySpec
from merging.policies.lambda_policy import PerLayerLambdaPolicy, ScalarLambdaPolicy, build_lambda_policy, extract_layer_index
from merging.transforms.registry import apply_transforms, get_transform, list_transforms, register_transform


def test_lambda_policies_extract_layers_describe_and_validate() -> None:
    key = "base_model.model.model.layers.3.self_attn.q_proj.weight"
    assert extract_layer_index(key) == 3
    assert extract_layer_index("bias") is None
    assert ScalarLambdaPolicy(0.25).describe() == {"type": "scalar", "value": 0.25}
    assert PerLayerLambdaPolicy(0.5, {3: 0.8}).lambda_for_key(key) == pytest.approx(0.8)
    assert build_lambda_policy(LambdaPolicySpec(type="per_layer", default=0.4, overrides={3: 0.9})).lambda_for_key(key) == pytest.approx(0.9)

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


def test_transform_registry_surface_and_identity_layer_norm_ties() -> None:
    expected = {"identity", "ties", "ties_scaffold", "layer_l2_normalize"}
    assert expected <= set(list_transforms())
    with pytest.raises(ValueError, match="Unknown transform"):
        get_transform("missing")

    key = "base_model.model.model.layers.3.self_attn.q_proj.weight"
    weights = {key: torch.tensor([3.0, 4.0]), "bias": torch.tensor([10.0])}
    assert torch.equal(apply_transforms(weights, [type("Spec", (), {"name": "identity", "params": {}})()])[key], weights[key])

    normalized = apply_transforms(
        weights,
        [type("Spec", (), {"name": "layer_l2_normalize", "params": {"target_norm": 10.0, "include_non_layer_keys": False}})()],
    )
    assert torch.allclose(normalized[key], torch.tensor([6.0, 8.0]))
    assert torch.equal(normalized["bias"], weights["bias"])

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


def test_transform_param_validation_and_duplicate_registration() -> None:
    weights = {"a": torch.tensor([1.0])}
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
        register_transform(" ", lambda value, params: value)
    with pytest.raises(ValueError, match="already registered"):
        register_transform("identity", lambda value, params: value)
