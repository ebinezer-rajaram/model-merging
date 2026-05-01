from __future__ import annotations

import torch

from core.evaluation import eval_utils
from tests.helpers.core import CoreDummyProcessor, TinyParameterModel


def test_eval_task_registry_accepts_custom_builder_and_rejects_duplicates(monkeypatch) -> None:
    def builder(**kwargs):
        return kwargs

    original_registry = dict(eval_utils._TASK_REGISTRY)
    monkeypatch.setattr(eval_utils, "_TASK_REGISTRY", dict(original_registry))
    eval_utils.register_eval_task("custom_core_task", builder, overwrite=True)
    assert "custom_core_task" in eval_utils.get_registered_eval_tasks()
    try:
        eval_utils.register_eval_task("custom_core_task", builder)
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("Expected duplicate registration failure")
    try:
        eval_utils.register_eval_task("", builder)
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("Expected empty task registration failure")


def test_special_tokens_and_delta_application_on_tiny_model() -> None:
    processor = CoreDummyProcessor()
    model = TinyParameterModel()
    eval_utils._configure_special_tokens(model, processor)
    assert model.config.pad_token_id == processor.tokenizer.pad_token_id
    assert model.generation_config.eos_token_id == processor.tokenizer.eos_token_id

    before = model.model.linear.weight.detach().clone()
    delta = torch.ones_like(before)
    eval_utils._apply_delta_weights(model, {"model.linear": delta})
    assert torch.equal(model.model.linear.weight, before + delta)

    eval_utils._apply_delta_weights(model, {"missing": delta, "model.linear": torch.ones(1, 1)})
    assert eval_utils._resolve_delta_param_name_local("model.linear", list(model.state_dict().keys())) == "model.linear.weight"


def test_eval_subset_tags_are_stable_for_equal_payloads() -> None:
    assert eval_utils.compute_eval_subset_tag({"b": 2, "a": 1}) == eval_utils.compute_eval_subset_tag({"a": 1, "b": 2})
    assert eval_utils.compute_task_eval_subset_tag(max_samples=4, shuffle=True, seed=1, stratified=False, label_column=None).startswith("subset_")
