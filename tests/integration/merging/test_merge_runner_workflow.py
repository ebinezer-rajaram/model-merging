from __future__ import annotations

from pathlib import Path

import pytest
import torch

from merging.engine import runner
from merging.engine.registry import MergeOutput
from tests.helpers.merging import fake_merge_method


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

    monkeypatch.setattr(runner, "resolve_adapter_specs", fake_resolve)
    monkeypatch.setattr(runner, "get_merge_method", lambda name: fake_merge_method(name="fake", required_params=("lambda",), merge_in_memory=fake_merge_in_memory))
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


def test_run_merge_applies_optimizer_overrides_and_save_fn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter_a = tmp_path / "a"
    adapter_b = tmp_path / "b"
    adapter_a.mkdir()
    adapter_b.mkdir()
    save_calls: list[dict] = []

    def fake_resolve(specs):
        return [(adapter_a, {"task": "a"}), (adapter_b, {"task": "b"})]

    def fake_merge_in_memory(**kwargs):
        assert kwargs["params"]["lambda"] == 0.9
        return MergeOutput(
            merged_delta={"x": torch.ones(1)},
            merged_weights={"x": torch.ones(1)},
            metadata={"merge_method": "fake_saveable", "source_adapters": kwargs["source_metadata"]},
        )

    method = fake_merge_method(
        name="fake_saveable",
        saveable=True,
        merge_in_memory=fake_merge_in_memory,
        save_fn=lambda **kwargs: save_calls.append(kwargs),
    )
    monkeypatch.setattr(runner, "resolve_adapter_specs", fake_resolve)
    monkeypatch.setattr(runner, "get_merge_method", lambda name: method)
    monkeypatch.setattr(runner, "build_merge_tag", lambda metadata, task_names: "tag")

    from merging.config.specs import MergeSpec, OptimizerSpec

    spec = MergeSpec(
        adapters=["a", "b"],
        method="fake_saveable",
        optimizer=OptimizerSpec(type="bayes", params={"best_params": {"lambda": 0.9}}),
    )
    result = runner.run_merge(
        adapter_specs=[],
        method="ignored",
        lambda_weight=None,
        params=None,
        merge_mode="common",
        output=str(tmp_path / "merged"),
        save_merged=True,
        show_progress=False,
        merge_spec=spec,
    )

    assert result.output_path is not None
    assert save_calls and save_calls[0]["params"]["lambda"] == 0.9


def test_run_merge_rejects_saving_non_saveable_method(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter_a = tmp_path / "a"
    adapter_b = tmp_path / "b"
    adapter_a.mkdir()
    adapter_b.mkdir()

    monkeypatch.setattr(runner, "resolve_adapter_specs", lambda specs: [(adapter_a, {"task": "a"}), (adapter_b, {"task": "b"})])
    monkeypatch.setattr(runner, "get_merge_method", lambda name: fake_merge_method(name="fake"))

    with pytest.raises(ValueError, match="cannot be saved"):
        runner.run_merge(
            adapter_specs=["a", "b"],
            method="fake",
            lambda_weight=None,
            params=None,
            merge_mode="common",
            output=str(tmp_path / "out"),
            save_merged=True,
            show_progress=False,
        )
