from __future__ import annotations

import json
from pathlib import Path

import pytest
from datasets import Dataset

from core.training.multitask_eval import (
    MultiTaskEvaluator,
    aggregate_interference_delta_arithmetic,
    aggregate_interference_delta_geometric,
)
from tests.helpers.core import FakeEvalSetup


def test_interference_delta_aggregates_handle_empty_zero_and_negative() -> None:
    assert aggregate_interference_delta_geometric([]) == float("-inf")
    assert aggregate_interference_delta_geometric([1.0, 4.0]) == pytest.approx(2.0)
    assert aggregate_interference_delta_geometric([0.0, 4.0]) == 0.0
    assert aggregate_interference_delta_geometric([-1.0]) == float("-inf")
    assert aggregate_interference_delta_arithmetic([]) == float("-inf")
    assert aggregate_interference_delta_arithmetic([1.0, 3.0]) == 2.0


def test_multitask_evaluator_subset_selection_supports_stratified_and_overrides(tmp_path: Path) -> None:
    setup = FakeEvalSetup(
        Dataset.from_list(
            [
                {"label": 0, "value": "a"},
                {"label": 0, "value": "b"},
                {"label": 1, "value": "c"},
                {"label": 1, "value": "d"},
            ]
        )
    )
    evaluator = MultiTaskEvaluator(
        tasks=["emotion"],
        eval_setups={"emotion": setup},
        task_generation_kwargs={"emotion": {}},
        split="validation",
        batch_size=2,
        compute_missing_interference_baselines=False,
        metrics_dir=tmp_path,
        eval_subset={"max_samples": 2, "stratified": True, "label_column": "label", "enabled": True},
        auto_plot=False,
    )
    assert len(evaluator.eval_setups["emotion"].dataset) == 2
    assert setup.applied_indices is not None


def test_multitask_evaluator_writes_history_and_added_task_metric(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run_evaluation(model, setup, batch_size, generation_kwargs, output_dir, processor, **kwargs):  # noqa: ANN001
        del model, setup, batch_size, output_dir, processor, kwargs
        return {"wer": 0.2} if generation_kwargs["task"] == "asr" else {"accuracy": 0.75}

    def fake_add_delta(task, metrics, split, show_summary, eval_subset=None):  # noqa: ANN001
        del split, show_summary, eval_subset
        metrics["interference_delta"] = 0.3 if task == "asr" else 0.8

    monkeypatch.setattr("core.training.multitask_eval.run_evaluation", fake_run_evaluation)
    monkeypatch.setattr("core.training.multitask_eval.maybe_add_interference_delta", fake_add_delta)
    monkeypatch.setattr("core.training.multitask_eval.maybe_compute_interference_baselines", lambda **kwargs: None)

    evaluator = MultiTaskEvaluator(
        tasks=["asr", "emotion"],
        eval_setups={
            "asr": FakeEvalSetup(Dataset.from_list([{"x": 1}, {"x": 2}])),
            "emotion": FakeEvalSetup(Dataset.from_list([{"x": 1}, {"x": 2}])),
        },
        task_generation_kwargs={"asr": {"task": "asr"}, "emotion": {"task": "emotion"}},
        split="validation",
        batch_size=2,
        compute_missing_interference_baselines=True,
        metrics_dir=tmp_path,
        selection_criterion="arithmetic_mean_interference_delta",
        selection_mode="added_task_metric",
        selected_primary_tasks=["asr"],
        auto_plot=False,
    )
    result = evaluator.evaluate(model=None, processor=None, global_step=7)
    assert result.metrics["eval_added_tasks_primary_oriented_mean"] == -0.2
    assert result.aggregate_delta == -0.2
    assert json.loads((tmp_path / "mtl_eval_history.jsonl").read_text(encoding="utf-8").splitlines()[0])["step"] == 7


def test_multitask_evaluator_validates_selection_knobs(tmp_path: Path) -> None:
    kwargs = dict(
        tasks=["asr"],
        eval_setups={"asr": FakeEvalSetup(Dataset.from_list([{"x": 1}]))},
        task_generation_kwargs={"asr": {}},
        split="validation",
        batch_size=1,
        compute_missing_interference_baselines=False,
        metrics_dir=tmp_path,
        auto_plot=False,
    )
    with pytest.raises(ValueError, match="selection_criterion"):
        MultiTaskEvaluator(**kwargs, selection_criterion="bad")
    with pytest.raises(ValueError, match="selection_mode"):
        MultiTaskEvaluator(**kwargs, selection_mode="bad")
