from __future__ import annotations

from pathlib import Path

import pytest
from datasets import Dataset

from core.training.multitask_eval import MultiTaskEvaluator
from core.training.train_multitask import MultiTaskCollator
from tests.helpers.core import FakeEvalSetup


def test_multitask_collator_routes_homogeneous_batches() -> None:
    collator = MultiTaskCollator({"asr": lambda rows: {"count": len(rows), "rows": rows}})
    assert collator([{"__task_name": "asr", "x": 1}, {"__task_name": "asr", "x": 2}])["rows"] == [{"x": 1}, {"x": 2}]
    with pytest.raises(ValueError, match="empty"):
        collator([])
    with pytest.raises(ValueError, match="homogeneous"):
        collator([{"__task_name": "asr"}, {"__task_name": "emotion"}])
    with pytest.raises(KeyError, match="No collator"):
        collator([{"__task_name": "missing"}])


def test_added_task_metric_selection_mode_workflow(monkeypatch, tmp_path: Path) -> None:
    def fake_run_evaluation(model, setup, batch_size, generation_kwargs, output_dir, processor, **kwargs):  # noqa: ANN001
        del model, setup, batch_size, output_dir, processor, kwargs
        task = generation_kwargs["__task"]
        if task == "asr":
            return {"wer": 0.2}
        return {"accuracy": 0.75}

    def fake_add_interference_delta(task, metrics, split, show_summary, eval_subset=None):  # noqa: ANN001
        del split, show_summary, eval_subset
        metrics["interference_delta"] = 0.3 if task == "asr" else 0.8

    monkeypatch.setattr("core.training.multitask_eval.run_evaluation", fake_run_evaluation)
    monkeypatch.setattr("core.training.multitask_eval.maybe_add_interference_delta", fake_add_interference_delta)
    monkeypatch.setattr("core.training.multitask_eval.maybe_compute_interference_baselines", lambda **kwargs: None)

    evaluator = MultiTaskEvaluator(
        tasks=["asr", "emotion"],
        eval_setups={"asr": FakeEvalSetup(Dataset.from_list([{"x": 1}])), "emotion": FakeEvalSetup(Dataset.from_list([{"x": 1}]))},
        task_generation_kwargs={"asr": {"__task": "asr"}, "emotion": {"__task": "emotion"}},
        split="validation",
        batch_size=2,
        compute_missing_interference_baselines=False,
        metrics_dir=tmp_path / "metrics",
        selection_criterion="arithmetic_mean_interference_delta",
        selection_mode="added_task_metric",
        selected_primary_tasks=["asr"],
        use_cache=False,
        eval_subset=None,
        wandb_project=None,
        auto_plot=False,
    )
    result = evaluator.evaluate(model=None, processor=None, global_step=1)
    assert result.metrics["eval_added_tasks_primary_oriented_mean"] == -0.2
    assert result.aggregate_delta == -0.2
