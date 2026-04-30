from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.training.multitask_eval import MultiTaskEvaluator


@dataclass
class _DummySetup:
    dataset: list
    data_collator: object = None
    compute_metrics: object = None


def test_added_task_metric_selection_mode(monkeypatch, tmp_path: Path) -> None:
    def _fake_run_evaluation(model, setup, batch_size, generation_kwargs, output_dir, processor, **kwargs):  # noqa: ANN001
        del kwargs
        task = generation_kwargs["__task"]
        if task == "asr":
            return {"wer": 0.2}
        return {"accuracy": 0.75}

    def _fake_add_interference_delta(task, metrics, split, show_summary, eval_subset=None):  # noqa: ANN001
        metrics["interference_delta"] = 0.3 if task == "asr" else 0.8

    monkeypatch.setattr("core.training.multitask_eval.run_evaluation", _fake_run_evaluation)
    monkeypatch.setattr("core.training.multitask_eval.maybe_add_interference_delta", _fake_add_interference_delta)
    monkeypatch.setattr("core.training.multitask_eval.maybe_compute_interference_baselines", lambda **kwargs: None)

    evaluator = MultiTaskEvaluator(
        tasks=["asr", "emotion"],
        eval_setups={"asr": _DummySetup(dataset=[1, 2]), "emotion": _DummySetup(dataset=[1, 2, 3])},
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
    # ASR is lower-is-better, so oriented score is negative.
    assert result.metrics["eval_added_tasks_primary_oriented_mean"] == -0.2
    # In added_task_metric mode aggregate_delta tracks the added-task primary selection value.
    assert result.aggregate_delta == -0.2
