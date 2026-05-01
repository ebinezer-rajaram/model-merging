from __future__ import annotations

from pathlib import Path

import pytest

from merging.config.unified import MergeConfig
from merging.evaluation import evaluate as merge_evaluate
from merging.evaluation import sweep as sweep_module


def test_grid_expansion_and_min_interference_scoring() -> None:
    assert sweep_module._expand_grid({"lambda": [0.1, 0.2], "scale": [1, 2]}) == [
        {"lambda": 0.1, "scale": 1},
        {"lambda": 0.1, "scale": 2},
        {"lambda": 0.2, "scale": 1},
        {"lambda": 0.2, "scale": 2},
    ]
    assert sweep_module._expand_grid({}) == [{}]

    score, details = sweep_module._score_min_interference(
        {"a": {"interference_delta": 0.9}, "b": {"interference_delta": 0.4}},
        constraint_nonnegative=True,
    )
    assert score == pytest.approx(0.4)
    assert details["mean_interference_delta"] == pytest.approx(0.65)

    score, details = sweep_module._score_min_interference({"a": {"accuracy": 1.0}}, constraint_nonnegative=True)
    assert score == float("-inf")
    assert details["reason"] == "missing_interference_delta"

    score, details = sweep_module._score_min_interference({"a": {"interference_delta": -0.1}}, constraint_nonnegative=True)
    assert score == float("-inf")
    assert details["min_interference_delta"] == -0.1


def test_run_sweep_grid_dispatches_evaluation_and_post_eval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = MergeConfig(
        adapters=["emotion", "intent"],
        method="weighted",
        search={"type": "grid", "grid": {"lambda": [0.1, 0.8]}},
        output_dir=tmp_path,
        split="validation",
        eval_tasks=["emotion"],
        save_merged=False,
        constraint_nonnegative=True,
        post_sweep_eval={"enabled": True, "split": "test"},
    )
    calls: list[dict] = []

    class _Method:
        params_defaults = {}

        def validate(self, num_adapters, params):
            return None

    def fake_evaluate_merged_adapter(**kwargs):
        calls.append(dict(kwargs))
        lam = kwargs["params"]["lambda"]
        return {"emotion": {"interference_delta": lam}}

    monkeypatch.setattr(sweep_module, "get_merge_method", lambda name: _Method())
    monkeypatch.setattr(sweep_module, "evaluate_merged_adapter", fake_evaluate_merged_adapter)
    monkeypatch.setattr(sweep_module, "_maybe_regen_plot", lambda summary_path: None)

    summary = sweep_module.run_sweep(config)

    assert len(summary["runs"]) == 2
    assert summary["best_index"] == 1
    assert summary["post_sweep_eval"]["score"] == pytest.approx(0.8)
    assert calls[-1]["split"] == "test"


def test_cleanup_runs_after_each_task_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    cleanup_calls = {"count": 0}
    monkeypatch.setattr(merge_evaluate, "_cleanup_cuda_memory", lambda show_summary=False: cleanup_calls.__setitem__("count", cleanup_calls["count"] + 1))

    results, task_status, failed_tasks = merge_evaluate._evaluate_tasks_with_retry(
        tasks_to_eval=["emotion", "intent"],
        split="test",
        show_summary=False,
        eval_subset=None,
        evaluate_once=lambda task: {"accuracy": 1.0, "task": task},
        allow_oom_retry=False,
        retry_adapter_path=None,
        batch_size=None,
        generate_confusion_matrix=False,
        compute_missing_interference_baselines=True,
    )

    assert failed_tasks == []
    assert sorted(results.keys()) == ["emotion", "intent"]
    assert cleanup_calls["count"] == 2
    assert task_status["emotion"] == {"retry_attempted": False, "retry_succeeded": False}


def test_oom_retry_and_non_oom_failure_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(merge_evaluate, "_cleanup_cuda_memory", lambda show_summary=False: None)
    retry_calls = {"count": 0}

    def fake_retry(**kwargs):
        retry_calls["count"] += 1
        return {"accuracy": 0.9}, None

    monkeypatch.setattr(merge_evaluate, "_retry_task_in_subprocess", fake_retry)
    results, task_status, failed_tasks = merge_evaluate._evaluate_tasks_with_retry(
        tasks_to_eval=["asr"],
        split="test",
        show_summary=False,
        eval_subset=None,
        evaluate_once=lambda task: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")),
        allow_oom_retry=True,
        retry_adapter_path=Path("/tmp/fake_run"),
        batch_size=4,
        generate_confusion_matrix=False,
        compute_missing_interference_baselines=True,
    )
    assert failed_tasks == []
    assert retry_calls["count"] == 1
    assert results["asr"]["accuracy"] == 0.9
    assert task_status["asr"] == {"retry_attempted": True, "retry_succeeded": True}

    results, task_status, failed_tasks = merge_evaluate._evaluate_tasks_with_retry(
        tasks_to_eval=["speech_qa"],
        split="test",
        show_summary=False,
        eval_subset=None,
        evaluate_once=lambda task: (_ for _ in ()).throw(ValueError("non-memory failure")),
        allow_oom_retry=True,
        retry_adapter_path=Path("/tmp/fake_run"),
        batch_size=None,
        generate_confusion_matrix=False,
        compute_missing_interference_baselines=True,
    )
    assert failed_tasks == ["speech_qa"]
    assert "error" in results["speech_qa"]
    assert task_status["speech_qa"] == {"retry_attempted": False, "retry_succeeded": False}
