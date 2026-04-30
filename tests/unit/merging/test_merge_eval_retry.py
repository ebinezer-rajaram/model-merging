from __future__ import annotations

from pathlib import Path

from merging.evaluation import evaluate as merge_evaluate


def test_cleanup_runs_after_each_task_attempt(monkeypatch):
    cleanup_calls = {"count": 0}

    def fake_cleanup(show_summary: bool = False):
        cleanup_calls["count"] += 1

    monkeypatch.setattr(merge_evaluate, "_cleanup_cuda_memory", fake_cleanup)

    def evaluate_once(task: str):
        return {"accuracy": 1.0, "task": task}

    results, task_status, failed_tasks = merge_evaluate._evaluate_tasks_with_retry(
        tasks_to_eval=["emotion", "intent"],
        split="test",
        show_summary=False,
        eval_subset=None,
        evaluate_once=evaluate_once,
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
    assert task_status["intent"] == {"retry_attempted": False, "retry_succeeded": False}


def test_oom_triggers_single_retry(monkeypatch):
    retry_calls = {"count": 0}

    def fake_cleanup(show_summary: bool = False):
        return None

    def evaluate_once(task: str):
        raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB.")

    def fake_retry(**kwargs):
        retry_calls["count"] += 1
        return {"accuracy": 0.9}, None

    monkeypatch.setattr(merge_evaluate, "_cleanup_cuda_memory", fake_cleanup)
    monkeypatch.setattr(merge_evaluate, "_retry_task_in_subprocess", fake_retry)

    results, task_status, failed_tasks = merge_evaluate._evaluate_tasks_with_retry(
        tasks_to_eval=["asr"],
        split="test",
        show_summary=False,
        eval_subset=None,
        evaluate_once=evaluate_once,
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


def test_non_oom_does_not_retry(monkeypatch):
    retry_calls = {"count": 0}

    def fake_cleanup(show_summary: bool = False):
        return None

    def evaluate_once(task: str):
        raise ValueError("some non-memory failure")

    def fake_retry(**kwargs):
        retry_calls["count"] += 1
        return {"accuracy": 1.0}, None

    monkeypatch.setattr(merge_evaluate, "_cleanup_cuda_memory", fake_cleanup)
    monkeypatch.setattr(merge_evaluate, "_retry_task_in_subprocess", fake_retry)

    results, task_status, failed_tasks = merge_evaluate._evaluate_tasks_with_retry(
        tasks_to_eval=["speech_qa"],
        split="test",
        show_summary=False,
        eval_subset=None,
        evaluate_once=evaluate_once,
        allow_oom_retry=True,
        retry_adapter_path=Path("/tmp/fake_run"),
        batch_size=None,
        generate_confusion_matrix=False,
        compute_missing_interference_baselines=True,
    )

    assert retry_calls["count"] == 0
    assert failed_tasks == ["speech_qa"]
    assert "error" in results["speech_qa"]
    assert task_status["speech_qa"] == {"retry_attempted": False, "retry_succeeded": False}
