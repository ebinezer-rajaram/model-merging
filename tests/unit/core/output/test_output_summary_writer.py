from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from core.output import summary_writer


def test_summary_writer_cleans_orders_and_extracts_delta_info() -> None:
    assert summary_writer._clean_metrics({"eval_accuracy": 0.5, "runtime": 1.0}) == {"accuracy": 0.5}
    assert summary_writer._order_splits(["z", "validation", "test"]) == ["test", "validation", "z"]
    assert summary_writer._primary_metric_name("asr") == "wer"
    assert summary_writer.build_selection("best", "accuracy", 0.9, "validation")["policy"] == "best"
    assert summary_writer.build_hyperparameters(learning_rate=1e-4, num_tasks=2)["num_tasks"] == 2
    assert summary_writer._extract_delta_info(
        {"interference_delta_meta": {"metric": "accuracy", "base": 0.1, "task_adapter": 0.2, "merged": 0.3}},
        {},
    ) == {"metric": "accuracy", "base": 0.1, "task_adapter": 0.2, "merged": 0.3}
    assert summary_writer._json_default(Path("x")) == "x"


def test_write_experiment_summary_annotates_results(monkeypatch, tmp_path: Path) -> None:
    results_mod = types.ModuleType("core.results")
    utils_mod = types.ModuleType("core.results.utils")
    utils_mod.derive_eval_context = lambda task, source_tasks, eval_tag: f"{task}:{eval_tag or 'none'}"
    monkeypatch.setitem(sys.modules, "core.results", results_mod)
    monkeypatch.setitem(sys.modules, "core.results.utils", utils_mod)

    output = tmp_path / "experiment_summary.json"
    summary_writer.write_experiment_summary(
        output_path=output,
        experiment_type="merge",
        run_id="run1",
        timestamp="2026-01-01T00:00:00",
        config_name="cfg",
        source_tasks=["asr"],
        method="weighted",
        results={"asr": {"test": {"eval_wer": 0.1, "eval_tag": "heldout", "eval_runtime": 9.0}}},
        source_files=["metrics.json"],
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "2"
    assert payload["source_tasks_key"] == "asr"
    assert payload["results"]["asr"]["test"]["wer"] == 0.1
    assert payload["results"]["asr"]["test"]["eval_context"] == "asr:heldout"
    assert payload["source_files"] == ["metrics.json"]


def test_write_experiment_summary_is_best_effort(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(summary_writer, "_write", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    summary_writer.write_experiment_summary(
        output_path=tmp_path / "summary.json",
        experiment_type="single_task",
        run_id=None,
        timestamp=None,
        config_name=None,
        source_tasks=[],
        method="asr",
        results={},
    )
