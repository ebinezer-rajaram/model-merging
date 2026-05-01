from __future__ import annotations

import json
from pathlib import Path

from core.training.run_manager import RunManager, migrate_final_directory


def test_run_manager_workflow_updates_registry_links_and_cleans_old_runs(tmp_path: Path) -> None:
    manager = RunManager(tmp_path / "adapter", metric_for_ranking="accuracy", greater_is_better=True)
    first = manager.runs_dir / "run_1"
    first.mkdir(parents=True)
    manager.register_run(first, {"accuracy": 0.5}, {"training": {}, "model": {"lora": {}}})
    second = manager.runs_dir / "run_2"
    second.mkdir(parents=True)
    manager.register_run(second, {"accuracy": 0.7}, {"training": {}, "model": {"lora": {}}})
    third = manager.runs_dir / "run_3"
    third.mkdir(parents=True)
    manager.register_run(third, {"accuracy": 0.6}, {"training": {}, "model": {"lora": {}}})

    registry = json.loads((tmp_path / "adapter" / "runs_registry.json").read_text(encoding="utf-8"))
    kept_ids = {run["run_id"] for run in registry["runs"]}
    assert kept_ids == {first.name, second.name, third.name}
    assert manager.get_best_run_path() == second.resolve()
    assert manager.get_latest_run_path() == third.resolve()
    assert not first.exists()


def test_migrate_final_directory_creates_run_registry_and_links(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    final = adapter / "final"
    final.mkdir(parents=True)
    (final / "adapter_config.json").write_text("{}", encoding="utf-8")

    assert migrate_final_directory(adapter, metric_for_ranking="accuracy", greater_is_better=True) is True
    assert not final.exists()
    registry = json.loads((adapter / "runs_registry.json").read_text(encoding="utf-8"))
    assert registry["runs"][0]["status"] == "migrated"
    assert (adapter / "best").exists()
    assert migrate_final_directory(adapter, metric_for_ranking="accuracy", greater_is_better=True) is False
